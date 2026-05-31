#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-subject RSA using Split-Half Correlation Matrices.
Supports: Voxel, Region, Network levels.
Supports Analysis Levels: Condition (raw), Task (11 groups), Broad Group (3 groups).

Memory Optimized: Uses a two-pass approach to identify common keys without holding data in RAM.
Output: PNG and PDF figures, Helvetica font.
usage:

# permute unit in voxel / region / network
# permute level in condition / task / group
# example command:

python subject_within_vs_cross_similarity.py \
    --subjects sub-01 sub-02 sub-03 sub-05 sub-06 \
    --input_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only \
    --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/task_rsa_cross_subject \
    --unit voxel \
    --level condition \
    --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
    --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii
"""

import argparse
import json
import pickle
import gc
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from utils import load_valid_vertex_mask, apply_mask_to_betas_dict

# ---------------------------------------------------------
# Analysis Level Definitions
# ---------------------------------------------------------
# Level: 'task' (The original granular groups)
LEVEL_TASK = [
    "nback_loc", "nback_ctg", "nback_obj",
    "ctxdm_col", "ctxdm_lco",
    "interdms_loc_ABAB", "interdms_loc_ABBA",
    "interdms_obj_ABAB", "interdms_obj_ABBA",
    "interdms_ctg_ABAB", "interdms_ctg_ABBA",
]

# Level: 'group' (Broad categories)
LEVEL_GROUP = [
    "nback", 
    "ctxdm", 
    "interdms"
]

# Level: 'condition' is dynamic (determined from data files)

# ---------------------------------------------------------
# Atlas & Math Helpers
# ---------------------------------------------------------
def load_glasser_labels(atlas_path: str):
    img = nib.load(atlas_path)
    lab = np.asarray(img.get_fdata()).squeeze().reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    valid_mask = (lab > 0)
    return lab, region_ids, valid_mask

def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]
    def avg_by_region(A):
        out = np.full((A.shape[0], len(region_ids)), np.nan, dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx.size:
                out[:, j] = np.nanmean(A[:, idx], axis=1)
        return out
    return avg_by_region

def load_region_to_network_map(assign_path: str, max_region_id: int):
    lines = Path(assign_path).read_text().strip().splitlines()
    arr = np.zeros(max_region_id + 1, dtype=int)
    for i, line in enumerate(lines, start=1):
        if i <= max_region_id:
            arr[i] = int(line.strip())
    return arr

def make_network_averager(region_ids: np.ndarray, region_to_net: np.ndarray, n_networks: int = 12):
    nets = [[] for _ in range(n_networks)]
    for j, rid in enumerate(region_ids):
        nid = region_to_net[rid]
        if 1 <= nid <= n_networks:
            nets[nid - 1].append(j)
    def avg_by_network(R):
        out = np.full((R.shape[0], n_networks), np.nan, dtype=np.float32)
        for k, cols in enumerate(nets):
            if cols:
                out[:, k] = np.nanmean(R[:, cols], axis=1)
        return out
    return avg_by_network

def zscore_rows(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

def corr_matrix_rowwise(A, B, eps=1e-8):
    A0 = A - np.nanmean(A, axis=1, keepdims=True)
    B0 = B - np.nanmean(B, axis=1, keepdims=True)
    Astd = np.nanstd(A0, axis=1, keepdims=True)
    Bstd = np.nanstd(B0, axis=1, keepdims=True)
    A0 = np.nan_to_num(A0 / (Astd + eps), nan=0.0)
    B0 = np.nan_to_num(B0 / (Bstd + eps), nan=0.0)
    n_feats = A.shape[1]
    return (A0 @ B0.T) / n_feats

def vectorize_upper_triangle(M):
    T = M.shape[0]
    idx = np.triu_indices(T, k=1)
    return M[idx]

def get_mapped_condition(raw_key, level, defined_list):
    """
    Determines how to group a raw key based on the analysis level.
    """
    if level == 'condition':
        return raw_key
    elif level in ['task', 'group']:
        # Match start of string to defined categories
        for cat in defined_list:
            if raw_key.startswith(cat):
                return cat
    return None

# ---------------------------------------------------------
# Core Logic
# ---------------------------------------------------------
def main():
    # --- 1. PLOTTING CONFIG ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 24
    })

    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_dir", required=True)
    
    # Unit options
    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="region")
    
    # Analysis Level options
    ap.add_argument("--level", choices=["condition", "task", "group"], default="task",
                    help="Level of analysis: 'condition' (raw keys), 'task' (11 groups), 'group' (3 broad groups)")
    
    ap.add_argument("--atlas_path", required=True)
    ap.add_argument("--network_assignments_path")
    ap.add_argument("--n_bootstrap", type=int, default=10)
    ap.add_argument("--zscore_within_units", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    
    # Valid vertex mask (excludes medial wall / zero-variance vertices)
    ap.add_argument("--apply_vertex_mask", type=int, default=1,
                    help="Apply valid vertex mask to exclude zero-variance vertices (default: 1)")
    ap.add_argument("--vertex_mask_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/valid_vertex_mask.npy",
                    help="Path to valid vertex mask")
    
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {outdir}")

    # Determine Mapping List based on arguments
    target_definitions = []
    if args.level == 'task':
        target_definitions = LEVEL_TASK
        print(f"Analysis Level: TASK (Looking for {len(target_definitions)} categories)")
    elif args.level == 'group':
        target_definitions = LEVEL_GROUP
        print(f"Analysis Level: GROUP (Looking for {len(target_definitions)} broad categories)")
    elif args.level == 'condition':
        print(f"Analysis Level: CONDITION (Using raw keys from data)")

    # -----------------------------------------------------------
    # PRE-SCAN: Find Intersection of Common Valid Tasks
    # -----------------------------------------------------------
    print("\n--- Phase 1: Scanning files for common valid keys ---")
    common_keys = None
    
    for subj in args.subjects:
        pkl_path = Path(args.input_root) / f"{subj}_task_condition_betas.pkl"
        if not pkl_path.exists():
            print(f"Skipping {subj} (File not found)")
            continue
            
        with open(pkl_path, "rb") as f:
            data_raw = pickle.load(f)
            
        subj_keys = set()
        
        if args.level == 'condition':
            for k, v in data_raw.items():
                if v and len(v) >= 2: # Need at least 2 reps
                    subj_keys.add(k)
        else:
            # Grouping logic
            group_counts = defaultdict(int)
            for k, v in data_raw.items():
                mapped = get_mapped_condition(k, args.level, target_definitions)
                if mapped and v:
                    group_counts[mapped] += len(v)
            
            for g, count in group_counts.items():
                if count >= 2:
                    subj_keys.add(g)

        if common_keys is None:
            common_keys = subj_keys
            print(f"[{subj}] Found {len(subj_keys)} valid items.")
        else:
            common_keys = common_keys.intersection(subj_keys)
            print(f"[{subj}] Intersection size: {len(common_keys)}")
            
        del data_raw
        gc.collect()

    if not common_keys or len(common_keys) < 2:
        print(f"Error: Less than 2 common items found across subjects. Found: {common_keys}")
        sys.exit(1)
        
    final_task_list = sorted(list(common_keys))
    T = len(final_task_list)
    print(f"\nFinal Common List ({T} items): {final_task_list[:5]} ...")

    # -----------------------------------------------------------
    # LOAD VALID VERTEX MASK
    # -----------------------------------------------------------
    vertex_mask = None
    if args.apply_vertex_mask:
        print(f"Loading valid vertex mask from {args.vertex_mask_path}...")
        vertex_mask = load_valid_vertex_mask(args.vertex_mask_path)
        n_valid = np.sum(vertex_mask)
        n_total = len(vertex_mask)
        print(f"  Mask: {n_valid:,}/{n_total:,} vertices ({n_valid/n_total*100:.1f}% valid)")
    
    # -----------------------------------------------------------
    # PREPARE ATLAS
    # -----------------------------------------------------------
    labels_vec, region_ids, atlas_valid_mask = load_glasser_labels(args.atlas_path)
    
    # Use our vertex mask if available, otherwise fall back to atlas valid mask
    if vertex_mask is not None:
        valid_mask = vertex_mask
        labels_vec_masked = labels_vec[vertex_mask]
    else:
        valid_mask = atlas_valid_mask
        labels_vec_masked = labels_vec[atlas_valid_mask]
    
    avg_by_region = None
    avg_by_network = None
    
    if args.unit == "voxel":
        n_units = int(np.sum(valid_mask))
    elif args.unit == "region":
        avg_by_region = make_region_averager(labels_vec_masked, region_ids)
        n_units = len(region_ids)
    elif args.unit == "network":
        if not args.network_assignments_path:
            raise SystemExit("Need --network_assignments_path")
        avg_by_region = make_region_averager(labels_vec_masked, region_ids)
        max_id = int(region_ids.max())
        reg2net = load_region_to_network_map(args.network_assignments_path, max_id)
        avg_by_network = make_network_averager(region_ids, reg2net)
        n_units = 12
    
    print(f"Unit: {args.unit} | Features: {n_units}")

    # -----------------------------------------------------------
    # MAIN LOOP: Compute RSA
    # -----------------------------------------------------------
    print("\n--- Phase 2: Computing RSA Matrices ---")
    subj_rsa_vectors = {}

    for subj in args.subjects:
        pkl_path = Path(args.input_root) / f"{subj}_task_condition_betas.pkl"
        if not pkl_path.exists(): continue
        
        print(f"Processing {subj} ...")
        with open(pkl_path, "rb") as f:
            data_raw = pickle.load(f)
            
        # Aggregation
        subject_task_matrices = {}
        
        for cond, reps in data_raw.items():
            if not reps: continue
            
            target = get_mapped_condition(cond, args.level, target_definitions if args.level != 'condition' else [])
            
            if target not in final_task_list:
                continue

            mat = np.stack([np.asarray(v).reshape(-1) for v in reps])
            
            # Apply vertex mask first (for all units)
            mat = mat[:, valid_mask]
            
            if args.unit == "voxel":
                pass  # Already masked
            elif args.unit == "region":
                mat = avg_by_region(mat)
            elif args.unit == "network":
                mat = avg_by_region(mat)
                mat = avg_by_network(mat)
            
            if target not in subject_task_matrices:
                subject_task_matrices[target] = []
            subject_task_matrices[target].append(mat)

        # Collapse blocks
        final_tasks = {}
        for t, blocks in subject_task_matrices.items():
            if blocks:
                R = np.vstack(blocks)
                if R.shape[0] >= 2:
                    final_tasks[t] = R
        
        del data_raw, subject_task_matrices

        # Calculate Split-Half
        vectors = []
        task_arrays = [final_tasks[t] for t in final_task_list] # Order matters!
        
        for b in range(args.n_bootstrap):
            M1 = np.full((T, n_units), np.nan, dtype=np.float32)
            M2 = np.full((T, n_units), np.nan, dtype=np.float32)
            
            valid_boot = True
            for i, R in enumerate(task_arrays):
                nt = R.shape[0]
                perm = rng.permutation(nt)
                mid = nt // 2
                
                m1 = np.nanmean(R[perm[:mid]], axis=0)
                m2 = np.nanmean(R[perm[mid:]], axis=0)
                
                if np.isnan(m1).any() or np.isnan(m2).any():
                    valid_boot = False; break
                M1[i] = m1
                M2[i] = m2
            
            if not valid_boot: continue
                
            if args.zscore_within_units:
                M1 = zscore_rows(M1)
                M2 = zscore_rows(M2)
            
            R_cross = corr_matrix_rowwise(M1, M2)
            R_sym = 0.5 * (R_cross + R_cross.T)
            vectors.append(vectorize_upper_triangle(R_sym))
            
        if len(vectors) < 2:
            print(f"  -> Warning: Not enough valid bootstraps for {subj}")
        else:
            subj_rsa_vectors[subj] = np.stack(vectors)
            print(f"  -> Success: {len(vectors)} bootstraps.")
            
        del final_tasks, task_arrays, vectors
        gc.collect()

    if not subj_rsa_vectors:
        sys.exit("No valid subjects processed.")

    # -----------------------------------------------------------
    # STATISTICAL ANALYSIS
    # -----------------------------------------------------------
    print("\nRunning Similarity Analysis...")
    within_corrs = []
    cross_corrs = []
    subjects = sorted(list(subj_rsa_vectors.keys()))
    
    # Within-Subject
    for subj in subjects:
        vecs = subj_rsa_vectors[subj]
        nb = vecs.shape[0]
        for i in range(nb):
            for j in range(i+1, nb):
                r = np.corrcoef(vecs[i], vecs[j])[0,1]
                if np.isfinite(r): within_corrs.append(r)
                    
    # Cross-Subject
    for i, s1 in enumerate(subjects):
        for j in range(i+1, len(subjects)):
            s2 = subjects[j]
            vecs1 = subj_rsa_vectors[s1]
            vecs2 = subj_rsa_vectors[s2]
            for v1 in vecs1:
                for v2 in vecs2:
                    r = np.corrcoef(v1, v2)[0,1]
                    if np.isfinite(r): cross_corrs.append(r)

    within_corrs = np.array(within_corrs)
    cross_corrs = np.array(cross_corrs)
    
    print(f"Within Mean={np.nanmean(within_corrs):.3f}")
    print(f"Cross Mean ={np.nanmean(cross_corrs):.3f}")

    # Save Arrays
    f_suffix = f"{args.unit}_{args.level}"
    np.save(outdir / f"within_subject_similarity_{f_suffix}.npy", within_corrs)
    print(f"Saved: {outdir / f'within_subject_similarity_{f_suffix}.npy'}")
    
    np.save(outdir / f"cross_subject_similarity_{f_suffix}.npy", cross_corrs)
    print(f"Saved: {outdir / f'cross_subject_similarity_{f_suffix}.npy'}")
    
    # -----------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------
    plt.figure(figsize=(8, 7)) 
    
    box = plt.boxplot([within_corrs, cross_corrs], 
                      labels=["Within-Subject", "Cross-Subject"], 
                      showmeans=True, 
                      patch_artist=True,
                      medianprops=dict(linewidth=2.5, color='black'),
                      boxprops=dict(linewidth=2),
                      whiskerprops=dict(linewidth=2),
                      capprops=dict(linewidth=2))
    
    colors = ['#AEC7E8', '#FFBB78'] 
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel(f"RSA Similarity (r)")
    plt.ylim(0, 1.1)
    plt.title(f"RSA Consistency ({args.unit.upper()} | {args.level.upper()})\n(Split-Half Matrices)")
    
    plt.grid(True, alpha=0.3, linewidth=1.5)
    plt.tight_layout()
    
    base_name = f"within_vs_cross_subject_box_{f_suffix}"
    
    png_path = outdir / f"{base_name}.png"
    pdf_path = outdir / f"{base_name}.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"Plots saved to:\n  {png_path}\n  {pdf_path}")
    plt.close()
    
    stats = {
        "unit": args.unit,
        "level": args.level,
        "subjects": subjects,
        "n_common_items": T,
        "within_mean": float(np.nanmean(within_corrs)),
        "cross_mean": float(np.nanmean(cross_corrs))
    }
    with open(outdir / f"summary_stats_{f_suffix}.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("✅ Done!")

if __name__ == "__main__":
    main()