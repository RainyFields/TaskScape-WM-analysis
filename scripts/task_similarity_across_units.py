
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Level Task Similarity (RSA) using Split-Half Cross-Correlation.

Levels of Analysis:
  1. Condition: Raw conditions from the pickle (e.g., nback_loc_***f1**f2***)
  2. Task: The 11 detailed tasks (e.g., nback_loc)
  3. Group: Broad domains (nback, ctxdm, interdms)

Methodology:
  For each level and each bootstrap iteration:
  1) Aggregate trials by label (Condition/Task/Group).
  2) Split trials into Half 1 and Half 2.
  3) Compute Mean Patterns M1 and M2.
  4) Cross-Correlate: R = Corr(M1, M2).
  5) Diagonal = Reliability; Off-diagonal = Similarity.

Outputs:
  Saved per level (condition/task/group) and unit (voxel/region/network).
  - <subj>_<level>_similarity_mean_matrix_per_<unit>.npy
  - <subj>_<level>_names_per_<unit>.json
  - <subj>_<level>_similarity_heatmap_per_<unit>.png
  - ... (stats and boxplots)


Usage:
    python task_similarity_across_units.py \
        --subject sub-03 \
        --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-03_task_condition_betas.pkl \
        --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
        --unit region \
        --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/task_similarity \
        --network_assignments_path \
        /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt
"""

import argparse
import json
import pickle
from pathlib import Path
import re

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from utils import load_valid_vertex_mask, apply_mask_to_betas_dict

# ---------------------------------------------------------
# Definitions of Hierarchy
# ---------------------------------------------------------

# Level 2: Task (11 items)
TASKS_DETAILED = [
    "nback_loc", "nback_ctg", "nback_obj",
    "ctxdm_col", "ctxdm_lco",
    "interdms_loc_ABAB", "interdms_loc_ABBA",
    "interdms_obj_ABAB", "interdms_obj_ABBA",
    "interdms_ctg_ABAB", "interdms_ctg_ABBA",
]

# Level 3: Group (3 items)
GROUPS_BROAD = [
    "nback",
    "ctxdm",
    "interdms"
]

# ---------------------------------------------------------
# Plotting Style
# ---------------------------------------------------------
def setup_plotting_style():
    """Sets publication-ready plotting style with safe font fallbacks."""
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        # Try Times New Roman first, then fall back to other standard serif fonts
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'figure.titlesize': 24,
        'lines.linewidth': 2,
        'image.cmap': 'RdBu_r'
    })

# ---------------------------------------------------------
# Atlas & Helper Functions
# ---------------------------------------------------------
def load_glasser_labels(atlas_path: str):
    img = nib.load(atlas_path)
    lab = np.asarray(img.get_fdata()).squeeze().reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids

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
    """Z-score each row independently across units."""
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

def corr_matrix_rowwise(A, B, eps=1e-8):
    """Pearson correlation between rows of A and B."""
    A0 = A - np.nanmean(A, axis=1, keepdims=True)
    B0 = B - np.nanmean(B, axis=1, keepdims=True)
    Astd = np.nanstd(A0, axis=1, keepdims=True)
    Bstd = np.nanstd(B0, axis=1, keepdims=True)
    A0 = np.nan_to_num(A0 / (Astd + eps), nan=0.0)
    B0 = np.nan_to_num(B0 / (Bstd + eps), nan=0.0)
    return (A0 @ B0.T) / A.shape[1]

# ---------------------------------------------------------
# Grouping Logic
# ---------------------------------------------------------
def get_label_for_level(raw_condition: str, level: str):
    """
    Returns the label for a given hierarchy level.
    Returns None if the condition does not fit the level's schema.
    """
    if level == "condition":
        # Use the raw condition name, but we might want to filter out 'rest' or weird ones if necessary
        return raw_condition

    elif level == "task":
        # Match against the 11 detailed tasks
        for t in TASKS_DETAILED:
            if raw_condition.startswith(t):
                return t
        return None

    elif level == "group":
        # Match against the 3 broad groups
        for g in GROUPS_BROAD:
            if raw_condition.startswith(g):
                return g
        return None

    else:
        raise ValueError(f"Unknown level: {level}")


# ---------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------
def run_analysis_for_level(
    level_name, 
    condition_betas, 
    unit_transformer, 
    n_units, 
    args, 
    outdir, 
    rng
):
    print(f"\n{'='*60}")
    print(f"Processing Level: {level_name.upper()}")
    print(f"{'='*60}")

    # 1. Aggregate Data by Label
    label_to_reps = {}
    
    for cond_name, reps_list in condition_betas.items():
        if not reps_list: 
            continue
            
        label = get_label_for_level(cond_name, level_name)
        if label is None:
            continue
            
        # Flatten and Transform
        reps = np.stack([np.asarray(v).reshape(-1) for v in reps_list], axis=0)
        
        if unit_transformer is not None:
            reps = unit_transformer(reps)
            
        if label not in label_to_reps:
            label_to_reps[label] = []
        label_to_reps[label].append(reps)

    # Concatenate lists within labels
    final_labels = []
    final_arrays = []
    
    sorted_keys = sorted(label_to_reps.keys())
    if level_name == "task":
        sorted_keys = [k for k in TASKS_DETAILED if k in label_to_reps]
    elif level_name == "group":
        sorted_keys = [k for k in GROUPS_BROAD if k in label_to_reps]

    for label in sorted_keys:
        R = np.vstack(label_to_reps[label])
        if R.shape[0] < 2:
            print(f"  ⚠️ Skipping {label}: only {R.shape[0]} samples (need >= 2).")
            continue
        final_labels.append(label)
        final_arrays.append(R)

    n_labels = len(final_labels)
    if n_labels < 2:
        print(f"  ⚠️ Not enough valid labels ({n_labels}) for {level_name} analysis. Skipping.")
        return

    print(f"  Included labels ({n_labels}): {final_labels}")

    # 2. Split-Half Bootstrap
    n_boot = args.n_bootstrap
    R_accum = np.zeros((n_labels, n_labels), dtype=np.float64)
    mask_accum = np.zeros((n_labels, n_labels), dtype=np.float64)
    
    within_samples = []
    between_samples = []
    
    for b in range(n_boot):
        M1 = np.full((n_labels, n_units), np.nan, dtype=np.float32)
        M2 = np.full((n_labels, n_units), np.nan, dtype=np.float32)
        
        valid_boot = True
        for i, R in enumerate(final_arrays):
            N_t = R.shape[0]
            perm = rng.permutation(N_t)
            mid = N_t // 2
            if mid < 1:
                valid_boot = False; break
            
            idx1, idx2 = perm[:mid], perm[mid:]
            M1[i, :] = np.nanmean(R[idx1, :], axis=0)
            M2[i, :] = np.nanmean(R[idx2, :], axis=0)
        
        if not valid_boot or np.isnan(M1).all() or np.isnan(M2).all():
            continue

        if args.zscore_within_units:
            M1 = zscore_rows(M1)
            M2 = zscore_rows(M2)

        R_cross = corr_matrix_rowwise(M1, M2)
        R_sym = 0.5 * (R_cross + R_cross.T)
        
        valid_mask = np.isfinite(R_sym)
        R_accum += np.nan_to_num(R_sym, nan=0.0)
        mask_accum += valid_mask.astype(float)
        
        diag_vals = np.diag(R_sym)
        off_vals = R_sym[~np.eye(n_labels, dtype=bool)]
        
        within_samples.append(diag_vals[np.isfinite(diag_vals)])
        between_samples.append(off_vals[np.isfinite(off_vals)])

    with np.errstate(invalid="ignore"):
        R_mean = R_accum / mask_accum
    
    within_cat = np.concatenate(within_samples) if within_samples else np.array([])
    between_cat = np.concatenate(between_samples) if between_samples else np.array([])
    
    # 3. Save Outputs
    prefix = f"{args.subject}_{level_name}"
    
    # Numpy/JSON Saves
    np.save(outdir / f"{prefix}_similarity_mean_matrix_per_{args.unit}.npy", R_mean)
    with open(outdir / f"{prefix}_names_per_{args.unit}.json", "w") as f:
        json.dump(final_labels, f, indent=2)
    np.save(outdir / f"{prefix}_within_similarity_samples_per_{args.unit}.npy", within_cat)
    np.save(outdir / f"{prefix}_between_similarity_samples_per_{args.unit}.npy", between_cat)

    # Heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(R_mean, vmin=-1, vmax=1, origin="upper", interpolation="nearest")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (r)", rotation=270, labelpad=20)
    plt.title(f"{args.subject} {level_name.capitalize()} Similarity\n(Unit: {args.unit})")
    
    if n_labels > 20:
        step = max(1, n_labels // 20)
        plt.xticks(np.arange(0, n_labels, step), final_labels[::step], rotation=90, fontsize=12)
        plt.yticks(np.arange(0, n_labels, step), final_labels[::step], fontsize=12)
    else:
        plt.xticks(range(n_labels), final_labels, rotation=90)
        plt.yticks(range(n_labels), final_labels)
        
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_similarity_heatmap_per_{args.unit}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"{prefix}_similarity_heatmap_per_{args.unit}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"saved heatmap to {outdir / f'{prefix}_similarity_heatmap_per_{args.unit}.png'}")

    # Boxplot
    if within_cat.size > 0 and between_cat.size > 0:
        t_res = ttest_ind(within_cat, between_cat, equal_var=False, nan_policy="omit")
        
        plt.figure(figsize=(8, 6))
        
        # CRITICAL FIXES:
        # 1. showfliers=False (Prevents crash on millions of outlier points in PDF)
        # 2. tick_labels instead of labels (Fixes deprecation warning)
        plt.boxplot(
            [within_cat, between_cat], 
            widths=0.6, 
            tick_labels=["Within\n(Reliability)", "Between\n(Similarity)"], 
            showmeans=True, 
            patch_artist=True,
            showfliers=False,  # <--- THIS FIXES THE CRASH
            boxprops=dict(facecolor='#D3D3D3', color='black'),
            medianprops=dict(color='red', linewidth=2)
        )
        
        plt.ylabel("Correlation (r)")
        plt.title(f"{level_name.capitalize()} Similarity ({args.unit})\n"
                  f"Δ={np.nanmean(within_cat)-np.nanmean(between_cat):.2f}, t={t_res.statistic:.2f}")
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(outdir / f"{prefix}_within_vs_between_box_per_{args.unit}.png", dpi=300, bbox_inches="tight")
        plt.savefig(outdir / f"{prefix}_within_vs_between_box_per_{args.unit}.pdf", format="pdf", bbox_inches="tight")
        plt.close()
        print(f"saved boxplot to {outdir / f'{prefix}_within_vs_between_box_per_{args.unit}.png'}")

        # Stats JSON
        stats = {
            "level": level_name,
            "subject": args.subject,
            "unit": args.unit,
            "n_labels": n_labels,
            "n_within_samples": len(within_cat),
            "n_between_samples": len(between_cat),
            "within_mean": float(np.nanmean(within_cat)),
            "between_mean": float(np.nanmean(between_cat)),
            "t_statistic": float(t_res.statistic),
            "p_value": float(t_res.pvalue)
        }
        with open(outdir / f"{prefix}_similarity_stats_per_{args.unit}.json", "w") as f:
            json.dump(stats, f, indent=2)

    print(f"  ✅ Results saved to {outdir}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="region")
    ap.add_argument("--atlas_path", help="Required for region/network")
    ap.add_argument("--network_assignments_path", help="Required for network")
    ap.add_argument("--n_bootstrap", type=int, default=100)
    ap.add_argument("--zscore_within_units", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--levels", nargs="+", default=["condition", "task", "group"],
                    help="Levels to analyze (default: condition task group)")
    
    # Valid vertex mask (excludes medial wall / zero-variance vertices)
    ap.add_argument("--apply_vertex_mask", type=int, default=1,
                    help="Apply valid vertex mask to exclude zero-variance vertices (default: 1)")
    ap.add_argument("--vertex_mask_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/valid_vertex_mask.npy",
                    help="Path to valid vertex mask")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    setup_plotting_style()

    # Load Valid Vertex Mask
    valid_mask = None
    if args.apply_vertex_mask:
        print(f"Loading valid vertex mask from {args.vertex_mask_path}...")
        valid_mask = load_valid_vertex_mask(args.vertex_mask_path)
        n_valid = np.sum(valid_mask)
        n_total = len(valid_mask)
        print(f"  Mask: {n_valid:,}/{n_total:,} vertices ({n_valid/n_total*100:.1f}% valid)")

    # Load Data
    with open(args.input_pickle, "rb") as f:
        condition_betas = pickle.load(f)
    
    # Apply valid vertex mask to betas
    if valid_mask is not None:
        condition_betas = apply_mask_to_betas_dict(condition_betas, valid_mask)

    # Determine Feature Size
    first_key = next(k for k, v in condition_betas.items() if v)
    n_features = np.asarray(condition_betas[first_key][0]).size
    print(f"Feature size: {n_features} (after mask)")

    # Setup Transformers
    unit_transformer = None
    n_units = n_features
    
    if args.unit != "voxel":
        if not args.atlas_path:
            raise ValueError("--atlas_path required for region/network units.")
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        
        # Apply mask to atlas labels
        if valid_mask is not None:
            labels_vec = labels_vec[valid_mask]
        
        if args.unit == "region":
            unit_transformer = make_region_averager(labels_vec, region_ids)
            n_units = len(region_ids)
        elif args.unit == "network":
            if not args.network_assignments_path:
                raise ValueError("--network_assignments_path required for network unit.")
            max_rid = int(region_ids.max())
            r2n = load_region_to_network_map(args.network_assignments_path, max_rid)
            # Chain transformers: Voxel -> Region -> Network
            reg_avg = make_region_averager(labels_vec, region_ids)
            net_avg = make_network_averager(region_ids, r2n)
            
            def voxel_to_net(X):
                return net_avg(reg_avg(X))
            
            unit_transformer = voxel_to_net
            n_units = 12

    print(f"Analysis Unit: {args.unit} (n={n_units})")

    # Run Analysis for each requested level
    for level in args.levels:
        try:
            run_analysis_for_level(
                level, 
                condition_betas, 
                unit_transformer, 
                n_units, 
                args, 
                outdir, 
                rng
            )
        except Exception as e:
            print(f"❌ Failed processing level {level}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()