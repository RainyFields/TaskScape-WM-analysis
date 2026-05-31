
#!/usr/bin/env python3
"""
=> Calculate task condition discriminability with flexible Spatial and Task resolutions.

USAGE EXAMPLES:

1. Region Level (Voxel resolution) - Task Group Level
   python script.py --subject sub-03 ... --grouping region --spatial_resolution voxel --task_resolution task_group

2. Network Level (Region resolution) - Raw Condition Level
   python script.py --subject sub-03 ... --grouping network --spatial_resolution region --task_resolution condition

3. Network Level (Voxel resolution) - Task Level
   python script.py --subject sub-03 ... --grouping network --spatial_resolution voxel --task_resolution task


python region_level_task_correlation.py \
  --subject sub-03 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-03_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/task_condition_similarity \
  --grouping network \
  --spatial_resolution region \
  --task_resolution task_group \
  --n_bootstrap 10
"""

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

# =============================================================================
#  CONFIG: VISUALIZATION STYLE (Helvetica Neue + Large Fonts)
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

# Large font sizes for readability
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 26

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def parse_condition_key(raw_key):
    """
    Automatically parses the raw key string to identify the Task (1 of 11)
    and the Group (1 of 3).
    """
    parts = raw_key.split('_')
    group_label = parts[0]  # 'ctxdm', 'nback', or 'interdms'
    
    task_label = raw_key # Default fallback
    
    if group_label == 'ctxdm':
        task_label = "_".join(parts[:2])
    elif group_label == 'nback':
        task_label = "_".join(parts[:2])
    elif group_label == 'interdms':
        task_label = "_".join(parts[:3])
        
    return task_label, group_label

def get_target_label(raw_key, resolution):
    """Returns the final label to use for aggregation based on resolution settings."""
    if resolution == "condition":
        return raw_key
    
    task_label, group_label = parse_condition_key(raw_key)
    
    if resolution == "task":
        return task_label
    elif resolution == "task_group":
        return group_label
        
    return raw_key

def load_glasser_labels(atlas_path: str):
    img = nib.load(atlas_path)
    # Flatten atlas to match voxel vector
    lab = np.asarray(img.get_fdata()).squeeze().reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0] # exclude background
    return lab, region_ids

def load_region_to_network_map(assign_path: str, max_region_id: int):
    if not assign_path:
        return None
    lines = Path(assign_path).read_text().strip().splitlines()
    arr = np.zeros(max_region_id + 1, dtype=int)
    for i, line in enumerate(lines, start=1):
        if i <= max_region_id:
            arr[i] = int(line.strip())
    return arr

def zscore_rows(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

def compute_correlation_matrix(A, B, eps=1e-8):
    """Pearson correlation between rows of A and B."""
    A_z = zscore_rows(A, eps)
    B_z = zscore_rows(B, eps)
    n = A.shape[1]
    return (A_z @ B_z.T) / n

def aggregate_data(raw_betas, resolution, min_reps):
    """
    Aggregates raw condition betas into the target resolution buckets.
    """
    aggregated = {}
    print(f"Aggregating data to resolution: {resolution.upper()}...")
    
    for raw_cond, mat_list in raw_betas.items():
        target_label = get_target_label(raw_cond, resolution)
        mat_stack = np.stack([np.asarray(x).reshape(-1) for x in mat_list])
        
        if target_label not in aggregated:
            aggregated[target_label] = []
        aggregated[target_label].append(mat_stack)
        
    final_data = {}
    valid_labels = []
    
    for label, blocks in aggregated.items():
        merged = np.concatenate(blocks, axis=0) 
        if merged.shape[0] >= min_reps:
            final_data[label] = merged
            valid_labels.append(label)
            
    valid_labels.sort()
    indexed_data = {i: final_data[lbl] for i, lbl in enumerate(valid_labels)}
    
    return indexed_data, valid_labels

# =============================================================================
#  MAIN SCRIPT
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--atlas_path", required=True)
    ap.add_argument("--network_assignments_path")
    
    ap.add_argument("--grouping", choices=["region", "network"], required=True,
                    help="Do you want a score per Region or per Network?")
    
    ap.add_argument("--spatial_resolution", choices=["voxel", "region"], required=True,
                    help="voxel: correlate voxel patterns. region: correlate averaged region patterns.")

    ap.add_argument("--task_resolution", choices=["condition", "task", "task_group"], default="condition",
                    help="condition: raw keys. task: 11 tasks. task_group: 3 main groups.")

    ap.add_argument("--n_bootstrap", type=int, default=100)
    ap.add_argument("--min_reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    
    # Validation
    if args.grouping == "region" and args.spatial_resolution == "region":
        print("Error: grouping='region' with spatial_resolution='region' is redundant/invalid.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    
    # -------------------------------------------------------------------------
    # 0. Setup Output Directory Automatically
    # -------------------------------------------------------------------------
    # Automatically append the grouping level to the output path
    subdir_name = f"{args.grouping}_level"
    outdir = Path(args.output_dir) / subdir_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {outdir}")

    # -------------------------------------------------------------------------
    # 1. Load Data & Atlas
    # -------------------------------------------------------------------------
    print(f"Loading betas from {args.input_pickle}...")
    with open(args.input_pickle, "rb") as f:
        raw_condition_betas = pickle.load(f)

    data_source_voxels, label_names = aggregate_data(raw_condition_betas, args.task_resolution, args.min_reps)
    C = len(data_source_voxels)
    
    if C < 2:
        print(f"Error: Only {C} conditions found. Need at least 2.")
        sys.exit(1)

    print(f"Loading atlas: {args.atlas_path}...")
    voxel_labels, region_ids = load_glasser_labels(args.atlas_path)
    
    n_features_total = data_source_voxels[0].shape[1]
    if voxel_labels.size != n_features_total:
        sys.exit(f"Atlas size ({voxel_labels.size}) != data features ({n_features_total})")

    # -------------------------------------------------------------------------
    # 2. Prepare Feature Selection
    # -------------------------------------------------------------------------
    final_data_source = {}   
    group_indices = {}       
    
    if args.spatial_resolution == "voxel":
        print(f"Spatial Resolution: VOXEL. Using raw voxel patterns.")
        final_data_source = data_source_voxels
        
        if args.grouping == "region":
            for rid in region_ids:
                group_indices[rid] = np.where(voxel_labels == rid)[0]
                
        elif args.grouping == "network":
            if not args.network_assignments_path:
                sys.exit("Network grouping requires --network_assignments_path")
            
            max_reg = int(region_ids.max())
            reg2net = load_region_to_network_map(args.network_assignments_path, max_reg)
            unique_networks = np.arange(1, 13)
            
            for nid in unique_networks:
                regions_in_net = [r for r in region_ids if (r < len(reg2net) and reg2net[r] == nid)]
                vox_indices = []
                for r in regions_in_net:
                    vox_indices.append(np.where(voxel_labels == r)[0])
                if vox_indices:
                    group_indices[nid] = np.concatenate(vox_indices)

    elif args.spatial_resolution == "region":
        print(f"Spatial Resolution: REGION. Averaging voxels -> regions first.")
        
        r_id_to_col = {rid: i for i, rid in enumerate(region_ids)}
        n_regions = len(region_ids)
        region_voxel_map = {rid: np.where(voxel_labels == rid)[0] for rid in region_ids}

        data_source_regions = {}
        for c_idx, mat_vox in data_source_voxels.items():
            n_reps = mat_vox.shape[0]
            mat_reg = np.zeros((n_reps, n_regions), dtype=np.float32)
            for rid in region_ids:
                v_idx = region_voxel_map[rid]
                if len(v_idx) > 0:
                    mat_reg[:, r_id_to_col[rid]] = np.mean(mat_vox[:, v_idx], axis=1)
                else:
                    mat_reg[:, r_id_to_col[rid]] = np.nan
            data_source_regions[c_idx] = mat_reg
        
        final_data_source = data_source_regions

        if args.grouping == "network":
            if not args.network_assignments_path:
                sys.exit("Network grouping requires --network_assignments_path")
            max_reg = int(region_ids.max())
            reg2net = load_region_to_network_map(args.network_assignments_path, max_reg)
            unique_networks = np.arange(1, 13)
            for nid in unique_networks:
                regions_in_net = [r for r in region_ids if (r < len(reg2net) and reg2net[r] == nid)]
                feature_indices = [r_id_to_col[r] for r in regions_in_net]
                if feature_indices:
                    group_indices[nid] = np.array(feature_indices)

    active_groups = [g for g in group_indices if len(group_indices[g]) >= 5]
    print(f"Processing {len(active_groups)} groups ({args.grouping}s)...")

    if args.grouping == "network":
        # Force 1 to 12 for networks to ensure output size is 12
        target_groups = np.arange(1, 13)
    else:
        # For regions, use all valid regions found in the atlas
        target_groups = np.sort(list(group_indices.keys()))

    print(f"Processing {len(target_groups)} groups ({args.grouping}s)...")

    # -------------------------------------------------------------------------
    # 3. Bootstrap RSA Loop
    # -------------------------------------------------------------------------
    similarity_scores = []
    accum_R_mean = np.zeros((C, C), dtype=np.float64)
    accum_count = 0

    # Pre-generate bootstrap splits
    bootstrap_splits = []
    for b in range(args.n_bootstrap):
        split_map = {}
        for c_idx in range(C):
            n_reps = final_data_source[c_idx].shape[0]
            perm = rng.permutation(n_reps)
            mid = n_reps // 2
            split_map[c_idx] = (perm[:mid], perm[mid:])
        bootstrap_splits.append(split_map)

    # Loop through ALL target groups (1-12)
    for gid in tqdm(target_groups):
        
        # --- CHECK: Do we have enough data for this group? ---
        # If the group is missing or has < 5 features, fill with NaN
        if (gid not in group_indices) or (len(group_indices[gid]) < 5):
            similarity_scores.append(np.nan)
            continue 
        
        # --- If valid, proceed with calculation ---
        feat_idx = group_indices[gid] 
        R_group_accum = np.zeros((C, C), dtype=np.float64)
        
        for b in range(args.n_bootstrap):
            splits = bootstrap_splits[b]
            vecs_A, vecs_B = [], []
            for c_idx in range(C):
                raw = final_data_source[c_idx][:, feat_idx]
                idx_a, idx_b = splits[c_idx]
                if len(idx_a) > 0 and len(idx_b) > 0:
                    vecs_A.append(np.mean(raw[idx_a], axis=0))
                    vecs_B.append(np.mean(raw[idx_b], axis=0))
                else:
                    nan_vec = np.zeros(len(feat_idx)) * np.nan
                    vecs_A.append(nan_vec)
                    vecs_B.append(nan_vec)

            Mat_A = np.stack(vecs_A)
            Mat_B = np.stack(vecs_B)

            if np.isnan(Mat_A).any():
                R = np.full((C,C), np.nan)
            else:
                R = compute_correlation_matrix(Mat_A, Mat_B)
            R_group_accum += R

        R_group_mean = R_group_accum / args.n_bootstrap
        off_diag_vals = R_group_mean[~np.eye(C, dtype=bool)]
        similarity_scores.append(np.nanmean(off_diag_vals))
        
        accum_R_mean += np.nan_to_num(R_group_mean, 0.0)
        accum_count += 1

    # -------------------------------------------------------------------------
    # 4. Save Outputs
    # -------------------------------------------------------------------------
    scores_arr = np.array(similarity_scores)
    group_ids_arr = np.array(target_groups)
    avg_R_global = accum_R_mean / accum_count if accum_count > 0 else np.zeros((C,C))

    base_name = f"{args.subject}_{args.grouping}_{args.spatial_resolution}_{args.task_resolution}"
    
    # Save NPY
    npy_path = outdir / f"{base_name}_scores.npy"
    np.save(npy_path, scores_arr)
    np.save(outdir / f"{base_name}_ids.npy", group_ids_arr)

    # Plot 1: Distribution
    plt.figure(figsize=(12, 10))
    plt.hist(scores_arr, bins=30, color='mediumpurple', edgecolor='black', alpha=0.8)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    mean_val = np.nanmean(scores_arr)
    plt.axvline(mean_val, color='red', linestyle='-', linewidth=4, label=f"Mean: {mean_val:.2f}")
    
    plt.title(f"Similarity Dist: {args.subject}\nGrouping: {args.grouping.upper()}")
    plt.xlabel(f"Similarity Score")
    plt.ylabel(f"Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(outdir / f"{base_name}_dist.png", dpi=300)
    print(f"Saving distribution plot to: {outdir / f'{base_name}_dist.png'}")
    plt.savefig(outdir / f"{base_name}_dist.pdf")
    print(f"Saving distribution plot to: {outdir / f'{base_name}_dist.pdf'}")
    plt.close()

    # Plot 2: Heatmap
    plt.figure(figsize=(14, 12)) 
    im = plt.imshow(avg_R_global, vmin=-0.5, vmax=0.5, origin="upper", cmap='RdBu_r')
    cbar = plt.colorbar(im, label="Pearson r", fraction=0.046, pad=0.04)
    
    plt.title(f"Avg Correlation Matrix\nGrouping: {args.grouping.upper()}")
    
    if C > 50:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"Conditions (N={C})")
        plt.ylabel(f"Conditions (N={C})")
    else:
        plt.xticks(range(C), label_names, rotation=90)
        plt.yticks(range(C), label_names)
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(outdir / f"{base_name}_heatmap.png", dpi=300)
    print(f"Saving heatmap plot to: {outdir / f'{base_name}_heatmap.png'}")
    plt.savefig(outdir / f"{base_name}_heatmap.pdf")
    print(f"Saving heatmap plot to: {outdir / f'{base_name}_heatmap.pdf'}")
    plt.close()

    print("\n" + "="*50)
    print(" PROCESSING COMPLETE")
    print("="*50)
    print(f"✅ Output Dir: {outdir}")
    print(f"✅ Saved .npy scores and IDs")
    print(f"✅ Saved .png and .pdf figures")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()