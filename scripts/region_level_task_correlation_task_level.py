#!/usr/bin/env python3
"""
=> to calculate task level (not task condition!) discriminability at region or network level using RSA.



For each defined group (Region or Network):
  1. Select voxels.
  2. Perform Split-Half Bootstrap RSA.
  3. Compute "Similarity Score" = Mean(Off-Diagonal Correlations).
     - High Score: Region treats conditions similarly (Generalized).
     - Low Score: Region treats conditions distinctly (Specialized).

Outputs:
  - NPY: Array of similarity scores (one per group).
  - PNG: Histogram of similarity scores (distribution of task similarity across the brain).
  - PNG: Average Heatmap (Global structure).

Example:

python region_level_task_correlation_task_level.py \
  --subject sub-03 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-03_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/task_similarity/network_level \
  --grouping network \
  --n_bootstrap 10
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

# --- DEFINED TARGET LIST ---
ALL_TASKS = [
    'nback_loc',
    'nback_ctg',
    'nback_obj',
    'ctxdm_col',
    'ctxdm_lco',
    'interdms_loc_ABAB',
    'interdms_loc_ABBA',
    'interdms_obj_ABAB',
    'interdms_obj_ABBA',
    'interdms_ctg_ABAB',
    'interdms_ctg_ABBA',
]

# def load_glasser_labels(atlas_path: str):
#     img = nib.load(atlas_path)
#     lab = np.asarray(img.get_fdata()).squeeze().reshape(-1).astype(int)
#     region_ids = np.unique(lab)
#     region_ids = region_ids[region_ids > 0] 
#     return lab, region_ids

# def load_region_to_network_map(assign_path: str, max_region_id: int):
#     lines = Path(assign_path).read_text().strip().splitlines()
#     arr = np.zeros(max_region_id + 1, dtype=int)
#     for i, line in enumerate(lines, start=1):
#         if i <= max_region_id:
#             arr[i] = int(line.strip())
#     return arr

# def zscore_rows(X, eps=1e-8):
#     X = np.asarray(X, dtype=np.float32)
#     mu = np.nanmean(X, axis=1, keepdims=True)
#     sd = np.nanstd(X, axis=1, keepdims=True)
#     return (X - mu) / (sd + eps)

# def compute_correlation_matrix(A, B, eps=1e-8):
#     """Pearson correlation between rows of A and B."""
#     A_z = zscore_rows(A, eps)
#     B_z = zscore_rows(B, eps)
#     n = A.shape[1]
#     return (A_z @ B_z.T) / n

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--subject", required=True)
#     ap.add_argument("--input_pickle", required=True)
#     ap.add_argument("--output_dir", required=True)
#     ap.add_argument("--atlas_path", required=True)
#     ap.add_argument("--grouping", choices=["region", "network"], default="region")
#     ap.add_argument("--network_assignments_path")
#     ap.add_argument("--n_bootstrap", type=int, default=100)
#     ap.add_argument("--min_reps", type=int, default=4)
#     ap.add_argument("--seed", type=int, default=0)

#     args = ap.parse_args()
#     rng = np.random.default_rng(args.seed)
#     outdir = Path(args.output_dir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     # 1. Load Data
#     print(f"Loading betas from {args.input_pickle}...")
#     with open(args.input_pickle, "rb") as f:
#         condition_betas = pickle.load(f)

#     # Filter data to only include the specific 11 tasks in ALL_TASKS
#     cond_data_voxels = {}
#     valid_cond_names = []
    
#     n_features_total = 0
    
#     print("Filtering for target conditions...")
#     for idx, task_name in enumerate(ALL_TASKS):
#         if task_name not in condition_betas:
#             print(f"  [MISSING] {task_name} not found in pickle.")
#             continue
            
#         # Get data
#         reps_list = condition_betas[task_name]
        
#         # Check reps
#         if len(reps_list) < args.min_reps:
#             print(f"  [SKIP] {task_name}: Not enough reps ({len(reps_list)} < {args.min_reps})")
#             continue
            
#         # Flatten and Store
#         mat = np.stack([np.asarray(x).reshape(-1) for x in reps_list])
#         cond_data_voxels[len(valid_cond_names)] = mat # Store using index 0..N
#         valid_cond_names.append(task_name)
#         n_features_total = mat.shape[1]

#     C = len(valid_cond_names)
#     print(f"Final matrix size: {C}x{C} conditions.")
    
#     if C < 2:
#         raise SystemExit("Not enough valid conditions to compute correlations.")

#     # 2. Load Atlas
#     print(f"Loading atlas: {args.atlas_path}...")
#     voxel_labels, region_ids = load_glasser_labels(args.atlas_path)
    
#     if voxel_labels.size != n_features_total:
#         raise SystemExit(f"Atlas size ({voxel_labels.size}) != data features ({n_features_total})")

#     # 3. Prepare Data Source and Groups
#     data_source = {}   
#     group_indices = {} 

#     if args.grouping == "region":
#         print("Grouping: REGION. Unit: VOXEL.")
#         data_source = cond_data_voxels
#         for rid in region_ids:
#             group_indices[rid] = np.where(voxel_labels == rid)[0]

#     elif args.grouping == "network":
#         print("Grouping: NETWORK. Unit: REGION (Aggregating voxels...).")
#         if not args.network_assignments_path:
#             raise SystemExit("--network_assignments_path required for network grouping")

#         # --- A. Create Region-Level Data ---
#         r_id_to_col = {rid: i for i, rid in enumerate(region_ids)}
#         n_regions = len(region_ids)
#         region_voxel_map = {rid: np.where(voxel_labels == rid)[0] for rid in region_ids}

#         cond_data_regions = {}
#         for c_idx, mat_vox in cond_data_voxels.items():
#             n_reps = mat_vox.shape[0]
#             mat_reg = np.zeros((n_reps, n_regions), dtype=np.float32)
            
#             for rid in region_ids:
#                 v_idx = region_voxel_map[rid]
#                 if len(v_idx) > 0:
#                     mat_reg[:, r_id_to_col[rid]] = np.mean(mat_vox[:, v_idx], axis=1)
#                 else:
#                     mat_reg[:, r_id_to_col[rid]] = np.nan
            
#             cond_data_regions[c_idx] = mat_reg
        
#         data_source = cond_data_regions

#         # --- B. Define Groups (Networks) ---
#         max_reg = int(region_ids.max())
#         reg2net = load_region_to_network_map(args.network_assignments_path, max_reg)
#         unique_networks = np.arange(1, 13)

#         for nid in unique_networks:
#             regions_in_net = [r for r in region_ids if (r < len(reg2net) and reg2net[r] == nid)]
#             feature_indices = [r_id_to_col[r] for r in regions_in_net]
#             if feature_indices:
#                 group_indices[nid] = np.array(feature_indices)

#     active_groups = [g for g in group_indices if len(group_indices[g]) >= 5]
#     print(f"Processing {len(active_groups)} groups...")

#     # 4. Main Loop (Bootstrap RSA)
#     similarity_scores = []
#     accum_R_mean = np.zeros((C, C), dtype=np.float64)
#     accum_count = 0

#     bootstrap_splits = []
#     for b in range(args.n_bootstrap):
#         split_map = {}
#         for c_idx in range(C):
#             n_reps = data_source[c_idx].shape[0]
#             perm = rng.permutation(n_reps)
#             mid = n_reps // 2
#             split_map[c_idx] = (perm[:mid], perm[mid:])
#         bootstrap_splits.append(split_map)

#     for gid in tqdm(active_groups):
#         feat_idx = group_indices[gid] 
#         R_group_accum = np.zeros((C, C), dtype=np.float64)
        
#         for b in range(args.n_bootstrap):
#             splits = bootstrap_splits[b]
#             vecs_A, vecs_B = [], []
            
#             for c_idx in range(C):
#                 raw = data_source[c_idx][:, feat_idx]
#                 idx_a, idx_b = splits[c_idx]
#                 if len(idx_a) > 0 and len(idx_b) > 0:
#                     vecs_A.append(np.mean(raw[idx_a], axis=0))
#                     vecs_B.append(np.mean(raw[idx_b], axis=0))
#                 else:
#                     nan_vec = np.zeros(len(feat_idx)) * np.nan
#                     vecs_A.append(nan_vec)
#                     vecs_B.append(nan_vec)

#             Mat_A = np.stack(vecs_A)
#             Mat_B = np.stack(vecs_B)

#             if np.isnan(Mat_A).any():
#                 R = np.full((C,C), np.nan)
#             else:
#                 R = compute_correlation_matrix(Mat_A, Mat_B)
            
#             R_group_accum += R

#         R_group_mean = R_group_accum / args.n_bootstrap
        
#         off_diag_vals = R_group_mean[~np.eye(C, dtype=bool)]
#         similarity_score = np.nanmean(off_diag_vals)
#         similarity_scores.append(similarity_score)
        
#         accum_R_mean += np.nan_to_num(R_group_mean, 0.0)
#         accum_count += 1

#     # 5. Save Outputs
#     scores_arr = np.array(similarity_scores)
#     group_ids_arr = np.array(active_groups)
    
#     if accum_count > 0:
#         avg_R_global = accum_R_mean / accum_count
#     else:
#         avg_R_global = np.zeros((C,C))

#     npy_path = outdir / f"{args.subject}_{args.grouping}_similarity_scores.npy"
#     np.save(npy_path, scores_arr)
#     np.save(outdir / f"{args.subject}_{args.grouping}_ids.npy", group_ids_arr)

#     # Plot 1: Distribution
#     plt.figure(figsize=(8, 6))
#     plt.hist(scores_arr, bins=30, color='mediumpurple', edgecolor='black', alpha=0.8)
#     plt.axvline(0, color='black', linestyle='--', alpha=0.5)
#     mean_val = np.nanmean(scores_arr)
#     plt.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f"Mean: {mean_val:.2f}")
#     plt.title(f"Distribution of Condition Similarity\n({args.grouping.capitalize()} Level) - {args.subject}")
#     plt.xlabel(f"Similarity Score")
#     plt.ylabel(f"Number of {args.grouping}s")
#     plt.legend()
#     dist_path = outdir / f"{args.subject}_{args.grouping}_similarity_distribution.png"
#     plt.tight_layout()
#     plt.savefig(dist_path, dpi=300)
#     plt.close()

#     # Plot 2: Heatmap
#     # Using a larger figure size for 11x11 labels
#     plt.figure(figsize=(10, 8))
#     im = plt.imshow(avg_R_global, vmin=-0.5, vmax=0.5, origin="upper", cmap='RdBu_r')
#     plt.colorbar(im, label="Pearson r")
    
#     # Use valid_cond_names for labels to show 11 specific tasks
#     plt.xticks(range(C), valid_cond_names, rotation=45, ha='right', fontsize=9)
#     plt.yticks(range(C), valid_cond_names, fontsize=9)
    
#     plt.title(f"Average Condition Correlation Matrix ({C}x{C})\n(Averaged across all {args.grouping}s)")
#     hm_path = outdir / f"{args.subject}_{args.grouping}_avg_heatmap.png"
#     plt.tight_layout()
#     plt.savefig(hm_path, dpi=300)
#     plt.close()

#     print(f"✅ Scores saved to: {npy_path}")
#     print(f"✅ Distribution plot: {dist_path}")
#     print(f"✅ Global heatmap: {hm_path}")

# if __name__ == "__main__":
#     main()