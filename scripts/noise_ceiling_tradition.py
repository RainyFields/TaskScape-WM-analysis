#!/usr/bin/env python3
"""
Noise ceiling per voxel (default) or per Glasser region (optional).

Input:
  - A pickle produced by your grouping script:
      { condition_key: [beta_vec, beta_vec, ...] }
    where each beta_vec is length = #voxels (MNI) or #surface vertices (Glasser).

Computation:
  - For each condition with >= min repetitions:
      * Randomly split repetitions into two halves
      * Average each half -> two condition-by-(voxels or regions) matrices
      * Pearson correlation per column between the two halves
  - Repeat bootstrap splits and average the column-wise r (Fisher z not needed
    here since we average across bootstraps of the same split-half statistic).
  - If --use_glasser, average voxel/vertex betas within each atlas label *per trial*
    before the split (region-level ceiling).

Outputs:
  - Histogram figure
  - Pickle of the per-(voxel|region) noise ceiling values

Example:
python noise_ceiling_tradition.py \
    --subject sub-01 \
    --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data/delay/trial_level_betas/grouped_betas/task_relevant_only/sub-01_task_condition_betas.pkl \
    --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/traditional_way \
    --n_bootstrap 1  --min_reps 4 --seed 123

Enable Glasser region-level:
  ... --use_glasser \
      --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nibabel as nib


# ---------- helpers ----------

def rowwise_correlation(X, Y):
    """Column-wise Pearson r between matrices X and Y (same shape). Returns (n_cols,) with NaNs for constant cols."""
    assert X.shape == Y.shape, f"Shape mismatch: {X.shape} vs {Y.shape}"
    n = X.shape[1]
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        x = X[:, i]
        y = Y[:, i]
        # guard against constants
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        r, _ = pearsonr(x, y)
        out[i] = r
    return out


def load_glasser_labels(atlas_path: str):
    """Load Glasser .dlabel/.nii labels -> (labels_vector, region_ids_sorted)."""
    lab_img = nib.load(atlas_path)
    lab = np.asarray(lab_img.get_fdata()).squeeze()  # 1D vector for CIFTI label or 3D/nd -> flatten
    lab = lab.reshape(-1)
    # keep positive labels
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0].astype(int)
    return lab, region_ids


def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    """Return a function avg_by_region(trials x features) -> (trials x n_regions)."""
    # Precompute boolean masks (or indices) per region
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]

    def avg_by_region(A):
        # A: (n_trials, n_features)
        out = np.full((A.shape[0], len(region_ids)), np.nan, dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx.size == 0:
                continue
            # nanmean over selected features
            out[:, j] = np.nanmean(A[:, idx], axis=1)
        return out

    return avg_by_region


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True,
                    help="Path to {condition: [beta_vec,...]} produced by grouping script")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument("--min_reps", type=int, default=4,
                    help="Minimum repetitions per condition to include")
    ap.add_argument("--seed", type=int, default=0)

    # Glasser toggle
    ap.add_argument("--use_glasser", action="store_true",
                    help="If set, average trial betas within Glasser regions before computing NC")
    ap.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii",
                    help="Path to Glasser atlas labels (.dlabel.nii or NIfTI) — used only if --use_glasser")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ----- load grouped betas -----
    input_path = Path(args.input_pickle)
    with open(input_path, "rb") as f:
        condition_betas = pickle.load(f)

    # filter conditions with enough reps
    valid_conditions = [k for k, v in condition_betas.items() if len(v) >= args.min_reps]
    if not valid_conditions:
        raise SystemExit(f"No valid conditions with at least {args.min_reps} repetitions found in {input_path}")

    # infer feature length from first condition
    first_vec = np.asarray(condition_betas[valid_conditions[0]][0])
    n_features = first_vec.size

    # ----- optional Glasser region mapping -----
    if args.use_glasser:
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        labels_vec = labels_vec.astype(int)

        if labels_vec.size != n_features:
            raise SystemExit(
                f"Atlas feature count ({labels_vec.size}) != beta vector length ({n_features}). "
                "Make sure your beta vectors and atlas live in the same space/indexing."
            )
        avg_by_region = make_region_averager(labels_vec, region_ids)
        n_units = len(region_ids)
        unit_name = "region"
    else:
        avg_by_region = None
        n_units = n_features
        unit_name = "voxel"

    # ----- bootstrap split-half NC -----
    boot_nc = []

    for b in range(args.n_bootstrap):
        half1_rows = []
        half2_rows = []

        for cond in valid_conditions:
            # stack repetitions for this condition: (n_reps, n_features)
            reps = np.stack([np.asarray(v).reshape(-1) for v in condition_betas[cond]], axis=0)
            if reps.shape[0] < args.min_reps:
                continue

            # (optional) average into regions per trial
            if avg_by_region is not None:
                reps = avg_by_region(reps)  # (n_reps, n_regions)

            # split reps into two halves
            perm = rng.permutation(reps.shape[0])
            mid = reps.shape[0] // 2
            A = reps[perm[:mid]].mean(axis=0)   # (n_units,)
            B = reps[perm[mid:]].mean(axis=0)   # (n_units,)

            # require both halves non-empty
            if A.size == 0 or B.size == 0:
                continue

            half1_rows.append(A)
            half2_rows.append(B)

        if not half1_rows or not half2_rows:
            # nothing usable this bootstrap
            continue

        H1 = np.stack(half1_rows, axis=0)   # (n_conditions_kept, n_units)
        H2 = np.stack(half2_rows, axis=0)   # (n_conditions_kept, n_units)

        nc_b = rowwise_correlation(H1, H2)  # (n_units,)
        boot_nc.append(nc_b)

    if not boot_nc:
        raise SystemExit("No bootstrap iterations produced valid splits — not enough data per condition.")

    noise_ceiling = np.nanmean(np.stack(boot_nc, axis=0), axis=0)  # (n_units,)

    # ----- save & plot -----
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    clipped_nc = np.clip(noise_ceiling, -1, 1)
    plt.figure(figsize=(10, 6))
    plt.hist(clipped_nc[np.isfinite(clipped_nc)], bins=60, edgecolor='black')
    plt.title(f"Noise Ceiling Distribution ({args.subject}, per-{unit_name})")
    plt.xlabel("Noise Ceiling (Pearson r)")
    plt.ylabel(f"Number of {unit_name}s")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = outdir / f"{args.subject}_noise_ceiling_distribution_per_{unit_name}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    nc_save_path = outdir / f"{args.subject}_noise_ceiling_per_{unit_name}.pkl"
    with open(nc_save_path, "wb") as f:
        pickle.dump(noise_ceiling, f)

    print(f"📊 Plot saved: {fig_path}")
    print(f"💾 Noise ceiling saved: {nc_save_path}")
    if args.use_glasser:
        # also save the region id order, so you can map results back later
        meta_path = outdir / f"{args.subject}_noise_ceiling_per_{unit_name}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({"region_ids": region_ids}, f)
        print(f"🧭 Region id order saved: {meta_path}")


if __name__ == "__main__":
    main()
