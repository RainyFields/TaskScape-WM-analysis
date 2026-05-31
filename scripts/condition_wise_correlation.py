#!/usr/bin/env python3
"""
Condition-similarity across units (voxel/region/network) using split-half patterns.

For each bootstrap:
  - For every condition with >= min_reps:
      * Randomly split its repetitions into two halves;
      * Average each half → vectors A_i, B_i in chosen unit space (length = n_units).
  - Build a correlation matrix R (C x C) where R[i,j] = corr(A_i, B_j) across units.
    Diagonal = within-condition; Off-diagonals = across-condition.

Across bootstraps:
  - Average R to a mean matrix for the heatmap.
  - Aggregate within vs across correlation samples to compare distributions.
  - Perform a bootstrap test on mean(within) - mean(across) and report p-value + CI.

Outputs:
  - Heatmap PNG of mean correlation matrix.
  - Box plot PNG (within vs across), with bootstrap p-value annotated.
  - JSON with summary stats and intervals.

Example:
python condition_wise_correlation.py \
  --subject sub-03 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-03_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/condition_similarity \
  --unit network --n_bootstrap 10 --min_reps 5 \
  --zscore_within_vectors 1 --apply_spearman_brown 0 --seed 0


"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from utils import load_valid_vertex_mask, apply_mask_to_betas_dict


# ----------------- helpers (mostly mirrored from your NC script) -----------------

def load_glasser_labels(atlas_path: str):
    img = nib.load(atlas_path)
    lab = np.asarray(img.get_fdata()).squeeze().reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids

def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]
    def avg_by_region(A):  # A: (n_trials, n_features)
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
        nid = region_to_net[rid]  # 1..12
        if 1 <= nid <= n_networks:
            nets[nid - 1].append(j)
    def avg_by_network(R):  # R: (n_trials, n_regions)
        out = np.full((R.shape[0], n_networks), np.nan, dtype=np.float32)
        for k, cols in enumerate(nets):
            if cols:
                out[:, k] = np.nanmean(R[:, cols], axis=1)
        return out
    return avg_by_network

def zscore_rows(X, eps=1e-8):
    """Z-score each row (vector) independently across units (columns)."""
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

def corr_matrix_rowwise(A, B, eps=1e-8):
    """
    Pearson correlation across units between all row pairs:
      R[i,j] = corr(A[i,:], B[j,:])
    Returns (A_rows x B_rows).
    NaN rows produce NaNs.
    """
    A0 = A - np.nanmean(A, axis=1, keepdims=True)
    B0 = B - np.nanmean(B, axis=1, keepdims=True)
    Astd = np.nanstd(A0, axis=1, keepdims=True)
    Bstd = np.nanstd(B0, axis=1, keepdims=True)
    A0 = A0 / (Astd + eps)
    B0 = B0 / (Bstd + eps)
    # Treat NaNs as 0 after standardization for dot-product; mask denominator by valid counts
    A0 = np.nan_to_num(A0, nan=0.0)
    B0 = np.nan_to_num(B0, nan=0.0)
    # Normalize by number of columns (units). After z-scoring per row, the Pearson corr ~ cosine sim.
    # Using simple dot/num_units is fine here.
    n_units = A.shape[1]
    R = (A0 @ B0.T) / n_units
    return R

def spearman_brown(r, k=2.0):
    return (k * r) / (1.0 + (k - 1.0) * r)


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True,
                    help="Path to {condition: [beta_vec,...]}")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=200)
    ap.add_argument("--min_reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # Unit selection (voxel/region/network)
    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="voxel")
    ap.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--network_assignments_path")

    # Analysis options
    ap.add_argument("--zscore_within_vectors", type=int, default=1,
                    help="Z-score each split-half vector across units before corr (recommended)")
    ap.add_argument("--apply_spearman_brown", type=int, default=0,
                    help="Apply SB (k=2) to diagonal entries only (split-half within-condition corr)")
    
    # Valid vertex mask (excludes medial wall / zero-variance vertices)
    ap.add_argument("--apply_vertex_mask", type=int, default=1,
                    help="Apply valid vertex mask to exclude zero-variance vertices (default: 1)")
    ap.add_argument("--vertex_mask_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/valid_vertex_mask.npy",
                    help="Path to valid vertex mask")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load valid vertex mask ----
    valid_mask = None
    if args.apply_vertex_mask:
        print(f"Loading valid vertex mask from {args.vertex_mask_path}...")
        valid_mask = load_valid_vertex_mask(args.vertex_mask_path)
        n_valid = np.sum(valid_mask)
        n_total = len(valid_mask)
        print(f"  Mask: {n_valid:,}/{n_total:,} vertices ({n_valid/n_total*100:.1f}% valid)")

    # ---- load betas ----
    with open(args.input_pickle, "rb") as f:
        condition_betas = pickle.load(f)
    
    # Apply valid vertex mask to betas
    if valid_mask is not None:
        condition_betas = apply_mask_to_betas_dict(condition_betas, valid_mask)

    valid_conditions = [k for k, v in condition_betas.items() if len(v) >= args.min_reps]
    if not valid_conditions:
        raise SystemExit(f"No conditions with >= {args.min_reps} reps.")

    # infer feature length
    first_vec = np.asarray(condition_betas[valid_conditions[0]][0]).reshape(-1)
    n_features = first_vec.size

    # ---- prep unit transforms ----
    avg_by_region = None
    avg_by_network = None
    region_ids = None

    if args.unit in ("region", "network"):
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        # Apply vertex mask to atlas labels
        if valid_mask is not None:
            labels_vec = labels_vec[valid_mask]
        if labels_vec.size != n_features:
            raise SystemExit(
                f"Atlas feature count ({labels_vec.size}) != beta length ({n_features})."
            )
        avg_by_region = make_region_averager(labels_vec, region_ids)

    if args.unit == "network":
        if not args.network_assignments_path:
            raise SystemExit("--network_assignments_path is required for --unit=network")
        max_region_id = int(region_ids.max())
        region_to_net = load_region_to_network_map(args.network_assignments_path, max_region_id)
        avg_by_network = make_network_averager(region_ids, region_to_net, n_networks=12)

    # Unit meta
    if args.unit == "voxel":
        n_units = n_features
        unit_name = "voxel"
    elif args.unit == "region":
        n_units = len(region_ids)
        unit_name = "region"
    else:
        n_units = 12
        unit_name = "network"

    # ---- bootstrap ----
    C = len(valid_conditions)
    R_accum = np.zeros((C, C), dtype=np.float64)
    within_all = []   # collect diagonal values across bootstraps
    across_all = []   # collect all off-diagonal values across bootstraps
    diffs = []        # per-bootstrap mean(within) - mean(across)

    for b in range(args.n_bootstrap):
        A_rows = []
        B_rows = []
        kept = []

        for idx, cond in enumerate(valid_conditions):
            reps = np.stack([np.asarray(v).reshape(-1) for v in condition_betas[cond]], axis=0)

            # average into units
            if avg_by_region is not None:
                reps = avg_by_region(reps)  # (n_reps, n_regions)
            if avg_by_network is not None:
                reps = avg_by_network(reps)  # (n_reps, 12)

            # split halves
            perm = rng.permutation(reps.shape[0])
            mid = reps.shape[0] // 2
            if mid == 0 or (reps.shape[0] - mid) == 0:
                continue
            A = reps[perm[:mid]].mean(axis=0)
            B = reps[perm[mid:]].mean(axis=0)

            A_rows.append(A)
            B_rows.append(B)
            kept.append(idx)

        if not A_rows:
            continue

        A = np.vstack(A_rows)  # (C_kept, n_units)
        B = np.vstack(B_rows)  # (C_kept, n_units)

        if args.zscore_within_vectors:
            A = zscore_rows(A)
            B = zscore_rows(B)

        # correlation matrix across units
        R = corr_matrix_rowwise(A, B)  # (C_kept x C_kept)

        # optional SB on diagonal (split-half of same condition)
        if args.apply_spearman_brown:
            d = np.clip(np.diag(R), -0.9999, 0.9999)
            d = spearman_brown(d, k=2.0)
            np.fill_diagonal(R, d)

        # accumulate into full CxC (some conditions may be dropped in this bootstrap)
        kept = np.array(kept, dtype=int)
        R_full = np.full((C, C), np.nan, dtype=np.float64)
        R_full[np.ix_(kept, kept)] = R

        # collect samples
        diag_vals = np.diag(R_full)
        off_vals = R_full[~np.eye(C, dtype=bool)]

        within = diag_vals[np.isfinite(diag_vals)]
        across = off_vals[np.isfinite(off_vals)]

        if within.size and across.size:
            within_all.append(within)
            across_all.append(across)
            diffs.append(np.nanmean(within) - np.nanmean(across))

        # add to accumulator as 0 where NaN? We’ll average with a mask
        R_accum += np.nan_to_num(R_full, nan=0.0)
        if b == 0:
            mask = np.isfinite(R_full).astype(np.float64)
            mask_accum = mask
        else:
            mask_accum += np.isfinite(R_full).astype(np.float64)

    # finalize mean matrix
    with np.errstate(invalid="ignore"):
        R_mean = np.divide(R_accum, mask_accum, out=np.full_like(R_accum, np.nan), where=mask_accum > 0)

    # flatten collected samples
    within_cat = np.concatenate(within_all) if within_all else np.array([])
    across_cat = np.concatenate(across_all) if across_all else np.array([])
    diffs = np.array(diffs) if diffs else np.array([])

    if within_cat.size == 0 or across_cat.size == 0:
        raise SystemExit("Not enough valid split-halves to compute statistics.")

    # bootstrap test: two-sided p for mean(within) > mean(across)
    mean_diff = float(np.nanmean(diffs))
    p_right = float(np.mean(diffs <= 0.0))  # fraction non-positive (null: no advantage)
    p_two_sided = float(2 * min(p_right, 1 - p_right))
    ci_lo, ci_hi = (np.nanpercentile(diffs, [2.5, 97.5]) if diffs.size else (np.nan, np.nan))

    # ---- plots ----
    # 1) Heatmap of mean correlation matrix
    plt.figure(figsize=(8, 7))
    im = plt.imshow(R_mean, vmin=-1, vmax=1, origin="upper", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="corr (across units)")
    plt.title(f"Mean split-half correlation across units\n{args.subject} (per-{unit_name}, {C} conditions)")
    plt.xlabel("B half (condition index)")
    plt.ylabel("A half (condition index)")
    heatmap_path = outdir / f"{args.subject}_cond_similarity_across_units_heatmap_per_{unit_name}.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Box plot of within vs across
    plt.figure(figsize=(6, 6))
    plt.boxplot([within_cat, across_cat], labels=["within (diag)", "across (off)"], showmeans=True)
    plt.ylabel("Correlation (across units)")
    plt.title(f"Within vs Across (bootstrapped)\nΔ={mean_diff:.3f}, p≈{p_two_sided:.4g}, 95% CI [{ci_lo:.3f},{ci_hi:.3f}]")
    box_path = outdir / f"{args.subject}_within_vs_across_box_per_{unit_name}.png"
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ---- save numbers ----
    # mean matrix and stats

    cond_names_json = outdir / f"{args.subject}_cond_names_per_{unit_name}.json"
    with open(cond_names_json, "w") as f:
        json.dump(valid_conditions, f, indent=2)

    npy_path = outdir / f"{args.subject}_cond_similarity_mean_matrix_per_{unit_name}.npy"
    np.save(npy_path, R_mean)

    stats = {
        "subject": args.subject,
        "unit": unit_name,
        "n_conditions": C,
        "n_bootstrap": args.n_bootstrap,
        "zscore_within_vectors": bool(args.zscore_within_vectors),
        "spearman_brown_on_diag": bool(args.apply_spearman_brown),
        "within_mean": float(np.nanmean(within_cat)),
        "across_mean": float(np.nanmean(across_cat)),
        "mean_diff_within_minus_across": mean_diff,
        "bootstrap_p_two_sided": p_two_sided,
        "bootstrap_diff_CI95": [float(ci_lo), float(ci_hi)],
        "within_n": int(within_cat.size),
        "across_n": int(across_cat.size),
        "notes": "Correlation across units; diagonal entries compare the two split halves of the same condition."
    }
    json_path = outdir / f"{args.subject}_cond_similarity_stats_per_{unit_name}.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Heatmap saved: {heatmap_path}")
    print(f"✅ Box plot saved: {box_path}")
    print(f"✅ Mean matrix saved: {npy_path}")
    print(f"✅ Stats JSON saved: {json_path}")


if __name__ == "__main__":
    main()
