#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize condition-wise correlations:

  - List all condition pairs whose correlation exceeds a given threshold
    (default: r >= 0.8).
  - Save them to a CSV.
  - Plot the distribution of all off-diagonal correlations as a histogram.

Inputs:
  - *_cond_similarity_mean_matrix_per_<unit>.npy   (R_mean: C x C)
  - *_cond_names_per_<unit>.json                   (["condA", "condB", ...])

Outputs:
  - <subject>_high_corr_pairs_per_<unit>.csv
  - <subject>_corr_distribution_per_<unit>.png

Example:
python summarize_condition_correlations.py \
  --subject sub-06 \
  --matrix_npy /project/def-pbellec/xuan/fmri_dataset_project/results/condition_similarity/sub-06_cond_similarity_mean_matrix_per_network.npy \
  --cond_json  /project/def-pbellec/xuan/fmri_dataset_project/results/condition_similarity/sub-06_cond_names_per_network.json \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/clustered \
  --unit network \
  --threshold 0.8
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def sanitize_corr(R):
    """Clamp to [-1,1], fill bad entries, symmetrize (same as clustering script)."""
    R = np.array(R, dtype=np.float64)
    R = np.clip(R, -1.0, 1.0, out=R)

    # Diagonal: set NaN diag to 1
    diag = np.diag(R).copy()
    diag[~np.isfinite(diag)] = 1.0
    np.fill_diagonal(R, diag)

    # Off-diagonal NaNs -> row-wise means, fallback to 0
    mask = np.isfinite(R)
    if not np.all(mask):
        row_means = np.where(
            np.any(mask, axis=1, keepdims=True),
            np.nanmean(np.where(mask, R, np.nan), axis=1, keepdims=True),
            0.0
        )
        R = np.where(mask, R, row_means)
        R = np.nan_to_num(R, nan=0.0)

    # Symmetrize & clamp
    R = 0.5 * (R + R.T)
    R = np.clip(R, -1.0, 1.0, out=R)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--matrix_npy", required=True, help="Path to *_mean_matrix_*.npy")
    ap.add_argument("--cond_json", required=True, help="Path to *_cond_names_*.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--unit", default="voxel", help="voxel|region|network (for file naming only)")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Correlation threshold for reporting condition pairs (default: 0.8).")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- Load inputs -----
    R = np.load(args.matrix_npy)
    cond_names = json.loads(Path(args.cond_json).read_text())
    if not (R.shape[0] == R.shape[1] == len(cond_names)):
        raise SystemExit("Matrix size and condition list length must match.")

    # Clean up the matrix a bit (same as clustering script)
    R = sanitize_corr(R)
    C = R.shape[0]

    # ----- Collect all off-diagonal correlations -----
    # Use upper triangle (i < j) to avoid duplicates and ignore diagonal
    iu, ju = np.triu_indices(C, k=1)
    all_corrs = R[iu, ju]

    # Basic stats (helpful for diagnostics)
    finite_corrs = all_corrs[np.isfinite(all_corrs)]
    if finite_corrs.size == 0:
        print("No finite off-diagonal correlations found.")
        return

    print(f"Total number of condition pairs (off-diagonal): {finite_corrs.size}")
    print(f"Correlation stats: "
          f"min={finite_corrs.min():.3f}, "
          f"max={finite_corrs.max():.3f}, "
          f"mean={finite_corrs.mean():.3f}, "
          f"median={np.median(finite_corrs):.3f}")

    # ----- Find high-correlation pairs -----
    mask_high = finite_corrs >= args.threshold
    high_pairs_indices = np.where(mask_high)[0]

    print(f"\nNumber of pairs with r >= {args.threshold:.2f}: {high_pairs_indices.size}")

    # Gather (cond1, cond2, r) for each high-corr pair
    high_pairs = []
    for idx in high_pairs_indices:
        i = int(iu[idx])
        j = int(ju[idx])
        r = float(R[i, j])
        high_pairs.append((cond_names[i], cond_names[j], r))

    # Sort by correlation descending
    high_pairs.sort(key=lambda x: x[2], reverse=True)

    # Print a summary to stdout
    print("\nHigh-correlation condition pairs (sorted by r, descending):")
    for name_i, name_j, r in high_pairs:
        print(f"  {name_i:40s}  <->  {name_j:40s}  r = {r:.3f}")

    # ----- Save high-corr pairs to CSV -----
    csv_path = outdir / f"{args.subject}_high_corr_pairs_per_{args.unit}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition_1", "condition_2", "correlation"])
        for name_i, name_j, r in high_pairs:
            w.writerow([name_i, name_j, f"{r:.6f}"])

    print(f"\n✅ Saved high-correlation pairs to: {csv_path}")

    # ----- Plot correlation distribution -----
    hist_png = outdir / f"{args.subject}_corr_distribution_per_{args.unit}.png"

    plt.figure(figsize=(6, 4))
    plt.hist(finite_corrs, bins=50, range=(-1, 1))
    plt.xlabel("Correlation (r)")
    plt.ylabel("Number of condition pairs")
    plt.title(f"{args.subject} condition-wise correlation distribution (per-{args.unit})")
    plt.tight_layout()
    plt.savefig(hist_png, dpi=300)
    plt.close()

    print(f"✅ Saved correlation distribution histogram to: {hist_png}")


if __name__ == "__main__":
    main()
