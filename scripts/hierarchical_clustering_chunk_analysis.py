#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze on-block vs off-block values from a reordered condition-by-condition matrix.

Inputs:
  - Reordered matrix:  <subject>_reordered_matrix_per_<unit>.npy
  - Cluster CSV:       <subject>_cluster_assignments_per_<unit>.csv
                        columns: condition, cluster_id, order_index

Outputs (prefix = --out_prefix, default based on subject/unit):
  - {prefix}_boxplot.png
  - {prefix}_values.csv         (columns: value, kind [on|off])
  - {prefix}_stats.json         (t-test & U-test summary)

Usage example:
python hierarchical_clustering_chunk_analysis.py \
  --matrix_npy /project/def-pbellec/xuan/fmri_dataset_project/results/clustered/sub-03_reordered_matrix_per_network.npy \
  --cluster_csv /project/def-pbellec/xuan/fmri_dataset_project/results/clustered/sub-03_cluster_assignments_per_network.csv \
  --out_prefix /project/def-pbellec/xuan/fmri_dataset_project/results/clustered/sub-03_network_blocks

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu


def fisher_z(r):
    """Fisher z-transform with clipping to keep numerical stability."""
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def collect_block_values(R_ord: np.ndarray, cluster_ids_ord: np.ndarray,
                         use_upper_triangle_only: bool = True,
                         exclude_diag: bool = True):
    """
    Collect within-cluster (on-block) and between-cluster (off-block) values
    from a reordered symmetric matrix R_ord.

    Returns:
        on_vals: np.ndarray, concatenated within-cluster values
        off_vals: np.ndarray, concatenated between-cluster values
    """
    n = R_ord.shape[0]
    assert R_ord.shape[1] == n, "Matrix must be square"
    assert len(cluster_ids_ord) == n, "Cluster list length must match matrix size"

    # Build masks
    row_ids = np.arange(n)
    I, J = np.meshgrid(row_ids, row_ids, indexing="ij")

    same_cluster = (cluster_ids_ord[I] == cluster_ids_ord[J])
    diff_cluster = ~same_cluster

    mask = np.ones_like(R_ord, dtype=bool)
    if exclude_diag:
        mask &= (I != J)

    if use_upper_triangle_only:
        mask &= (I < J)

    on_mask = mask & same_cluster
    off_mask = mask & diff_cluster

    on_vals = R_ord[on_mask]
    off_vals = R_ord[off_mask]
   
    return on_vals, off_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_npy", required=True,
                    help="Path to <subject>_reordered_matrix_per_<unit>.npy")
    ap.add_argument("--cluster_csv", required=True,
                    help="Path to <subject>_cluster_assignments_per_<unit>.csv")
    ap.add_argument("--out_prefix", default=None,
                    help="Prefix for outputs (png/csv/json). If omitted, derived from inputs.")
    ap.add_argument("--fisher_z", action="store_true",
                    help="Apply Fisher z-transform to values before statistical tests.")
    ap.add_argument("--include_diag", action="store_true",
                    help="Include diagonal entries; by default they are excluded.")
    ap.add_argument("--use_full_matrix", action="store_true",
                    help="Use all symmetric entries (i,j and j,i). Default uses upper triangle only.")
    args = ap.parse_args()

    # Load matrix
    R_ord = np.load(args.matrix_npy)
    if R_ord.ndim != 2 or R_ord.shape[0] != R_ord.shape[1]:
        raise SystemExit("Reordered matrix must be a square 2D array.")

    # Load cluster IDs in the SAME order as the matrix axes.
    # Your CSV has an 'order_index' column matching the reordered order (0..C-1).
    df = pd.read_csv(args.cluster_csv)
    if not {"condition", "cluster_id", "order_index"}.issubset(df.columns):
        raise SystemExit("Cluster CSV must have columns: condition, cluster_id, order_index")

    df_sorted = df.sort_values("order_index").reset_index(drop=True)
    cluster_ids_ord = df_sorted["cluster_id"].to_numpy()

    # Collect values
    on_vals, off_vals = collect_block_values(
        R_ord=R_ord,
        cluster_ids_ord=cluster_ids_ord,
        use_upper_triangle_only=(not args.use_full_matrix),
        exclude_diag=(not args.include_diag),
    )

    if args.fisher_z:
        on_stats_vals = fisher_z(on_vals)
        off_stats_vals = fisher_z(off_vals)
    else:
        on_stats_vals = on_vals
        off_stats_vals = off_vals

    # Basic sanity
    if on_stats_vals.size == 0:
        raise SystemExit("No on-block values found (do you only have one condition or empty clusters?).")
    if off_stats_vals.size == 0:
        print("WARNING: No off-block values found (perhaps only one cluster). Stats will be limited.")

    # Welch's t-test (unequal variances)
    t_stat, p_t = (np.nan, np.nan)
    if off_stats_vals.size > 0:
        t_stat, p_t = ttest_ind(on_stats_vals, off_stats_vals, equal_var=False)

    # Mann–Whitney U (non-parametric)
    u_stat, p_u = (np.nan, np.nan)
    if off_stats_vals.size > 0:
        try:
            u_stat, p_u = mannwhitneyu(on_stats_vals, off_stats_vals, alternative="two-sided")
        except ValueError:
            # Happens if all values identical in one group
            u_stat, p_u = (np.nan, np.nan)

    # Prepare outputs
    if args.out_prefix is None:
        # Derive something readable from file names
        base = Path(args.matrix_npy).stem.replace("_reordered_matrix", "")
        out_prefix = f"{base}_block_analysis"
    else:
        out_prefix = args.out_prefix
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # Save boxplot
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.boxplot([on_stats_vals, off_stats_vals] if off_stats_vals.size > 0 else [on_stats_vals],
               labels=["on-block", "off-block"] if off_stats_vals.size > 0 else ["on-block"])
    ylabel = "value (Fisher z)" if args.fisher_z else "correlation"
    ax.set_ylabel(ylabel)
    ax.set_title("On-block vs Off-block")
    ax.grid(True, axis="y", alpha=0.3)
    png_path = f"{out_prefix}_boxplot.png"
    # ----- Compute difference & annotate -----
    mean_diff = np.mean(on_stats_vals) - np.mean(off_stats_vals)
    p_value = p_t if not np.isnan(p_t) else np.nan

    # Add text annotation on the plot
    ax.text(1.5, max(np.max(on_stats_vals), np.max(off_stats_vals)),
            f"Δmean = {mean_diff:.3f}\np = {p_value:.3g}",
            ha="center", va="bottom", fontsize=10, color="black")

    # Save figure
    png_path = f"{out_prefix}_boxplot.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----- Print summary -----
    print(f"Mean on-block:  {np.mean(on_stats_vals):.4f}")
    print(f"Mean off-block: {np.mean(off_stats_vals):.4f}")
    print(f"Difference (on - off): {mean_diff:.4f}")
    print(f"Welch t-test p-value:  {p_value:.3g}")


    # Save raw sampled values
    values_csv = f"{out_prefix}_values.csv"
    if off_stats_vals.size > 0:
        df_vals = pd.DataFrame({
            "value": np.concatenate([on_stats_vals, off_stats_vals]),
            "kind": (["on"] * len(on_stats_vals)) + (["off"] * len(off_stats_vals)),
        })
    else:
        df_vals = pd.DataFrame({"value": on_stats_vals, "kind": ["on"] * len(on_stats_vals)})
    df_vals.to_csv(values_csv, index=False)

    # Summary stats
    stats = {
        "fisher_z": bool(args.fisher_z),
        "include_diag": bool(args.include_diag),
        "use_full_matrix": bool(args.use_full_matrix),
        "n_on": int(on_stats_vals.size),
        "n_off": int(off_stats_vals.size),
        "on_mean": float(np.mean(on_stats_vals)),
        "on_std": float(np.std(on_stats_vals, ddof=1)) if on_stats_vals.size > 1 else float("nan"),
        "off_mean": float(np.mean(off_stats_vals)) if off_stats_vals.size > 0 else float("nan"),
        "off_std": float(np.std(off_stats_vals, ddof=1)) if off_stats_vals.size > 1 else float("nan"),
        "welch_t_stat": float(t_stat) if np.isfinite(t_stat) else None,
        "welch_t_pvalue": float(p_t) if np.isfinite(p_t) else None,
        "mannwhitney_u_stat": float(u_stat) if np.isfinite(u_stat) else None,
        "mannwhitney_u_pvalue": float(p_u) if np.isfinite(p_u) else None,
    }
    json_path = f"{out_prefix}_stats.json"
    Path(json_path).write_text(json.dumps(stats, indent=2))

    # Console report
    print(f"✅ Saved boxplot: {png_path}")
    print(f"✅ Saved values:  {values_csv}")
    print(f"✅ Saved stats:   {json_path}")
    print(f"Counts → on: {stats['n_on']} | off: {stats['n_off']}")
    print(f"Means  → on: {stats['on_mean']:.4f} | off: {stats['off_mean'] if stats['n_off'] else 'NA'}")
    if stats["welch_t_pvalue"] is not None:
        print(f"Welch t-test: t={stats['welch_t_stat']:.3f}, p={stats['welch_t_pvalue']:.3g}")
    if stats["mannwhitney_u_pvalue"] is not None:
        print(f"Mann–Whitney U: U={stats['mannwhitney_u_stat']:.3g}, p={stats['mannwhitney_u_pvalue']:.3g}")

if __name__ == "__main__":
    main()
