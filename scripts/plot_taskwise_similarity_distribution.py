#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot similarity distributions for each of the 11 task groups
using the condition-by-condition similarity matrix.

Inputs:
  - *_cond_similarity_mean_matrix_per_<unit>.npy
  - *_cond_names_per_<unit>.json

Output (for each task):
  - <subject>_<task>_similarity_distribution_per_<unit>.png
  - <subject>_<task>_similarity_values_per_<unit>.npy

Usage:
python plot_taskwise_similarity_distribution.py \
    --subject sub-03 \
    --matrix_npy /project/def-pbellec/xuan/fmri_dataset_project/results/condition_similarity/sub-03_cond_similarity_mean_matrix_per_network.npy \
    --cond_json  /project/def-pbellec/xuan/fmri_dataset_project/results/condition_similarity/sub-03_cond_names_per_network.json \
    --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/taskwise_similarity/ \
    --unit network


"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TASK_GROUPS = [
    "nback_loc",
    "nback_ctg",
    "nback_obj",
    "ctxdm_col",
    "ctxdm_lco",
    "interdms_loc_ABAB",
    "interdms_loc_ABBA",
    "interdms_obj_ABAB",
    "interdms_obj_ABBA",
    "interdms_ctg_ABAB",
    "interdms_ctg_ABBA",
]

def vectorize_upper_triangle(M):
    """vectorize upper triangle excluding diagonal"""
    idx = np.triu_indices_from(M, k=1)
    return M[idx]

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--matrix_npy", required=True)
    ap.add_argument("--cond_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--unit", default="region")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    R = np.load(args.matrix_npy)
    cond_names = json.loads(Path(args.cond_json).read_text())

    assert R.shape[0] == R.shape[1] == len(cond_names), "Matrix and condition list must match."

    # Map conditions into tasks
    task_to_indices = {t: [] for t in TASK_GROUPS}

    for i, name in enumerate(cond_names):
        for t in TASK_GROUPS:
            if name.startswith(t):
                task_to_indices[t].append(i)
                break

    print(f"Found task membership counts:")
    for t in TASK_GROUPS:
        print(f"  {t}: {len(task_to_indices[t])}")

    # For each task group, extract its similarity values and plot distribution
    for t in TASK_GROUPS:
        idx = task_to_indices[t]
        if len(idx) < 2:
            print(f"[{t}] Skipping: only {len(idx)} conditions.")
            continue

        submat = R[np.ix_(idx, idx)]
        vals = vectorize_upper_triangle(submat)

        # Save raw values
        np.save(outdir / f"{args.subject}_{t}_similarity_values_per_{args.unit}.npy", vals)

        # Plot distribution
        plt.figure(figsize=(7, 5))
        plt.hist(vals, bins=30, alpha=0.75, color="steelblue")
        plt.title(f"{args.subject} | {t} similarity distribution\n(per-{args.unit})")
        plt.xlabel("correlation")
        plt.ylabel("frequency")
        plt.grid(alpha=0.3)

        fig_path = outdir / f"{args.subject}_{t}_similarity_distribution_per_{args.unit}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"Saved: {fig_path}")

    print("Done.")

if __name__ == "__main__":
    main()
