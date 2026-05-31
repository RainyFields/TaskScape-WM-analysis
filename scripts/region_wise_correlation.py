#!/usr/bin/env python3
"""
Unit-wise correlation across all task conditions.

For each voxel/region/network:
  - Build a response vector over all task conditions (averaged across repetitions).
  - Correlate these vectors across units.

Result:
  - A num_units x num_units correlation matrix telling you how similar
    units are in their task-condition tuning profile.

Inputs
------
- input_pickle: {condition_key: [beta_vec, ...]}
    Produced by task_condition_identification.py
    beta_vec length = n_vertices (Glasser surface).

- atlas_path: Glasser_LR_Dense64k.dlabel.nii
- optional network_assignments_path: region -> network mapping (1..N)

Modes
-----
--unit voxel   : use raw vertices (n_units = n_vertices)
--unit region  : average Glasser vertices into parcels (n_units = n_regions)
--unit network : average regions into networks (n_units = n_networks, default 12)

Outputs
-------
- <subject>_unit_by_unit_correlation_across_conditions_per_<unit>.npy
    shape: (n_units, n_units)

- <subject>_condition_names_per_<unit>.json
    list of condition keys (order matches first axis of cond_unit_responses)

- <subject>_unit_ids_per_<unit>.json
    voxel indices / region_ids / network_indices

- <subject>_cond_by_unit_responses_per_<unit>.npy
    shape: (n_conditions, n_units)

- <subject>_unit_by_unit_correlation_heatmap_per_<unit>.png

Example
-------
python region_wise_correlation.py \
  --subject sub-03 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-03_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --unit voxel \
  --min_reps 5 \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/task_rsa
"""

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pickle


# -----------------------
# Atlas helpers
# -----------------------

def load_glasser_labels(atlas_path: str):
    img = nib.load(atlas_path)
    lab = np.asarray(img.get_fdata()).squeeze().reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids


def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]

    def avg_by_region(A):  # A: (n_samples, n_vertices)
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


def make_network_averager(region_ids: np.ndarray,
                          region_to_net: np.ndarray,
                          n_networks: int = 12):
    nets = [[] for _ in range(n_networks)]
    for j, rid in enumerate(region_ids):
        nid = region_to_net[rid]  # 1..n_networks
        if 1 <= nid <= n_networks:
            nets[nid - 1].append(j)

    def avg_by_network(R):  # R: (n_samples, n_regions)
        out = np.full((R.shape[0], n_networks), np.nan, dtype=np.float32)
        for k, cols in enumerate(nets):
            if cols:
                out[:, k] = np.nanmean(R[:, cols], axis=1)
        return out

    return avg_by_network


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True,
                    help="Path to <subj>_task_condition_betas.pkl")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="region")

    ap.add_argument("--atlas_path",
                    help="Glasser dlabel (required for region/network units)")
    ap.add_argument("--network_assignments_path",
                    help="Text file: one network ID per region (1..N), needed for unit=network")

    ap.add_argument("--min_reps", type=int, default=3,
                    help="Minimum # repetitions per condition to include")
    ap.add_argument("--zscore_within_units", type=int, default=1,
                    help="Z-score each unit's response vector across conditions before corr")

    args = ap.parse_args()
    subj = args.subject

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load grouped condition betas ----
    with open(args.input_pickle, "rb") as f:
        condition_betas = pickle.load(f)

    print(f"Loaded {len(condition_betas)} conditions from {args.input_pickle}")

    # Filter by min_reps
    cond_names = [k for k, v in condition_betas.items() if len(v) >= args.min_reps]
    if not cond_names:
        raise SystemExit(f"No conditions with >= {args.min_reps} reps.")

    cond_names = sorted(cond_names)
    print(f"{len(cond_names)} conditions remaining after min_reps filter.")

    # Infer feature length (n_vertices)
    first_vec = np.asarray(condition_betas[cond_names[0]][0]).reshape(-1)
    n_vertices = first_vec.size
    print(f"Feature length (n_vertices): {n_vertices}")

    # ---- Determine units & averaging ----
    if args.unit == "voxel":
        n_units = n_vertices
        unit_name = "voxel"
        labels_vec = None
        region_ids = None
        avg_by_region = None
        avg_by_network = None
    else:
        if not args.atlas_path:
            raise SystemExit("--atlas_path is required for unit=region/network")
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        if labels_vec.size != n_vertices:
            raise SystemExit(
                f"Atlas vertex count ({labels_vec.size}) != beta length ({n_vertices})."
            )
        avg_by_region = make_region_averager(labels_vec, region_ids)

        if args.unit == "region":
            n_units = len(region_ids)
            unit_name = "region"
            avg_by_network = None
        else:
            # network
            if not args.network_assignments_path:
                raise SystemExit("--network_assignments_path is required for unit=network")
            max_region_id = int(region_ids.max())
            region_to_net = load_region_to_network_map(
                args.network_assignments_path, max_region_id
            )
            n_networks = 12
            avg_by_network = make_network_averager(region_ids, region_to_net,
                                                   n_networks=n_networks)
            n_units = n_networks
            unit_name = "network"

    print(f"Unit type: {unit_name}, n_units = {n_units}")

    # ---- Build condition-by-unit response matrix ----
    n_conditions = len(cond_names)
    cond_unit_responses = np.full((n_conditions, n_units),
                                  np.nan, dtype=np.float32)

    for i, cond in enumerate(cond_names):
        reps = np.stack(
            [np.asarray(v).reshape(-1) for v in condition_betas[cond]],
            axis=0
        )  # (n_reps, n_vertices)
        mean_vert = np.nanmean(reps, axis=0, keepdims=True)  # (1, n_vertices)

        if args.unit == "voxel":
            unit_vec = mean_vert.reshape(-1)  # (n_vertices,)
        else:
            # region averaged: (1, n_regions)
            mean_reg = avg_by_region(mean_vert)
            if args.unit == "region":
                unit_vec = mean_reg.reshape(-1)  # (n_regions,)
            else:
                # network averaged: (1, n_networks)
                mean_net = avg_by_network(mean_reg)
                unit_vec = mean_net.reshape(-1)  # (n_networks,)

        cond_unit_responses[i, :] = unit_vec

    # ---- Optionally z-score within each unit (column) across conditions ----
    X = cond_unit_responses.copy()  # shape: (n_conditions, n_units)

    if args.zscore_within_units:
        col_means = np.nanmean(X, axis=0, keepdims=True)
        col_stds = np.nanstd(X, axis=0, keepdims=True)
        X = (X - col_means) / (col_stds + 1e-8)

    # Handle any all-NaN columns by leaving them as NaN correlations later
    # ---- Correlation across conditions between units ----
    # We want corr between columns (units), so rowvar=False
    R_units = np.corrcoef(X, rowvar=False)  # (n_units, n_units)

    # ---- Save outputs ----
    corr_path = outdir / f"{subj}_unit_by_unit_correlation_across_conditions_per_{unit_name}.npy"
    np.save(corr_path, R_units)

    cond_names_path = outdir / f"{subj}_condition_names_per_{unit_name}.json"
    with open(cond_names_path, "w") as f:
        json.dump(cond_names, f, indent=2)

    if unit_name == "voxel":
        unit_ids = list(range(n_units))
    elif unit_name == "region":
        unit_ids = [int(r) for r in region_ids]
    else:
        unit_ids = list(range(1, n_units + 1))  # network indices 1..n_networks

    unit_ids_path = outdir / f"{subj}_unit_ids_per_{unit_name}.json"
    with open(unit_ids_path, "w") as f:
        json.dump(unit_ids, f, indent=2)

    resp_path = outdir / f"{subj}_cond_by_unit_responses_per_{unit_name}.npy"
    np.save(resp_path, cond_unit_responses)

    print(f"✅ Unit-by-unit correlation saved to: {corr_path}")
    print(f"✅ Condition names saved to: {cond_names_path}")
    print(f"✅ Unit IDs saved to: {unit_ids_path}")
    print(f"✅ Condition-by-unit responses saved to: {resp_path}")

    # ---- Heatmap ----
    plt.figure(figsize=(7, 6))
    im = plt.imshow(R_units, vmin=-1, vmax=1,
                    origin="upper", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="corr across conditions")
    plt.title(f"{subj} unit-by-unit correlation\n(per-{unit_name}, {n_conditions} conditions)")
    plt.xlabel(f"{unit_name} index")
    plt.ylabel(f"{unit_name} index")
    plt.tight_layout()

    heatmap_path = outdir / f"{subj}_unit_by_unit_correlation_heatmap_per_{unit_name}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Heatmap saved to: {heatmap_path}")


if __name__ == "__main__":
    main()
