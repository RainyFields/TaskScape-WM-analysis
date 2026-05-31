#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Task-Level Activation Aggregation (Region & Network) — supports coarse/fine tasks
Author: Xiaoxuan Lei (McGill University / Mila)
Updated: 2025-01-10

Added feature:
--------------
Task grouping now supports THREE modes:

  --task_level coarse    → nback, interdms, ctxdm
  --task_level fine      → nback_loc, nback_ctg, interdms_loc_ABAB, etc.
  --task_level both      → saves BOTH coarse and fine outputs

===============================================================================

python condition_level_visualization.py --subj sub-05 --task_level fine
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib


# -----------------------------------------------------------------------------
# Helpers (same as before)
# -----------------------------------------------------------------------------

def load_glasser_labels(atlas_path: str):
    lab_img = nib.load(atlas_path)
    lab = np.asarray(lab_img.get_fdata()).squeeze()
    lab = lab.reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids


def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]

    def avg_by_region(A):
        out = np.full((A.shape[0], len(region_ids)), np.nan, dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx.size > 0:
                out[:, j] = np.nanmean(A[:, idx], axis=1)
        return out

    return avg_by_region


def load_region_to_network_map(assign_path: str, max_region_id: int, n_networks=12):
    txt = Path(assign_path).read_text().strip().splitlines()
    arr = np.zeros(max_region_id + 1, dtype=int)
    for i, line in enumerate(txt, start=1):
        if i <= max_region_id:
            arr[i] = int(line.strip())
    return arr


def make_network_averager(region_ids, region_to_net, n_networks=12):
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


# -----------------------------------------------------------------------------
# NEW: task grouping strategies
# -----------------------------------------------------------------------------

def coarse_task_name(cond_key: str) -> str:
    """
    Example:
    'interdms_loc_ABAB_loc1_obj3*loc2_obj4' → 'interdms'
    """
    return cond_key.split("_", 1)[0]


def fine_task_name(cond_key: str) -> str:
    """
    Example:
    'interdms_loc_ABAB_loc1_obj3*loc2_obj4' → 'interdms_loc_ABAB'
    """
    parts = cond_key.split("_")
    if len(parts) < 2:
        return parts[0]  # fallback
    return "_".join(parts[:2])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", required=True)

    parser.add_argument("--task_level",
                        choices=["coarse", "fine", "both"],
                        default="coarse",
                        help="Define task granularity: coarse (default), fine, or both.")

    parser.add_argument("--input_pkl_root",
                        default="/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only")

    parser.add_argument("--out_root",
                        default="/project/def-pbellec/xuan/fmri_dataset_project/results/task_level_avg_activity")

    parser.add_argument("--atlas_path",
                        default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")

    parser.add_argument("--network_assignments_path",
                        default="/project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt")

    args = parser.parse_args()
    subj = args.subj

    # ------------------------------
    # Load grouped betas
    # ------------------------------
    input_pkl = Path(args.input_pkl_root) / f"{subj}_task_condition_betas.pkl"
    with open(input_pkl, "rb") as f:
        grouped = pickle.load(f)

    first_key = next(iter(grouped))
    first_vec = np.asarray(grouped[first_key][0]).reshape(-1)
    n_features = first_vec.size

    labels_vec, region_ids = load_glasser_labels(args.atlas_path)
    if labels_vec.size != n_features:
        raise ValueError("Atlas and beta vector size mismatch.")

    avg_by_region = make_region_averager(labels_vec, region_ids)
    region_to_net = load_region_to_network_map(args.network_assignments_path,
                                               max_region_id=int(region_ids.max()))
    avg_by_network = make_network_averager(region_ids, region_to_net)

    save_dir = Path(args.out_root) / subj
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Build task→conditions mapping for both levels
    # ------------------------------
    coarse_map = defaultdict(list)
    fine_map   = defaultdict(list)

    for cond_key, betas in grouped.items():
        coarse_map[coarse_task_name(cond_key)].append((cond_key, betas))
        fine_map[fine_task_name(cond_key)].append((cond_key, betas))

    # ------------------------------
    # Helper to compute task-level means
    # ------------------------------
    def compute_and_save(task_name, cond_list):
        cond_region_means = []

        for _, rep_list in cond_list:
            if not rep_list:
                continue
            reps = np.stack([np.asarray(b).reshape(-1) for b in rep_list], axis=0)
            reps_region = avg_by_region(reps)
            cond_region_means.append(np.nanmean(reps_region, axis=0))

        if not cond_region_means:
            return

        task_region_mean = np.stack(cond_region_means, axis=0).mean(axis=0)

        np.save(save_dir / f"{task_name}_avg_activation.npy",
                task_region_mean.astype(np.float32))

        task_network_mean = avg_by_network(task_region_mean[np.newaxis, :])[0]
        np.save(save_dir / f"{task_name}_network_avg_activation.npy",
                task_network_mean.astype(np.float32))

    # ------------------------------
    # Output coarse level
    # ------------------------------
    if args.task_level in ("coarse", "both"):
        print("\n=== Saving COARSE task-level results ===")
        for task_name, cond_list in sorted(coarse_map.items()):
            compute_and_save(task_name, cond_list)

    # ------------------------------
    # Output fine level
    # ------------------------------
    if args.task_level in ("fine", "both"):
        print("\n=== Saving FINE task-level results ===")
        for task_name, cond_list in sorted(fine_map.items()):
            compute_and_save(task_name, cond_list)

    print("\n🎉 Done generating task-level activation features.")


if __name__ == "__main__":
    main()
