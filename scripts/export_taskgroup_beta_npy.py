#!/usr/bin/env python3
"""
Export task-group mean beta .npy files for brain visualization.

For each subject and each task group (dms, 1back, interdms, ctxdm), this script
creates region-level and network-level mean beta vectors.
"""

import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np


SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
TASK_GROUPS = ["dms", "1back", "interdms", "ctxdm"]

INPUT_DIR = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/"
    "trial_level_betas/grouped_betas/task_relevant_only"
)
ATLAS_PATH = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
NETWORK_ASSIGNMENTS_PATH = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt"
)
OUTPUT_DIR = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/results/beta_visualization/task_group_level_npy"
)


def get_group_from_key(key: str):
    lk = key.lower()
    if lk.startswith("nback"):
        return "1back"
    if lk.startswith("ctxdm"):
        return "ctxdm"
    if lk.startswith("interdms"):
        return "interdms"
    if lk.startswith("dms"):
        return "dms"
    return None


def load_glasser():
    lab = np.asarray(nib.load(str(ATLAS_PATH)).get_fdata()).squeeze().astype(int).reshape(-1)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids


def load_region_to_network(max_region_id: int):
    lines = NETWORK_ASSIGNMENTS_PATH.read_text().strip().splitlines()
    arr = np.zeros(max_region_id + 1, dtype=int)
    for i, line in enumerate(lines, start=1):
        if i <= max_region_id:
            arr[i] = int(line.strip())
    return arr


def mean_region_from_vertex(beta_vertex: np.ndarray, labels: np.ndarray, region_ids: np.ndarray):
    out = np.full(len(region_ids), np.nan, dtype=np.float32)
    for i, rid in enumerate(region_ids):
        idx = labels == rid
        if np.any(idx):
            out[i] = np.nanmean(beta_vertex[idx])
    return out


def mean_network_from_region(beta_region: np.ndarray, region_ids: np.ndarray, region_to_net: np.ndarray):
    out = np.full(12, np.nan, dtype=np.float32)
    for net_id in range(1, 13):
        rid_sel = region_ids[region_to_net[region_ids] == net_id]
        if rid_sel.size == 0:
            continue
        cols = [np.where(region_ids == rid)[0][0] for rid in rid_sel]
        out[net_id - 1] = np.nanmean(beta_region[cols])
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels, region_ids = load_glasser()
    region_to_net = load_region_to_network(int(region_ids.max()))

    np.save(OUTPUT_DIR / "region_ids.npy", region_ids.astype(np.int32))
    np.save(OUTPUT_DIR / "network_ids.npy", np.arange(1, 13, dtype=np.int32))

    manifest = {}

    for subject in SUBJECTS:
        pkl_path = INPUT_DIR / f"{subject}_task_condition_betas.pkl"
        if not pkl_path.exists():
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        grouped = {k: [] for k in TASK_GROUPS}
        for key, reps in data.items():
            group = get_group_from_key(str(key))
            if group is None:
                continue
            grouped[group].extend(reps)

        sub_dir = OUTPUT_DIR / subject
        sub_dir.mkdir(parents=True, exist_ok=True)

        manifest[subject] = {}
        region_stack = []
        network_stack = []
        availability = {}
        n_reps = {}

        for group in TASK_GROUPS:
            reps = grouped[group]
            n_reps[group] = len(reps)
            availability[group] = len(reps) > 0

            if len(reps) == 0:
                beta_region = np.full(len(region_ids), np.nan, dtype=np.float32)
                beta_network = np.full(12, np.nan, dtype=np.float32)
            else:
                reps_mat = np.stack([np.asarray(v).reshape(-1) for v in reps], axis=0)
                beta_vertex = np.nanmean(reps_mat, axis=0).astype(np.float32)
                beta_region = mean_region_from_vertex(beta_vertex, labels, region_ids)
                beta_network = mean_network_from_region(beta_region, region_ids, region_to_net)

            region_stack.append(beta_region)
            network_stack.append(beta_network)

            region_path = sub_dir / f"{subject}_taskgroup-{group}_beta_region.npy"
            network_path = sub_dir / f"{subject}_taskgroup-{group}_beta_network.npy"
            np.save(region_path, beta_region)
            np.save(network_path, beta_network)

            manifest[subject][f"{group}_region"] = str(region_path)
            manifest[subject][f"{group}_network"] = str(network_path)

        region_mat = np.stack(region_stack, axis=0).astype(np.float32)
        network_mat = np.stack(network_stack, axis=0).astype(np.float32)
        np.save(sub_dir / f"{subject}_taskgroup_region_matrix.npy", region_mat)
        np.save(sub_dir / f"{subject}_taskgroup_network_matrix.npy", network_mat)
        np.save(sub_dir / f"{subject}_taskgroup_order.npy", np.array(TASK_GROUPS, dtype=object))

        info = {
            "subject": subject,
            "taskgroup_order": TASK_GROUPS,
            "n_repetitions_per_group": n_reps,
            "availability": availability,
            "region_n": int(len(region_ids)),
            "network_n": 12,
            "note": "Groups with no data are saved as all-NaN arrays.",
        }
        (sub_dir / f"{subject}_taskgroup_export_info.json").write_text(json.dumps(info, indent=2))

    (OUTPUT_DIR / "file_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

