#!/usr/bin/env python3
"""Convert a legacy grouped-beta pickle (task_condition_identification.py output, contaminated) into the
same bundle format used by noise_ceiling_clean.py, so the corrected estimator can be run on the OLD
mis-indexed / duplicated inputs (wrong-column diagnostic). Also counts byte-identical duplicate vectors.

Task grouping reproduces noise_ceiling_final_version.group_betas(aggregation='task') (substring match of
the 12 task names over the condition keys, longest first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from utils import load_valid_vertex_mask  # noqa: E402
from noise_ceiling_clean import TASK_VARIANTS  # noqa: E402
from build_clean_delay_betas_nc import load_atlas, N_VERTICES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", required=True)
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--atlas_path", default=str(PROJECT / "data/Glasser_LR_Dense64k.dlabel.nii"))
    ap.add_argument("--network_assignments_path", default=str(PROJECT / "data/cortex_parcel_network_assignments.txt"))
    ap.add_argument("--vertex_mask_path", default=str(PROJECT / "data/valid_vertex_mask.npy"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    valid_mask = load_valid_vertex_mask(args.vertex_mask_path).astype(bool)
    _, W, Wn, _, _ = load_atlas(Path(args.atlas_path), Path(args.network_assignments_path), valid_mask)

    with open(args.pickle, "rb") as f:
        raw = pickle.load(f)
    targets = sorted(TASK_VARIANTS, key=len, reverse=True)
    recs, region, network = [], [], []
    hashes = {}
    n_dup = 0
    for key, vecs in raw.items():
        tv = next((t for t in targets if t in key), None)
        if tv is None:
            continue
        for i, v in enumerate(vecs):
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size != N_VERTICES:
                raise SystemExit(f"vector size {v.size} for key {key}")
            h = hashlib.md5(v.tobytes()).hexdigest()
            first = hashes.setdefault(h, f"{key}[{i}]")
            is_dup = first != f"{key}[{i}]"
            n_dup += int(is_dup)
            vv = v.astype(np.float64)
            vv[~valid_mask] = 0.0
            R = vv @ W
            region.append(R.astype(np.float32))
            network.append((R @ Wn).astype(np.float32))
            recs.append({"record_id": f"legacy_{len(recs):06d}", "subject": args.subj, "condition_key": key, "task_variant_lookup": tv,
                         "task_variant_verified": tv, "label_resolved": True, "retain_trial_level": True, "retain_item_level": True,
                         "phase": "delay", "stim_order": np.nan, "vec_md5": h, "is_byte_duplicate_of_earlier": is_dup,
                         "first_occurrence": first, "finite": bool(np.all(np.isfinite(v[valid_mask])))})
    prov = pd.DataFrame(recs)
    prov.to_csv(out / f"{args.subj}_delay_provenance.csv", index=False)
    np.save(out / f"{args.subj}_delay_region.npy", np.stack(region))
    np.save(out / f"{args.subj}_delay_network.npy", np.stack(network))
    per_tv = prov.groupby("task_variant_lookup").agg(n_records=("record_id", "size"), n_unique_vectors=("vec_md5", "nunique")).reset_index()
    per_tv["n_duplicates"] = per_tv.n_records - per_tv.n_unique_vectors
    per_tv["frac_duplicates"] = per_tv.n_duplicates / per_tv.n_records
    per_tv.to_csv(out / f"{args.subj}_legacy_duplicates_per_variant.csv", index=False)
    man = {"subject": args.subj, "pickle": args.pickle, "n_records": int(len(prov)), "n_unique_vectors": int(len(hashes)),
           "n_byte_duplicate_records": int(n_dup), "frac_duplicate_records": float(n_dup / len(prov)),
           "n_nonfinite_records": int((~prov.finite).sum()), "task_variants_present": sorted(prov.task_variant_lookup.unique())}
    (out / f"{args.subj}_manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    print(per_tv.to_string(index=False))


if __name__ == "__main__":
    main()
