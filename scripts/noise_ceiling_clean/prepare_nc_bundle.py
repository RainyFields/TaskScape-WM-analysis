#!/usr/bin/env python3
"""Turn the validated shared clean delay-beta H5 into region/network matrices for noise_ceiling_clean.py.

Consumes (read-only) data/encoding_delay/clean_delay_betas/{subj}_delay_betas.h5 and the matching
provenance, and refuses to run unless that subject's validation JSON reports zero wrong columns, zero
encoding rows, zero duplicate keys, zero identical vectors, and zero non-finite values.

Region means use the shared valid-vertex mask (same convention as the legacy noise-ceiling script and
as similarity_analysis_clean.Aggregator). Networks are the unweighted mean of the 360 region means.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import paths  # TaskScape-WM inputs only; rejects the legacy tree
N_VERTICES = 64984
N_REGIONS = 360
EXTREME_P95 = 100.0  # same record-level rule as similarity_analysis_clean.EXTREME_PSC


def load_averagers(mask: np.ndarray):
    lab = np.asarray(nib.load(str(paths.glasser_dlabel())).get_fdata()).reshape(-1).astype(int)
    if lab.size != N_VERTICES:
        raise SystemExit(f"atlas size {lab.size}")
    net = np.loadtxt(paths.parcel_networks()).astype(int)
    if net.size != N_REGIONS:
        raise SystemExit("network assignment length != 360")
    valid = np.where(mask)[0]
    lab_v = lab[valid]
    W = np.zeros((valid.size, N_REGIONS), np.float32)
    for j, rid in enumerate(range(1, N_REGIONS + 1)):
        m = lab_v == rid
        if m.sum() == 0:
            raise SystemExit(f"region {rid} has no valid vertices")
        W[m, j] = 1.0 / m.sum()
    Wn = np.zeros((N_REGIONS, 12), np.float32)
    for k in range(1, 13):
        cols = np.where(net == k)[0]
        Wn[cols, k - 1] = 1.0 / cols.size
    return valid, W, Wn, net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subj", required=True)
    ap.add_argument("--clean_dir", default=None, help="default: <dataset>/derivatives/glm-clean-delay/<subj>")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--output_root", default=None)
    args = ap.parse_args()
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard(f"prepare_nc_bundle_{args.subj}")
    paths.require_not_legacy(args.out_dir, "--out_dir")
    clean = paths.require_not_legacy(Path(args.clean_dir) if args.clean_dir else paths.clean_delay_dir() / args.subj, "--clean_dir")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val = json.loads((clean / f"{args.subj}_validation.json").read_text())
    checks = {
        "wrong_column": val["vector_verification"]["wrong_column"],
        "encoding": val["n_encoding_in_delay_dataset"],
        "duplicate_keys": val["n_duplicate_keys"],
        "identical_vectors": val["n_physically_identical_vectors"],
        "nan_inf": val["n_nan_inf"],
    }
    bad = {k: v for k, v in checks.items() if v != 0}
    if bad:
        raise SystemExit(f"refusing unclean input for {args.subj}: {bad}")

    prov_all = pd.read_csv(clean / f"{args.subj}_provenance.csv")
    if not (prov_all.phase == "delay").all():
        raise SystemExit("non-delay rows in provenance")
    ret = prov_all[prov_all.retained].copy()
    ret = ret.sort_values("record_id").reset_index(drop=True)
    if not np.array_equal(ret.record_id.to_numpy(), np.arange(len(ret))):
        raise SystemExit("retained record_id is not 0..n-1 in order")

    # Trial-level correctness: drop every retained delay of a trial that has any incorrect delay row.
    trial_key = ["session", "bold_task", "run", "trial_number"]
    bad_trials = set(map(tuple, prov_all.loc[~prov_all.retained, trial_key].drop_duplicates().to_numpy()))
    ret["retain_trial_level"] = [tuple(r) not in bad_trials for r in ret[trial_key].to_numpy()]

    mask = np.load(paths.valid_vertex_mask()).astype(bool)
    if mask.size != N_VERTICES:
        raise SystemExit("mask size")
    valid, W, Wn, net = load_averagers(mask)

    region = np.empty((len(ret), N_REGIONS), np.float32)
    network = np.empty((len(ret), 12), np.float32)
    p50 = np.empty(len(ret), np.float32)
    p95 = np.empty(len(ret), np.float32)
    mx = np.empty(len(ret), np.float32)
    argmax = np.empty(len(ret), np.int32)

    with h5py.File(clean / f"{args.subj}_delay_betas.h5", "r") as f:
        betas = f["betas"]
        if betas.shape != (len(ret), N_VERTICES):
            raise SystemExit(f"H5 shape {betas.shape} != {(len(ret), N_VERTICES)}")
        if f.attrs.get("phase", "delay") != "delay":
            raise SystemExit("H5 phase is not delay")
        for start in range(0, len(ret), args.chunk):
            stop = min(start + args.chunk, len(ret))
            chunk = betas[start:stop][:, valid].astype(np.float32)
            if not np.all(np.isfinite(chunk)):
                raise SystemExit(f"non-finite values in rows {start}:{stop}")
            region[start:stop] = chunk @ W
            network[start:stop] = region[start:stop] @ Wn
            abs_c = np.abs(chunk)
            p50[start:stop] = np.percentile(abs_c, 50, axis=1)
            p95[start:stop] = np.percentile(abs_c, 95, axis=1)
            mx[start:stop] = abs_c.max(axis=1)
            argmax[start:stop] = valid[abs_c.argmax(axis=1)]
            print(f"[{args.subj}] rows {stop}/{len(ret)}", flush=True)

    if not np.all(np.isfinite(region)) or not np.all(np.isfinite(network)):
        raise SystemExit("non-finite region/network means")

    ret["abs_beta_p50"] = p50
    ret["abs_beta_p95"] = p95
    ret["abs_beta_max"] = mx
    ret["argmax_vertex"] = argmax
    ret["extreme_record"] = p95 > EXTREME_P95
    ret["phase"] = "delay"
    # columns the estimator reads
    keep_cols = [c for c in [
        "record_id", "subject", "session", "run", "bold_task", "task_variant", "task_family",
        "label_lookup", "label_blockfile", "label_action", "label_resolved",
        "trial_number", "stim_order", "phase", "trial_index", "glm_column_full",
        "retained", "retain_trial_level", "legacy_multiplicity",
        "source_beta_file", "source_design_file", "source_events_file",
        "abs_beta_p50", "abs_beta_p95", "abs_beta_max", "argmax_vertex", "extreme_record",
    ] if c in ret.columns or c in ("retain_trial_level", "extreme_record", "abs_beta_p50", "abs_beta_p95", "abs_beta_max", "argmax_vertex")]
    ret[keep_cols].to_csv(out / f"{args.subj}_delay_provenance.csv", index=False)
    np.save(out / f"{args.subj}_delay_region.npy", region)
    np.save(out / f"{args.subj}_delay_network.npy", network)
    extreme = ret.loc[ret.extreme_record, ["record_id", "session", "run", "bold_task", "task_variant",
                                            "trial_number", "stim_order", "trial_index",
                                            "abs_beta_p50", "abs_beta_p95", "abs_beta_max", "argmax_vertex",
                                            "source_beta_file"]]
    extreme.to_csv(out / f"{args.subj}_extreme_records.csv", index=False)

    counts = []
    for tv, g in ret.groupby("task_variant"):
        counts.append({
            "subject": args.subj, "task_variant": tv, "n_runs": g.groupby(["session", "run"]).ngroups,
            "n_unique_delay_betas": len(g),
            "n_trial_level_correct": int(g.retain_trial_level.sum()),
            "n_extreme_p95_gt_100": int(g.extreme_record.sum()),
            "n_legacy_multiplicity_2": int((g.legacy_multiplicity == 2).sum()) if "legacy_multiplicity" in g else None,
            "half_floor": len(g) // 2, "odd": bool(len(g) % 2),
        })
    pd.DataFrame(counts).sort_values("task_variant").to_csv(out / f"{args.subj}_trial_counts.csv", index=False)
    manifest = {
        "subject": args.subj,
        "source_h5": str(clean / f"{args.subj}_delay_betas.h5"),
        "source_validation": checks,
        "cifti_axis_sha1": val.get("cifti_axis_sha1"),
        "n_retained": int(len(ret)),
        "n_trial_level": int(ret.retain_trial_level.sum()),
        "n_extreme_p95_gt_100": int(ret.extreme_record.sum()),
        "n_relabelled_records": int(ret.label_action.astype(str).str.startswith("relabel").sum()) if "label_action" in ret else None,
        "n_unresolved_records": int((~ret.label_resolved.astype(bool)).sum()) if "label_resolved" in ret else None,
        "spatial": {"region_order": "Glasser ids 1..360", "network_order": "Cole-Anticevic 1..12",
                    "network_of_region": net.tolist(), "valid_vertices": int(mask.sum()),
                    "aggregation": "mean of valid vertices in the parcel; network = unweighted mean of region means"},
        "extreme_rule": f"record excluded in the sensitivity variant if p95 |PSC| on valid vertices > {EXTREME_P95}",
        "passed": True,
    }
    (out / f"{args.subj}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "spatial"}, indent=2))


if __name__ == "__main__":
    main()
