#!/usr/bin/env python3
"""Build a clean, fully provenanced delay-beta dataset for the noise-ceiling analysis.

One record per *physical* delay-period GLM beta (subject, session, run, GLM column). The GLM column is
the ``trial_index`` of the UNFILTERED design CSV written by ``glm_analysis.py`` and is asserted against
the regressor names stored in the H5 file (``task_regressor_names``), so a mis-indexed or encoding
column cannot enter the dataset. Correct-trial filtering is recorded as boolean flags (never by
dropping rows), so original trial indices are preserved.

Per subject the script writes

  <out_root>/<subj>/<subj>_delay_provenance.csv   one row per delay beta (all, with retain flags)
  <out_root>/<subj>/<subj>_delay_region.npy       (n_records x 360) region means over valid vertices
  <out_root>/<subj>/<subj>_delay_network.npy      (n_records x 12) unweighted mean of region means
  <out_root>/<subj>/<subj>_delay_vertex.h5        (optional, --save_vertex) n_records x 64984 float32
  <out_root>/<subj>/<subj>_manifest.json          validation report (fail-fast assertions + counts)
  <out_root>/<subj>/<subj>_inputs.csv             every H5/design CSV consumed (size, mtime, md5 of CSV)

Task-variant labels: ``task_variant_lookup`` comes from data/study_designs (legacy lookup, DMS ->
dms_loc); ``task_variant_verified`` / ``label_action`` / ``label_resolved`` come from an optional
block-file verification table (--verified_labels). Nothing is relabelled silently: both columns are kept.

Example (compute node):
  python build_clean_delay_betas_nc.py --subj sub-01 \
      --out_root /project/.../.worktrees/nc_regen/data/encoding_delay/clean_delay_betas_nc \
      --verified_labels results/noise_ceiling/regenerated_clean/labels/run_labels_verified.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # worktree scripts/ (utils, task_condition_identification)
from utils import load_valid_vertex_mask  # noqa: E402
import task_condition_identification as tci  # noqa: E402  (lookup helper only)

TASK_VARIANTS = [
    "dms_loc",
    "nback_loc", "nback_ctg", "nback_obj",
    "ctxdm_col", "ctxdm_lco",
    "interdms_loc_ABAB", "interdms_loc_ABBA",
    "interdms_obj_ABAB", "interdms_obj_ABBA",
    "interdms_ctg_ABAB", "interdms_ctg_ABBA",
]
FAMILY = {"dms": "DMS", "nback": "1-Back", "ctxdm": "CTXDM", "interdms": "InterDMS"}
NAME_RE = re.compile(r"^trial(\d{3})_item(\d{2})_(encoding|delay)$")
N_VERTICES = 64984
N_REGIONS = 360
N_NETWORKS = 12


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_negative(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return not bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"false", "b", "0"}


def is_positive(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1"}


def parse_run(beta_file: Path):
    stem = beta_file.name.replace("_betas.h5", "")
    parts = dict(p.split("-", 1) for p in stem.split("_") if "-" in p)
    return parts["sub"], parts["ses"], parts["task"], int(parts["run"]), stem


def load_atlas(atlas_path: Path, assign_path: Path, valid_mask: np.ndarray):
    lab = np.asarray(nib.load(str(atlas_path)).get_fdata()).squeeze().reshape(-1).astype(int)
    if lab.size != N_VERTICES:
        raise SystemExit(f"Atlas has {lab.size} vertices, expected {N_VERTICES}")
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    if len(region_ids) != N_REGIONS or not np.array_equal(region_ids, np.arange(1, N_REGIONS + 1)):
        raise SystemExit(f"Unexpected Glasser region ids: n={len(region_ids)}")
    # region-averaging matrix over VALID vertices only (same convention as the legacy script)
    W = np.zeros((N_VERTICES, N_REGIONS), dtype=np.float64)
    n_valid_per_region = np.zeros(N_REGIONS, dtype=int)
    for j, rid in enumerate(region_ids):
        idx = np.where((lab == rid) & valid_mask)[0]
        n_valid_per_region[j] = idx.size
        if idx.size:
            W[idx, j] = 1.0 / idx.size
    if (n_valid_per_region == 0).any():
        raise SystemExit("Some Glasser regions have no valid vertices")
    net = np.array([int(l.strip()) for l in assign_path.read_text().strip().splitlines()], dtype=int)
    if net.size != N_REGIONS or net.min() < 1 or net.max() > N_NETWORKS:
        raise SystemExit(f"Bad network assignment file: size={net.size}")
    Wn = np.zeros((N_REGIONS, N_NETWORKS), dtype=np.float64)
    for k in range(1, N_NETWORKS + 1):
        cols = np.where(net == k)[0]
        Wn[cols, k - 1] = 1.0 / cols.size
    return lab, W, Wn, net, n_valid_per_region


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", required=True)
    ap.add_argument("--glm_root", default=str(PROJECT / "data/encoding_delay/trial_level_betas"))
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--verified_labels", default="", help="run_labels_verified.csv from block-file matching")
    ap.add_argument("--atlas_path", default=str(PROJECT / "data/Glasser_LR_Dense64k.dlabel.nii"))
    ap.add_argument("--network_assignments_path", default=str(PROJECT / "data/cortex_parcel_network_assignments.txt"))
    ap.add_argument("--vertex_mask_path", default=str(PROJECT / "data/valid_vertex_mask.npy"))
    ap.add_argument("--save_vertex", action="store_true", help="also store vertex-level delay betas (large)")
    ap.add_argument("--extreme_abs_psc", type=float, default=50.0,
                    help="|PSC| threshold used only to COUNT implausible vertices per record (no exclusion)")
    args = ap.parse_args()

    t0 = time.time()
    subj = args.subj
    glm_root = Path(args.glm_root) / subj
    out_dir = Path(args.out_root) / subj
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = load_valid_vertex_mask(args.vertex_mask_path).astype(bool)
    if valid_mask.size != N_VERTICES:
        raise SystemExit("valid vertex mask size mismatch")
    lab, W, Wn, net, n_valid_per_region = load_atlas(Path(args.atlas_path), Path(args.network_assignments_path), valid_mask)

    lookup_df = pd.read_csv(PROJECT / f"data/study_designs/{subj}_design_design_with_converted.tsv", sep="\t")
    verified = None
    if args.verified_labels:
        verified = pd.read_csv(args.verified_labels)
        verified = verified[verified.subject == subj].copy()
        verified["key"] = verified.session + "_" + verified.bold_task + "_" + verified.run.str.replace("run-", "").astype(int).astype(str)
        verified = verified.set_index("key")

    beta_files = sorted(glm_root.glob("ses-*/func/*_betas.h5"))
    if not beta_files:
        raise SystemExit(f"No betas under {glm_root}")

    records = []
    region_rows, network_rows, vertex_blocks = [], [], []
    inputs = []
    violations = []  # fail-fast at the end with a full list
    seen_keys = set()
    vector_hashes = {}
    skipped_runs = []

    for bf in beta_files:
        s, ses, bold_task, run, stem = parse_run(bf)
        design_csv = bf.with_name(f"{stem}_design.csv")
        if not design_csv.exists():
            skipped_runs.append({"run": stem, "reason": "missing design csv"})
            continue
        df = pd.read_csv(design_csv)
        df.columns = df.columns.str.strip()
        with h5py.File(bf, "r") as f:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["task_regressor_names"]]
            K = f["betas"].shape[1]
            P = f["betas"].shape[0]
            delay_rows = df.index[df["type"].astype(str).str.strip().str.lower() == "delay"].to_numpy()
            cols = df.loc[delay_rows, "trial_index"].to_numpy().astype(int)
            betas_delay = f["betas"][:, :][:, cols].astype(np.float32) if len(cols) else np.zeros((P, 0), np.float32)
        inputs.append({"subject": subj, "run": stem, "betas_h5": str(bf), "h5_bytes": bf.stat().st_size,
                       "h5_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(bf.stat().st_mtime)),
                       "design_csv": str(design_csv), "design_md5": md5_file(design_csv), "K_task": int(K)})

        # ---------- structural assertions (fail fast) ----------
        if P != N_VERTICES:
            violations.append(f"{stem}: betas has {P} vertices")
        if K != len(df) or K != len(names):
            violations.append(f"{stem}: K_task={K} but design rows={len(df)} names={len(names)}")
            continue
        if not np.array_equal(df["trial_index"].to_numpy(), np.arange(K)):
            violations.append(f"{stem}: trial_index is not 0..K-1")
            continue
        for i, name in enumerate(names):
            m = NAME_RE.match(name)
            row = df.iloc[i]
            if not m or int(m.group(1)) != int(row["trialNumber"]) or m.group(3) != str(row["type"]).strip().lower() \
                    or int(m.group(2)) != int(row["item_index"]):
                violations.append(f"{stem}: column {i} name {name} != design row (trial {row['trialNumber']}, item {row['item_index']}, {row['type']})")
                break

        # ---------- task-variant label ----------
        try:
            label_lookup = tci.get_detailed_task_name(subj, ses, bold_task, f"run-{run}", lookup_df, strict=True)
        except Exception as e:  # noqa: BLE001
            label_lookup = f"LOOKUP_FAIL:{type(e).__name__}"
        label_ver, action, resolved = label_lookup, "not_checked", True
        if verified is not None:
            key = f"{ses}_{bold_task}_{run}"
            if key in verified.index:
                vr = verified.loc[key]
                label_ver, action, resolved = str(vr["final_label"]), str(vr["action"]), bool(vr["resolved"])
            else:
                label_ver, action, resolved = label_lookup, "missing_in_verification_table", False

        # ---------- trial-level correctness ----------
        df["_neg"] = df["is_correct"].apply(is_negative)
        df["_pos"] = df["is_correct"].apply(is_positive)
        trial_stats = df.groupby("trialNumber").agg(n_correct=("_pos", "sum"), n_incorrect=("_neg", "sum"))

        # ---------- records ----------
        for j, ridx in enumerate(delay_rows):
            row = df.loc[ridx]
            col = int(row["trial_index"])
            if not names[col].endswith("_delay"):
                violations.append(f"{stem}: selected column {col} ({names[col]}) is not a delay regressor")
            key = (subj, ses, f"run-{run}", col)
            if key in seen_keys:
                violations.append(f"{stem}: duplicate record key {key}")
            seen_keys.add(key)
            vec = betas_delay[:, j]
            h = hashlib.md5(vec.tobytes()).hexdigest()
            if h in vector_hashes:
                violations.append(f"{stem}: column {col} is byte-identical to {vector_hashes[h]}")
            vector_hashes[h] = f"{stem}:col{col}"
            if not np.all(np.isfinite(vec[valid_mask])):
                violations.append(f"{stem}: non-finite values in column {col}")
            ts = trial_stats.loc[row["trialNumber"]]
            n_resp = int(ts["n_correct"] + ts["n_incorrect"])
            if n_resp == 0:
                trial_status = "no_response_recorded"
            elif ts["n_incorrect"] == 0:
                trial_status = "correct"
            else:
                trial_status = "incorrect"
            item_status = "correct" if row["_pos"] else ("incorrect" if row["_neg"] else "no_response_at_item")
            absv = np.abs(vec[valid_mask])
            rec = {
                "record_id": f"{subj}_{ses}_run-{run}_col{col:03d}",
                "subject": subj, "session": ses, "run": f"run-{run}", "bold_task": bold_task,
                "original_trial_index": col, "GLM_beta_column": col, "glm_regressor_name": names[col],
                "trialNumber": int(row["trialNumber"]), "stim_order": int(row["stim_order"]), "item_index": int(row["item_index"]),
                "phase": "delay",
                "task_variant_lookup": label_lookup, "task_variant_verified": label_ver,
                "label_action": action, "label_resolved": resolved,
                "task_family": FAMILY.get(label_lookup.split("_")[0], "unknown"),
                "trial_n_responses": n_resp, "trial_n_correct": int(ts["n_correct"]), "trial_n_incorrect": int(ts["n_incorrect"]),
                "correct_status": trial_status, "item_correct_status": item_status,
                "retain_trial_level": trial_status == "correct",
                "retain_item_level": not row["_neg"],  # legacy row-level rule
                "onset_time": float(row["onset_time"]), "offset_time": float(row["offset_time"]),
                "location": row.get("location"), "category": row.get("category"), "object": row.get("object"),
                "source_beta_file": str(bf), "source_event_file": str(design_csv),
                "vec_md5": h,
                "max_abs_psc_valid": float(absv.max()), "p999_abs_psc_valid": float(np.percentile(absv, 99.9)),
                "n_vertices_abs_gt_thr": int((absv > args.extreme_abs_psc).sum()),
                "mean_psc_valid": float(vec[valid_mask].mean()),
            }
            records.append(rec)
        # aggregate (linear; identical to averaging vertices within region after masking)
        Bd = betas_delay.astype(np.float64)
        Bd[~valid_mask, :] = 0.0
        R = (Bd.T @ W)  # n_delay x 360
        region_rows.append(R.astype(np.float32))
        network_rows.append((R @ Wn).astype(np.float32))
        if args.save_vertex:
            vertex_blocks.append(betas_delay.T)
        print(f"{stem}: {len(delay_rows)} delay betas, label={label_lookup} verified={label_ver} ({action})", flush=True)

    prov = pd.DataFrame(records)
    region = np.concatenate(region_rows, axis=0)
    network = np.concatenate(network_rows, axis=0)
    if len(prov) != region.shape[0]:
        violations.append("provenance rows != region rows")
    if not np.all(np.isfinite(region)) or not np.all(np.isfinite(network)):
        violations.append("non-finite region/network values")

    # ---------- counts per task variant ----------
    counts = []
    for lab_col in ["task_variant_lookup", "task_variant_verified"]:
        for tv, g in prov.groupby(lab_col):
            counts.append({
                "label_source": lab_col, "task_variant": tv,
                "n_runs": g[["session", "run"]].drop_duplicates().shape[0],
                "n_trials_total": g[["session", "run", "trialNumber"]].drop_duplicates().shape[0],
                "n_trials_correct": g[g.retain_trial_level][["session", "run", "trialNumber"]].drop_duplicates().shape[0],
                "n_delay_betas_total": len(g),
                "n_delay_betas_retained_trial_level": int(g.retain_trial_level.sum()),
                "n_delay_betas_retained_item_level": int(g.retain_item_level.sum()),
                "n_delay_betas_unresolved_label": int((~g.label_resolved).sum()),
            })
    counts = pd.DataFrame(counts)

    missing_variants = sorted(set(TASK_VARIANTS) - set(prov.task_variant_lookup.unique()))
    if missing_variants:
        violations.append(f"missing task variants (lookup labels): {missing_variants}")
    unknown = sorted(set(prov.task_variant_lookup.unique()) - set(TASK_VARIANTS))
    if unknown:
        violations.append(f"unknown task variant labels: {unknown}")

    manifest = {
        "subject": subj,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_runs_processed": len(inputs), "n_runs_skipped": len(skipped_runs), "skipped_runs": skipped_runs,
        "n_delay_records": int(len(prov)),
        "n_unique_record_keys": int(len(seen_keys)),
        "n_unique_vector_md5": int(len(vector_hashes)),
        "n_encoding_columns_selected": int((~prov.glm_regressor_name.str.endswith("_delay")).sum()),
        "n_wrong_column_mappings": int(sum("!= design row" in v for v in violations)),
        "n_duplicate_physical_betas": int(len(prov) - len(vector_hashes)),
        "n_nonfinite": int(sum("non-finite" in v for v in violations)),
        "n_retained_trial_level": int(prov.retain_trial_level.sum()),
        "n_retained_item_level": int(prov.retain_item_level.sum()),
        "n_records_unresolved_label": int((~prov.label_resolved).sum()),
        "n_records_relabelled_by_verification": int(prov.label_action.astype(str).str.startswith("relabel").sum()),
        "correct_status_counts": prov.correct_status.value_counts().to_dict(),
        "spatial_axes": {"n_vertices": N_VERTICES, "n_valid_vertices": int(valid_mask.sum()), "n_regions": N_REGIONS,
                          "n_networks": N_NETWORKS, "region_order": "Glasser label ids 1..360 (atlas order)",
                          "network_order": "Cole-Anticevic ids 1..12", "network_of_region": net.tolist(),
                          "n_valid_vertices_per_region": n_valid_per_region.tolist(),
                          "vertex_mask_md5": md5_file(Path(args.vertex_mask_path)), "atlas_md5": md5_file(Path(args.atlas_path)),
                          "network_assignments_md5": md5_file(Path(args.network_assignments_path))},
        "extreme_abs_psc_threshold": args.extreme_abs_psc,
        "n_records_with_any_vertex_gt_thr": int((prov.n_vertices_abs_gt_thr > 0).sum()),
        "violations": violations,
        "passed": len(violations) == 0,
        "elapsed_sec": round(time.time() - t0, 1),
        "command": " ".join(sys.argv),
    }

    prov.to_csv(out_dir / f"{subj}_delay_provenance.csv", index=False)
    counts.to_csv(out_dir / f"{subj}_task_variant_counts.csv", index=False)
    pd.DataFrame(inputs).to_csv(out_dir / f"{subj}_inputs.csv", index=False)
    np.save(out_dir / f"{subj}_delay_region.npy", region)
    np.save(out_dir / f"{subj}_delay_network.npy", network)
    if args.save_vertex:
        with h5py.File(out_dir / f"{subj}_delay_vertex.h5", "w") as f:
            f.create_dataset("delay_betas", data=np.concatenate(vertex_blocks, axis=0), compression="gzip", compression_opts=1)
            f.create_dataset("record_id", data=np.array(prov.record_id.tolist(), dtype="S"))
    with open(out_dir / f"{subj}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({k: v for k, v in manifest.items() if k not in ("spatial_axes", "violations")}, indent=2))
    if violations:
        print("\nVALIDATION FAILED:\n  " + "\n  ".join(violations[:50]), file=sys.stderr)
        sys.exit(1)
    print(f"OK: {subj} clean delay dataset written to {out_dir}")


if __name__ == "__main__":
    main()
