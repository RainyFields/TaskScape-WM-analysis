#!/usr/bin/env python3
"""Cross-subject validation of the clean delay-beta dataset built by build_clean_delay_betas.py.

Fails (non-zero exit) on: any per-subject assertion failure, wrong-column vectors, duplicate records,
encoding betas in the delay set, differing CIFTI axis across subjects or vs the Glasser atlas, empty
Glasser parcel / network after masking, NaN/Inf, missing task variants, or trial counts that are not
fully explained by (legacy records - duplicates) and correct-only filtering.

Outputs (in --out_dir): validation_table.csv (per subject x task), validation_report.md, validation_summary.json
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py, nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import paths  # TaskScape-WM inputs only; rejects the legacy tree
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
EXPECTED_VARIANTS = ["dms_loc", "nback_loc", "nback_ctg", "nback_obj", "ctxdm_lco", "ctxdm_col",
                     "interdms_loc_ABAB", "interdms_loc_ABBA", "interdms_ctg_ABAB", "interdms_ctg_ABBA",
                     "interdms_obj_ABAB", "interdms_obj_ABBA"]


def md_table(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def atlas_axis_sha1():
    img = nib.load(str(paths.glasser_dlabel()))
    ax = img.header.get_axis(1)
    h = hashlib.sha1(); h.update(np.asarray(ax.vertex).astype(np.int64).tobytes()); h.update("|".join(ax.name).encode())
    h.update(json.dumps(dict(ax.nvertices), sort_keys=True).encode())
    return h.hexdigest(), np.asarray(img.get_fdata()).reshape(-1).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="default: <dataset>/derivatives/glm-clean-delay (per-subject folders) or a flat dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--output_root", default=None)
    args = ap.parse_args()
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard("validate_clean_delay_betas")
    data = paths.require_not_legacy(Path(args.data_dir or paths.clean_delay_dir()), "--data_dir")
    out = paths.require_not_legacy(Path(args.out_dir), "--out_dir"); out.mkdir(parents=True, exist_ok=True)
    failures = []

    atlas_sha, labels = atlas_axis_sha1()
    mask = np.load(paths.valid_vertex_mask())
    net = np.loadtxt(paths.parcel_networks()).astype(int)
    parcel_counts = {p: int(((labels == p) & mask).sum()) for p in range(1, 361)}
    empty_parcels = [p for p, n in parcel_counts.items() if n == 0]
    if empty_parcels:
        failures.append(f"empty Glasser parcels after masking: {empty_parcels}")
    net_counts = {int(n): int(sum(parcel_counts[p] for p in range(1, 361) if net[p - 1] == n)) for n in sorted(set(net))}
    if any(v == 0 for v in net_counts.values()):
        failures.append(f"empty network: {net_counts}")

    rows, subj_summ = [], []
    for subj in SUBJECTS:
        vj = ((data / subj) if (data / subj).is_dir() else data) / f"{subj}_validation.json"
        if not vj.exists():
            failures.append(f"{subj}: validation.json missing (builder failed?)"); continue
        v = json.loads(vj.read_text())
        prov = pd.read_csv(((data / subj) if (data / subj).is_dir() else data) / f"{subj}_provenance.csv")
        cnt = pd.read_csv(((data / subj) if (data / subj).is_dir() else data) / f"{subj}_counts_per_task.csv")
        with h5py.File(((data / subj) if (data / subj).is_dir() else data) / f"{subj}_delay_betas.h5", "r") as f:
            n_rec, P = f["betas"].shape; sha = f.attrs["cifti_axis_sha1"]
            sample = f["betas"][:64]
        if sha != atlas_sha:
            failures.append(f"{subj}: CIFTI axis {sha[:8]} != atlas {atlas_sha[:8]}")
        if v["vector_verification"]["wrong_column"] != 0:
            failures.append(f"{subj}: {v['vector_verification']['wrong_column']} wrong-column vectors")
        if v["n_duplicate_keys"] or v["n_physically_identical_vectors"]:
            failures.append(f"{subj}: duplicates keys={v['n_duplicate_keys']} identical={v['n_physically_identical_vectors']}")
        if v["n_encoding_in_delay_dataset"]:
            failures.append(f"{subj}: encoding betas in delay set")
        if v["n_nan_inf"] or not np.all(np.isfinite(sample)):
            failures.append(f"{subj}: NaN/Inf")
        if n_rec != v["n_delay_retained"] or n_rec != int(prov.retained.sum()):
            failures.append(f"{subj}: record count mismatch h5={n_rec} json={v['n_delay_retained']} prov={int(prov.retained.sum())}")
        missing = sorted(set(EXPECTED_VARIANTS) - set(v["task_variants"]))
        if missing:
            failures.append(f"{subj}: missing task variants {missing}")
        extra = sorted(set(v["task_variants"]) - set(EXPECTED_VARIANTS))
        if extra:
            failures.append(f"{subj}: unexpected task variants {extra}")
        # count reconciliation: delay rows = retained + excluded; legacy = retained + duplicates
        bad = cnt[(cnt.n_delay_rows != cnt.n_retained + cnt.excluded_incorrect) | (cnt.legacy_records != cnt.n_retained + cnt.duplicates_removed)]
        if len(bad):
            failures.append(f"{subj}: unexplained count change in {bad.task_variant.tolist()}")
        for r in cnt.itertuples():
            rows.append(dict(subject=subj, task_variant=r.task_variant, n_runs=r.n_runs, n_glm_columns=r.n_glm_columns,
                             n_delay_rows=r.n_delay_rows, retained_delay_betas=r.n_retained, excluded_incorrect=r.excluded_incorrect,
                             legacy_records_pair_plus_single=r.legacy_records, duplicates_removed=r.duplicates_removed))
        subj_summ.append(dict(subject=subj, n_runs=v["n_runs"], n_glm_columns=v["n_glm_columns_total"], n_delay_rows=v["n_delay_rows"],
                              retained=v["n_delay_retained"], excluded_incorrect=v["n_delay_excluded_incorrect"],
                              legacy_records=v["n_legacy_records"], duplicates_removed=v["n_duplicates_removed"],
                              runs_relabelled=v["n_runs_relabelled"], runs_unresolved=v["n_runs_unresolved"],
                              vectors_checked=v["vector_verification"]["checked"], wrong_column=v["vector_verification"]["wrong_column"],
                              const_masked_vertices=v["n_masked_vertices_const"], abs_beta_p99_9=round(v["masked_abs_beta_percentiles"]["99.9"], 2),
                              abs_beta_max=round(v["masked_abs_beta_percentiles"]["100"], 1),
                              records_absbeta_gt100=v["n_records_with_abs_beta_gt_100"], cifti_axis_sha1=sha[:10]))
    tab = pd.DataFrame(rows); tab.to_csv(out / "validation_table.csv", index=False)
    summ = pd.DataFrame(subj_summ); summ.to_csv(out / "validation_per_subject.csv", index=False)
    if len(set(summ.cifti_axis_sha1)) > 1:
        failures.append("CIFTI axis differs across subjects")

    status = "PASS" if not failures else "FAIL"
    lines = [f"# Clean delay-beta dataset validation: **{status}**", "",
             f"Atlas CIFTI brain-model axis sha1: `{atlas_sha[:10]}`; all subject datasets must match it.",
             f"Glasser parcels with valid vertices: {360 - len(empty_parcels)}/360 (min {min(parcel_counts.values())} vertices). "
             f"Networks: {len(net_counts)} (min {min(net_counts.values())} vertices).", "",
             "## Failures", ""] + ([f"- {f}" for f in failures] or ["none"]) + \
            ["", "## Per subject", "", md_table(summ), "", "## Per subject x task variant", "", md_table(tab)]
    (out / "validation_report.md").write_text("\n".join(lines) + "\n")
    json.dump(dict(status=status, failures=failures, atlas_axis_sha1=atlas_sha, parcel_min_vertices=min(parcel_counts.values()),
                   network_vertex_counts=net_counts), open(out / "validation_summary.json", "w"), indent=2)
    print("\n".join(lines[:12]))
    print(summ.to_string(index=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
