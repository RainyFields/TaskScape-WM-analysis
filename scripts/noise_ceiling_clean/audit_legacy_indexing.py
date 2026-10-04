#!/usr/bin/env python3
"""Simulate the legacy grouping (task_condition_identification.py at commit 6af3058, --correct_only
--target_selection delay_of_second) on the design CSVs only, and count how many stored records pointed to
the wrong GLM column, to an encoding column, or duplicated a physical delay beta.

The legacy code (a) dropped rows with is_correct == False and THEN set col_index = arange(len(df)) so every
row after the first dropped one was shifted onto a lower column; (b) stored the delay following stimulus i2
for every pair (i1,i2) AND for every self pair (i,i), so the same physical beta entered the task pool twice.
The noise-ceiling script then pooled all condition keys of a task variant, so duplicates could land in both
split halves.

Outputs (in --out_dir): legacy_indexing_audit_records.csv (one row per legacy stored record) and
legacy_indexing_audit_summary.csv (per subject x task variant).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import task_condition_identification as tci  # noqa: E402

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]


def is_negative(v):
    if isinstance(v, (bool, np.bool_)):
        return not bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"false", "b", "0"}


def simulate_run(design_csv: Path, task_variant: str):
    df0 = pd.read_csv(design_csv)
    df0.columns = df0.columns.str.strip()
    df0 = tci._canonicalize_design_columns(df0)
    df0["true_col"] = df0["trial_index"].astype(int)
    df0["type_l"] = df0["type"].astype(str).str.lower()
    # ---- legacy behaviour ----
    df = df0[~df0["is_correct"].apply(is_negative)].copy()
    df["col_index"] = np.arange(len(df), dtype=int)  # <-- the bug
    df = df.reset_index(drop=True)
    df["stim_order"] = pd.to_numeric(df["stim_order"], errors="coerce")
    df = df.dropna(subset=["stim_order"])
    df["stim_order"] = df["stim_order"].astype(int)
    pairs = tci.determine_task_type(task_variant)
    out = []
    for _, tdf in df.groupby("trialNumber"):
        enc = tdf[tdf["type_l"].str.contains("encod")]
        dl = tdf[tdf["type_l"].str.contains("delay")]
        for i1, i2 in pairs:
            if enc[enc.stim_order == i1].empty or enc[enc.stim_order == i2].empty:
                continue
            target = dl[dl.stim_order == (i2 if i1 != i2 else i1)]
            if target.empty:
                continue
            r = target.iloc[0]
            legacy_col = int(r["col_index"])
            true_col = int(r["true_col"])
            lc_row = df0.iloc[legacy_col]
            out.append({
                "trialNumber": int(r["trialNumber"]), "pair": f"{i1}-{i2}", "self_pair": i1 == i2,
                "intended_true_col": true_col, "legacy_col": legacy_col,
                "wrong_column": legacy_col != true_col,
                "legacy_col_type": lc_row["type_l"],
                "legacy_col_trialNumber": int(lc_row["trialNumber"]),
                "legacy_col_is_encoding": lc_row["type_l"] == "encoding",
                "legacy_col_other_trial": int(lc_row["trialNumber"]) != int(r["trialNumber"]),
            })
    return out, int(df0["is_correct"].apply(is_negative).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--glm_root", default=str(PROJECT / "data/encoding_delay/trial_level_betas"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for subj in SUBJECTS:
        lookup = pd.read_csv(PROJECT / f"data/study_designs/{subj}_design_design_with_converted.tsv", sep="\t")
        for bf in sorted((Path(args.glm_root) / subj).glob("ses-*/func/*_betas.h5")):
            stem = bf.name.replace("_betas.h5", "")
            parts = dict(p.split("-", 1) for p in stem.split("_") if "-" in p)
            try:
                tv = tci.get_detailed_task_name(subj, parts["ses"], parts["task"], f"run-{parts['run']}", lookup, strict=True)
            except Exception as e:  # noqa: BLE001
                tv = f"LOOKUP_FAIL:{type(e).__name__}"
            recs, n_dropped = simulate_run(bf.with_name(f"{stem}_design.csv"), tv)
            for r in recs:
                r.update({"subject": subj, "session": f"ses-{parts['ses']}", "run": f"run-{int(parts['run'])}", "task_variant": tv,
                          "n_rows_dropped_as_incorrect": n_dropped})
            rows.extend(recs)
        print(subj, "done", flush=True)
    rec = pd.DataFrame(rows)
    rec.to_csv(out / "legacy_indexing_audit_records.csv", index=False)

    def summarize(g):
        # a physical beta is identified by (session, task variant/run, column); different BOLD tasks reuse run numbers
        phys = g[["session", "task_variant", "run", "intended_true_col"]].drop_duplicates().shape[0]
        return pd.Series({
            "n_stored_records": len(g),
            "n_unique_physical_delay_betas_intended": phys,
            "n_duplicate_records": len(g) - phys,
            "frac_duplicate_records": (len(g) - phys) / len(g),
            "n_wrong_column": int(g.wrong_column.sum()),
            "frac_wrong_column": g.wrong_column.mean(),
            "n_encoding_column": int(g.legacy_col_is_encoding.sum()),
            "frac_encoding_column": g.legacy_col_is_encoding.mean(),
            "n_other_trial_column": int(g.legacy_col_other_trial.sum()),
        })

    summ = rec.groupby(["subject", "task_variant"]).apply(summarize).reset_index()
    summ.to_csv(out / "legacy_indexing_audit_summary.csv", index=False)
    tot = rec.groupby("subject").apply(summarize).reset_index()
    tot.to_csv(out / "legacy_indexing_audit_by_subject.csv", index=False)
    print(tot.to_string(index=False))
    allg = summarize(rec)
    print("\nALL SUBJECTS:\n", allg.to_string())
    (out / "legacy_indexing_audit_all.json").write_text(allg.to_json(indent=2))


if __name__ == "__main__":
    main()
