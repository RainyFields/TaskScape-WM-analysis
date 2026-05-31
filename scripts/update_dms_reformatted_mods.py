#!/usr/bin/env python3
"""
Update reformatted DMS behavior TSVs with objmod/ctgmod/angmod/locmod from (ref, loc).

Uses the non-DMS LOC stimulus bank (stim 0-15 table): for DMS, stim_index = ref + 8*loc
maps to the same object/category/viewpoint features as interdms/ctxdm/1back_loc.

Reads loc and ref from raw behavioral files only; writes reformatted_behavior only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
RAW_ROOT = ROOT / "data" / "behavior"
REFORM_ROOT = ROOT / "data" / "reformated_behavior"

# (ref, loc) -> (objmod, ctgmod, angmod, locmod) from non-DMS LOC stim table
REF_LOC_LOOKUP: dict[tuple[int, int], tuple[int, int, int, int]] = {}
for ref in range(8):
    for loc in (0, 1):
        REF_LOC_LOOKUP[(ref, loc)] = (
            ref // 2,      # objmod
            ref // 4,      # ctgmod
            ref % 2,       # angmod
            loc,           # locmod
        )


def raw_path_for_reformatted(reform_path: Path) -> Path | None:
    rel = reform_path.relative_to(REFORM_ROOT)
    raw = RAW_ROOT / rel
    return raw if raw.exists() else None


def trial_ref_loc_map(raw_df: pd.DataFrame) -> dict[tuple[int, int], tuple[int, int]]:
    """(trialNumber, stim_order) -> (ref, loc)."""
    raw_df = raw_df.copy()
    raw_df.columns = raw_df.columns.str.strip()
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for _, row in raw_df.iterrows():
        trial = int(row["TrialNumber"])
        for stim_order in (1, 2):
            ref = int(row[f"ref{stim_order}"])
            loc = int(row[f"loc{stim_order}"])
            out[(trial, stim_order)] = (ref, loc)
    return out


def update_reformatted_file(reform_path: Path) -> bool:
    raw_path = raw_path_for_reformatted(reform_path)
    if raw_path is None:
        print(f"[skip] no raw file for {reform_path}")
        return False

    raw_df = pd.read_csv(raw_path, sep="\t")
    if "TrialNumber" not in raw_df.columns:
        print(f"[skip] not a behavioral TSV: {raw_path}")
        return False

    ref_loc_by_trial = trial_ref_loc_map(raw_df)

    df = pd.read_csv(reform_path, sep="\t")
    df.columns = df.columns.str.strip()

    objmods, ctgmods, angmods, locmods = [], [], [], []
    for _, row in df.iterrows():
        key = (int(row["trialNumber"]), int(row["stim_order"]))
        if key not in ref_loc_by_trial:
            raise KeyError(f"Missing ref/loc for trial {key} in {raw_path}")
        ref, loc = ref_loc_by_trial[key]
        objmod, ctgmod, angmod, locmod = REF_LOC_LOOKUP[(ref, loc)]
        objmods.append(objmod)
        ctgmods.append(ctgmod)
        angmods.append(angmod)
        locmods.append(locmod)

    df["locmod"] = locmods
    df["ctgmod"] = ctgmods
    df["objmod"] = objmods
    df["angmod"] = angmods

    col_order = [
        "trialNumber",
        "stim_order",
        "locmod",
        "ctgmod",
        "objmod",
        "angmod",
        "type",
        "is_correct",
        "onset_time",
        "offset_time",
    ]
    df = df[col_order]
    df.to_csv(reform_path, sep="\t", index=False)
    return True


def main() -> None:
    paths = sorted(REFORM_ROOT.glob("sub-*/ses-*/func/*task-dms*_events.tsv"))
    ok = 0
    for p in paths:
        if update_reformatted_file(p):
            ok += 1
            print(f"updated {p}")
    print(f"\nDone: {ok}/{len(paths)} reformatted DMS files updated.")


if __name__ == "__main__":
    main()
