#!/usr/bin/env python3
"""Assemble a single BIDS release of the CNeuroMod multi-task working-memory dataset.

Physically copies (dereferencing git-annex symlinks) raw imaging + behavioral
events, fMRIPrep derivatives, and single-trial GLM betas into the hierarchy
described in the manuscript::

    <dataset_root>/
        dataset_description.json
        events.json
        README, CHANGES, .bidsignore
        task-*_bold.json
        sub-XX/ses-YYY/{anat,fmap,func}
        derivatives/fmriprep/sub-XX/...
        derivatives/glm/sub-XX/ses-YYY/func/*_betas.h5 + *_desc-design_events.tsv

Notes
-----
* `eyetrack/` is intentionally omitted (no source data).
* `participants.tsv`/`participants.json` are intentionally omitted for now.
* `sub-02/ses-000` (pilot session, variant tasks) is excluded entirely.
* `*__dup*` duplicate-acquisition files are excluded (not BIDS-valid names).
* func `*_events.tsv` are rebuilt from `data/reformated_behavior` (row order
  matches the GLM beta order). Runs without a reformatted file fall back to the
  original raw wide-format events so no BOLD run is left without events.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import h5py
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
RAW_ROOT = Path(
    "/project/def-pbellec/xuan/cneuromod.multfs.fmriprep/"
    "sourcedata/cneuromod.multfs.raw"
)
FMRIPREP_ROOT = Path("/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
REFORMAT_ROOT = PROJECT / "data" / "reformated_behavior"
BETAS_ROOT = PROJECT / "data" / "encoding_delay" / "trial_level_betas"

TR = 1.49
EXCLUDE_SESSIONS = {("sub-02", "ses-000")}  # pilot session, dropped entirely

RSYNC_BASE = [
    "rsync", "-rLt",
    "--chmod=Du+rwx,Fu+rw,go+r",
    "--exclude=*__dup*",
]

BOLD_RE = re.compile(r"(sub-\d+)_(ses-\d+)_task-([A-Za-z0-9]+)_run-(\d+)_bold\.nii\.gz$")
BETA_RE = re.compile(r"(sub-\d+)_(ses-\d+)_task-([A-Za-z0-9]+)_run-(\d+)_betas\.h5$")

# Mapping of stim_order (1-indexed within trial) -> raw response_{k} column index,
# mirroring scripts/reformat_behavior.py response scoring per task.
RESPONSE_MAP = {
    "1back": {2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    "ctxdm": {3: 2},
    "dms": {2: 1},
    "interdms": {3: 2, 4: 3},
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str]) -> None:
    log("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def sessions_for(sub: str) -> list[str]:
    subdir = RAW_ROOT / sub
    out = []
    for ses in sorted(p.name for p in subdir.glob("ses-*") if p.is_dir()):
        if (sub, ses) in EXCLUDE_SESSIONS:
            continue
        out.append(ses)
    return out


def norm_run(run_label: str) -> str:
    """run-1 -> run-01 (zero-pad to 2 digits)."""
    return f"run-{int(run_label):02d}"


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def broken_symlinks(root: Path) -> list[str]:
    res = subprocess.run(
        ["find", str(root), "-xtype", "l"],
        capture_output=True, text=True,
    )
    return [l for l in res.stdout.splitlines() if l.strip()]


def preflight(dataset_root: Path, subjects: list[str]) -> None:
    log("=== Preflight ===")
    dataset_root.mkdir(parents=True, exist_ok=True)
    # Free space on the target filesystem.
    df = subprocess.run(
        ["df", "-h", str(dataset_root)], capture_output=True, text=True
    )
    log(df.stdout.strip())
    # Broken annex symlinks would make rsync -L fail (no internet on nodes).
    problems = []
    for sub in subjects:
        for root in (RAW_ROOT / sub, FMRIPREP_ROOT / sub):
            if not root.exists():
                continue
            broken = broken_symlinks(root)
            if broken:
                problems += broken
                log(f"  [BROKEN SYMLINKS] {root}: {len(broken)} (annex content not fetched)")
    if problems:
        log(f"ABORT: {len(problems)} broken symlink(s); run `datalad get` first.")
        for p in problems[:20]:
            log("    " + p)
        sys.exit(2)
    log("  Preflight OK (no broken symlinks).")


# --------------------------------------------------------------------------- #
# Component: raw imaging
# --------------------------------------------------------------------------- #
def copy_raw(sub: str, dataset_root: Path) -> None:
    log(f"=== copy_raw {sub} ===")
    for ses in sessions_for(sub):
        src_ses = RAW_ROOT / sub / ses
        dst_ses = dataset_root / sub / ses
        for dtype in ("anat", "fmap", "func"):
            src = src_ses / dtype
            if not src.is_dir():
                continue
            dst = dst_ses / dtype
            dst.mkdir(parents=True, exist_ok=True)
            cmd = list(RSYNC_BASE)
            if dtype == "func":
                # events handled separately; dup files excluded by base.
                cmd += ["--exclude=*_events.tsv", "--exclude=*_event__dup*"]
            cmd += [f"{src}/", f"{dst}/"]
            run(cmd)
        # session-level scans.tsv
        scans = src_ses / f"{sub}_{ses}_scans.tsv"
        if scans.exists():
            dst_ses.mkdir(parents=True, exist_ok=True)
            run(["cp", str(scans), str(dst_ses / scans.name)])


# --------------------------------------------------------------------------- #
# Component: events
# --------------------------------------------------------------------------- #
def _clean_key(v) -> object:
    """Normalize a raw key-press value; missing/no-response -> NA."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return pd.NA
    s = str(v).strip()
    if s in ("", "--", "None", "nan"):
        return pd.NA
    return s


def _raw_response_lookup(raw_events: Path, task: str) -> dict:
    """Map (trial_number, stim_order) -> (response_key, response_time) from the
    raw wide-format events, following the per-task response indexing."""
    rmap = RESPONSE_MAP.get(task, {})
    if not rmap:
        return {}
    raw = pd.read_csv(raw_events, sep="\t", encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    lookup: dict = {}
    for _, r in raw.iterrows():
        tn = r.get("TrialNumber")
        if pd.isna(tn):
            continue
        tn = int(tn)
        for stim_order, k in rmap.items():
            key = _clean_key(r.get(f"response_{k}"))
            rt = r.get(f"response_{k}_time")
            try:
                rt = round(float(rt), 4) if (key is not pd.NA and not pd.isna(rt)) else pd.NA
            except (TypeError, ValueError):
                rt = pd.NA
            lookup[(tn, stim_order)] = (key, rt)
    return lookup


def convert_events(reformatted: Path, out_tsv: Path,
                   task: str | None = None, raw_events: Path | None = None) -> None:
    df = pd.read_csv(reformatted, sep="\t")
    acc_map = {True: 1, False: 0, "True": 1, "False": 0}
    out = pd.DataFrame({
        "onset": df["onset_time"].astype(float).round(4),
        "duration": (df["offset_time"].astype(float) - df["onset_time"].astype(float)).round(4),
        "trial_type": df["type"],
        "trial_number": df["trialNumber"],
        "stim_order": df["stim_order"],
        "location": df["locmod"],
        "category": df["ctgmod"],
        "object": df["objmod"],
        "response": pd.NA,
        "response_time": pd.NA,
        "accuracy": df["is_correct"].map(lambda v: acc_map.get(v, pd.NA)),
    })
    # merge raw key-press / RT onto the response-bearing delay rows
    if task and raw_events and raw_events.exists():
        lookup = _raw_response_lookup(raw_events, task)
        for i, row in out.iterrows():
            if row["trial_type"] != "delay":
                continue
            hit = lookup.get((int(row["trial_number"]), int(row["stim_order"])))
            if hit is not None:
                out.at[i, "response"], out.at[i, "response_time"] = hit
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep="\t", index=False, na_rep="n/a")


def build_events(sub: str, dataset_root: Path) -> None:
    log(f"=== build_events {sub} ===")
    for ses in sessions_for(sub):
        src_func = RAW_ROOT / sub / ses / "func"
        if not src_func.is_dir():
            continue
        dst_func = dataset_root / sub / ses / "func"
        dst_func.mkdir(parents=True, exist_ok=True)
        for bold in sorted(src_func.glob("*_bold.nii.gz")):
            if "__dup" in bold.name:
                continue
            m = BOLD_RE.search(bold.name)
            if not m:
                continue
            _, _, task, run_lbl = m.groups()
            stem = bold.name[: -len("_bold.nii.gz")]  # sub-XX_ses-YYY_task-T_run-RR
            out_tsv = dst_func / f"{stem}_events.tsv"
            reform = REFORMAT_ROOT / sub / ses / "func" / f"{stem}_events.tsv"
            if reform.exists():
                convert_events(reform, out_tsv, task=task,
                               raw_events=src_func / f"{stem}_events.tsv")
            else:
                # fall back to original raw wide-format events
                raw_ev = src_func / f"{stem}_events.tsv"
                if raw_ev.exists():
                    run(["cp", str(raw_ev), str(out_tsv)])
                    log(f"    [fallback raw events] {out_tsv.name}")
                else:
                    log(f"    [WARN] no events for {bold.name}")


# --------------------------------------------------------------------------- #
# Component: fMRIPrep
# --------------------------------------------------------------------------- #
def copy_fmriprep(sub: str, dataset_root: Path) -> None:
    log(f"=== copy_fmriprep {sub} ===")
    dst = dataset_root / "derivatives" / "fmriprep"
    dst.mkdir(parents=True, exist_ok=True)
    # per-subject directory
    run(list(RSYNC_BASE) + [f"{FMRIPREP_ROOT / sub}/", f"{dst / sub}/"])
    # per-subject html reports at the fmriprep top level
    for html in sorted(FMRIPREP_ROOT.glob(f"{sub}_ses-*.html")):
        run(["cp", str(html), str(dst / html.name)])
    main_html = FMRIPREP_ROOT / f"{sub}.html"
    if main_html.exists():
        run(["cp", str(main_html), str(dst / main_html.name)])


# --------------------------------------------------------------------------- #
# Component: GLM betas
# --------------------------------------------------------------------------- #
def export_design_matrix(h5_path: Path, out_tsv: Path) -> None:
    """Write the GLM design matrix used for the betas (post-censoring) as TSV.

    Read from the H5 (not the on-disk *_design_matrix.csv) so the matrix rows
    align with the volumes actually used to estimate the betas.
    """
    with h5py.File(h5_path, "r") as h:
        dm = h["design_matrix"][:]
        names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in h["design_col_names"][:]
        ]
    pd.DataFrame(dm, columns=names).to_csv(out_tsv, sep="\t", index=False)


def copy_glm(sub: str, dataset_root: Path) -> None:
    log(f"=== copy_glm {sub} ===")
    for ses in sessions_for(sub):
        src_func = BETAS_ROOT / sub / ses / "func"
        if not src_func.is_dir():
            continue
        dst_func = dataset_root / "derivatives" / "glm" / sub / ses / "func"
        dst_func.mkdir(parents=True, exist_ok=True)
        for h5 in sorted(src_func.glob("*_betas.h5")):
            m = BETA_RE.search(h5.name)
            if not m:
                continue
            _, _, task, run_lbl = m.groups()
            stem = f"{sub}_{ses}_task-{task}_{norm_run(run_lbl)}"
            run(["cp", str(h5), str(dst_func / f"{stem}_betas.h5")])
            # design matrix used to estimate the betas (exact regressor order)
            export_design_matrix(h5, dst_func / f"{stem}_designmatrix.tsv")


# --------------------------------------------------------------------------- #
# Component: top-level metadata
# --------------------------------------------------------------------------- #
def write_meta(dataset_root: Path) -> None:
    log("=== write_meta ===")
    dataset_root.mkdir(parents=True, exist_ok=True)

    # dataset_description.json (reuse raw authorship)
    raw_dd = json.loads((RAW_ROOT / "dataset_description.json").read_text())
    dd = {
        "Name": "CNeuroMod multi-task working memory fMRI dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "License": "PLACEHOLDER (CC0 / CC-BY-4.0)",
        "Authors": raw_dd.get("Authors", []),
        "Acknowledgements": raw_dd.get("Acknowledgements", ""),
        "HowToAcknowledge": raw_dd.get("HowToAcknowledge", ""),
        "Funding": raw_dd.get("Funding", []),
        "ReferencesAndLinks": raw_dd.get("ReferencesAndLinks", []),
        "OpenNeuroAccession": "PLACEHOLDER (REPLACE-WITH-OPENNEURO-DOI)",
    }
    (dataset_root / "dataset_description.json").write_text(json.dumps(dd, indent=4))

    # top-level events.json data dictionary
    events_dd = {
        "onset": {"LongName": "Event onset", "Description": "Onset of the event relative to the start of the run.", "Units": "s"},
        "duration": {"LongName": "Event duration", "Description": "Duration of the event (offset - onset).", "Units": "s"},
        "trial_type": {"LongName": "Trial phase", "Description": "Phase of the working-memory trial.", "Levels": {"encoding": "Stimulus encoding window.", "delay": "Post-stimulus delay window (2 TR)."}},
        "trial_number": {"LongName": "Trial number", "Description": "1-indexed trial number within the run."},
        "stim_order": {"LongName": "Stimulus order", "Description": "1-indexed position of the stimulus within the trial."},
        "location": {"LongName": "Stimulus location feature", "Description": "Location modality value of the stimulus (task-dependent coding)."},
        "category": {"LongName": "Stimulus category feature", "Description": "Category modality value of the stimulus (0 when not applicable, e.g. DMS)."},
        "object": {"LongName": "Stimulus object/identity feature", "Description": "Object (identity) modality value of the stimulus."},
        "response": {"LongName": "Response key press", "Description": "Raw key press recorded on the response-bearing delay row (n/a if no response or no response expected).", "Levels": {"x": "match / target key", "b": "non-match / non-target key"}},
        "response_time": {"LongName": "Response time", "Description": "Raw recorded time of the key press on the run clock (same time base as onset). n/a if no response.", "Units": "s"},
        "accuracy": {"LongName": "Response accuracy", "Description": "Whether the response on the response-bearing delay row was correct.", "Levels": {"1": "correct", "0": "incorrect"}},
    }
    (dataset_root / "events.json").write_text(json.dumps(events_dd, indent=4))

    # task sidecars from raw top level (only the 4 analyzed tasks; pilot
    # variants 1backcat/1backloc/ctxlco/dmsloc belong only to the dropped ses-000)
    for task in ("1back", "ctxdm", "dms", "interdms"):
        tj = RAW_ROOT / f"task-{task}_bold.json"
        if tj.exists():
            run(["cp", str(tj), str(dataset_root / tj.name)])

    # README / CHANGES / .bidsignore
    (dataset_root / "README").write_text(
        "CNeuroMod multi-task working memory fMRI dataset\n"
        "================================================\n\n"
        "BIDS dataset (Gorgolewski et al., 2016). Raw imaging and behavioral\n"
        "events at the top level; fMRIPrep derivatives and single-trial GLM beta\n"
        "estimates under derivatives/. Eye-tracking and participants metadata are\n"
        "not included in this build.\n\n"
        "Subjects: sub-01, sub-02, sub-03, sub-05, sub-06.\n"
        "Tasks: 1back, ctxdm, dms, interdms.\n\n"
        "derivatives/glm: per-run single-trial betas (*_betas.h5) accompanied by the\n"
        "design matrix used to estimate them (*_designmatrix.tsv; rows = retained\n"
        "volumes, columns = confounds/drifts + per-trial encoding/delay regressors).\n"
        "The beta order matches the design-matrix task-regressor columns.\n\n"
        "func/*_events.tsv: trial-level behavioral events. Stimulus features\n"
        "(location, category, object) are coded per stimulus within each trial, while\n"
        "the GLM regressors are defined over encoding and delay periods, so these\n"
        "event rows are not necessarily in one-to-one correspondence with the beta\n"
        "estimates. Use the design matrix in derivatives/glm for the exact mapping.\n"
    )
    (dataset_root / "CHANGES").write_text(
        f"1.0.0\t{date.today().isoformat()}\n\t- Initial BIDS release build.\n"
    )
    (dataset_root / ".bidsignore").write_text(
        "derivatives/glm/**/*.h5\n"
        "derivatives/glm/**/*_designmatrix.tsv\n"
    )

    # derivatives/glm dataset_description.json
    glm_dir = dataset_root / "derivatives" / "glm"
    glm_dir.mkdir(parents=True, exist_ok=True)
    glm_dd = {
        "Name": "Single-trial GLM beta estimates",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{
            "Name": "glm_analysis.py",
            "Description": "Trial-level encoding+delay OLS GLM on 4mm-FWHM-smoothed "
                           "Glasser64k CIFTI surface data.",
        }],
        "SourceDatasets": [{"URL": "PLACEHOLDER (GITHUB-ANALYSIS-REPO-URL)"}],
    }
    (glm_dir / "dataset_description.json").write_text(json.dumps(glm_dd, indent=4))

    # preserve fMRIPrep's own dataset_description.json
    fp_dst = dataset_root / "derivatives" / "fmriprep"
    fp_dst.mkdir(parents=True, exist_ok=True)
    fp_dd = FMRIPREP_ROOT / "dataset_description.json"
    if fp_dd.exists():
        run(["cp", str(fp_dd), str(fp_dst / fp_dd.name)])
    log("  Metadata written.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
COMPONENT_FUNCS = {
    "raw": copy_raw,
    "events": build_events,
    "fmriprep": copy_fmriprep,
    "glm": copy_glm,
}


def main() -> None:
    import os
    default_root = Path(os.environ.get("SCRATCH", str(PROJECT))) / "cneuromod_multfs_bids"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_root", type=Path, default=default_root)
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument(
        "--components", nargs="+",
        default=["raw", "events", "fmriprep", "glm"],
        choices=["raw", "events", "fmriprep", "glm", "meta"],
    )
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()

    log(f"Dataset root : {args.dataset_root}")
    log(f"Subjects     : {args.subjects}")
    log(f"Components    : {args.components}")

    per_subject = [c for c in args.components if c != "meta"]
    if per_subject and not args.skip_preflight:
        preflight(args.dataset_root, args.subjects)

    for sub in args.subjects:
        for comp in per_subject:
            COMPONENT_FUNCS[comp](sub, args.dataset_root)

    if "meta" in args.components:
        write_meta(args.dataset_root)

    log("DONE.")


if __name__ == "__main__":
    main()
