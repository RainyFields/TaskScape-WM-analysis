#!/usr/bin/env python3
"""
Aggregate per-run trial-level GLM betas into task-level vertex activation maps (Glasser 64k).

For each subject and task, loads all sessions/runs, computes per-run mean activation
(mean across task trial regressors at each vertex), then averages across runs with nanmean.

Outputs (64984,) float32 .npy files suitable for topographic surface figures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from encoding_delay_beta_compliance import is_compliant_h5, iter_run_beta_h5

N_VERTICES = 64984
TASKS = ("1back", "ctxdm", "dms", "interdms")
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05", "sub-06")


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subj", default=None, help="Single subject (default: all SUBJECTS)")
    p.add_argument(
        "--h5_root",
        type=Path,
        default=Path(
            "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas"
        ),
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path(
            "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/"
            "beta_activation_glasser64k/task_averaged"
        ),
    )
    p.add_argument(
        "--require_compliant",
        action="store_true",
        default=True,
        help="Skip H5 files that fail PSC+smoothing compliance check (default: on).",
    )
    p.add_argument(
        "--aggregation",
        choices=("mean_trials_then_runs", "mean_abs_trials_then_runs"),
        default="mean_trials_then_runs",
        help="Per-run map: mean across trial betas, or mean of absolute trial betas.",
    )
    return p.parse_args()


def parse_h5_path(h5_path: Path) -> tuple[str, str, str]:
    name = h5_path.name
    ses_m = re.search(r"(ses-\d+)", name)
    task_m = re.search(r"task-([^_]+)", name)
    run_m = re.search(r"(run-\d+)", name)
    if not (ses_m and task_m and run_m):
        raise ValueError(f"Cannot parse session/task/run from: {h5_path}")
    return ses_m.group(1), task_m.group(1), run_m.group(1)


def load_run_activation(h5_path: Path, aggregation: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        betas = f["betas"][:].astype(np.float32)
        if betas.shape[0] != N_VERTICES:
            raise ValueError(f"{h5_path}: betas shape {betas.shape}, expected ({N_VERTICES}, K)")
        valid = (
            f["valid_mask"][:].astype(bool)
            if "valid_mask" in f
            else np.ones(N_VERTICES, dtype=bool)
        )

    if aggregation == "mean_trials_then_runs":
        act = np.nanmean(betas, axis=1)
    else:
        act = np.nanmean(np.abs(betas), axis=1)

    act = act.astype(np.float32)
    act[~valid] = np.nan
    return act


def nanmean_stack(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ValueError("empty stack")
    return np.nanmean(np.stack(stack, axis=0), axis=0).astype(np.float32)


def process_subject(
    subj: str,
    h5_root: Path,
    out_root: Path,
    require_compliant: bool,
    aggregation: str,
) -> dict:
    subj_dir = h5_root / subj
    if not subj_dir.exists():
        raise FileNotFoundError(f"Missing subject directory: {subj_dir}")

    by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    run_manifest: list[dict[str, str]] = []
    skipped = []

    for h5_path in iter_run_beta_h5(h5_root):
        if subj not in h5_path.parts:
            continue
        if require_compliant:
            ok, diag = is_compliant_h5(h5_path)
            if not ok:
                skipped.append({"file": h5_path.name, "reasons": diag.get("fail_reasons")})
                continue

        ses, task, run = parse_h5_path(h5_path)
        act = load_run_activation(h5_path, aggregation)
        by_task[task].append(act)
        run_manifest.append(
            {"file": h5_path.name, "session": ses, "task": task, "run": run}
        )

    out_subj = out_root / subj
    out_subj.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "subject": subj,
        "n_vertices": N_VERTICES,
        "aggregation": aggregation,
        "source_h5_root": str(h5_root.resolve()),
        "tasks": {},
    }

    for task in TASKS:
        stack = by_task.get(task, [])
        if not stack:
            arr = np.full(N_VERTICES, np.nan, dtype=np.float32)
            n_runs = 0
        else:
            arr = nanmean_stack(stack)
            n_runs = len(stack)

        fname = f"{subj}_task-{task}_beta_activation_vertex_avg.npy"
        out_path = out_subj / fname
        np.save(out_path, arr)
        manifest["tasks"][task] = {
            "path": str(out_path.relative_to(out_root)),
            "n_runs": n_runs,
            "n_finite_vertices": int(np.isfinite(arr).sum()),
        }

    np.save(out_subj / f"{subj}_task_order.npy", np.array(TASKS, dtype=object))
    manifest["task_order"] = list(TASKS)
    manifest["n_runs_total"] = len(run_manifest)
    manifest["runs"] = run_manifest
    manifest["skipped_noncompliant"] = skipped

    info_path = out_subj / f"{subj}_export_info.json"
    info_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    args = get_args()
    subjects = [args.subj] if args.subj else list(SUBJECTS)
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_manifest = {}
    for subj in subjects:
        print(f"Processing {subj}...")
        manifest = process_subject(
            subj,
            args.h5_root.resolve(),
            args.out_root.resolve(),
            args.require_compliant,
            args.aggregation,
        )
        all_manifest[subj] = manifest
        for task, info in manifest["tasks"].items():
            print(f"  task-{task}: {info['n_runs']} runs -> {info['path']}")

    (args.out_root / "file_manifest.json").write_text(
        json.dumps(all_manifest, indent=2) + "\n"
    )
    print(f"Done. Outputs under {args.out_root}")


if __name__ == "__main__":
    main()
