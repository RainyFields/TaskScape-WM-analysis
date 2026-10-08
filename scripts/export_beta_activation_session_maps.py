#!/usr/bin/env python3
"""
Export session-level beta activation vertex maps and apply Level 1 DMS demean.

Upstream H5 layout (example: sub-03_ses-001_task-dms_run-1_betas.h5):
  - betas: (64984, K_task) float32 — axis 0 = vertices, axis 1 = trial regressors
  - valid_mask: (64984,) uint8 — 58,396 valid cortex vertices
  - task_regressor_names: K_task names like trial001_item01_encoding / _delay
  - attrs: psc_equation, smoothing, regressor_level=trial, K_task, etc.

Aggregation (matches aggregate_subject_beta_activation_vertex.py):
  1. Per run: nanmean(betas, axis=1) -> (64984,) run map; invalid vertices -> NaN
  2. Per (subject, session, task): nanmean across runs in that session -> session map
  3. Level 1 correction (before cross-session averaging):
       g_s = nanmean(DMS session map over valid/finite vertices)
       corrected_{T,s} = session_map_{T,s} - g_s
  4. Task-averaged output: nanmean(corrected_{T,s}) across sessions

Session maps are written under --session-out-root/{subj}/.
Corrected task-averaged maps under --corrected-out-root/{subj}/.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding_delay_beta_compliance import is_compliant_h5, iter_run_beta_h5
from config import paths  # noqa: E402  TaskScape-WM inputs only; rejects the legacy tree

N_VERTICES = 64984
BASELINE_TASK = "dms"
TASKS = ("1back", "ctxdm", "dms", "interdms")
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05", "sub-06")
# Input (TaskScape-WM): derivatives/glm/<sub>/<ses>/func/*_run-NN_betas.h5 (complete H5: betas, valid_mask, attrs).
# legacy equivalent: data/encoding_delay/trial_level_betas
# Outputs: <output_root>/beta_activation_glasser64k/{session_level,task_averaged_level1_dms_demean}/<sub>/
# legacy equivalent: data/encoding_delay/beta_activation_glasser64k/{session_level,task_averaged_level1_dms_demean}


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subj", default=None, help="Single subject (default: all SUBJECTS)")
    p.add_argument("--h5-root", type=Path, default=None, help="default: <dataset>/derivatives/glm")
    p.add_argument("--session-out-root", type=Path, default=None,
                   help="default: <output_root>/beta_activation_glasser64k/session_level")
    p.add_argument("--corrected-out-root", type=Path, default=None,
                   help="default: <output_root>/beta_activation_glasser64k/task_averaged_level1_dms_demean")
    p.add_argument("--dataset_root", default=None, help="TaskScape-WM root (default: config.paths / $TASKSCAPE_DATASET_ROOT)")
    p.add_argument("--output_root", default=None, help="analysis output root (default: config.paths / $TASKSCAPE_OUTPUT_ROOT)")
    p.add_argument(
        "--require-compliant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip H5 files failing PSC+smoothing compliance check.",
    )
    p.add_argument(
        "--export-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-session vertex maps.",
    )
    p.add_argument(
        "--apply-level1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply DMS session baseline subtraction and write corrected task averages.",
    )
    a = p.parse_args()
    paths.configure(a.dataset_root, a.output_root)
    paths.install_legacy_guard("export_beta_activation_session_maps")
    a.h5_root = paths.require_not_legacy(a.h5_root or paths.glm_dir(), "--h5-root")
    a.session_out_root = paths.require_not_legacy(
        a.session_out_root or paths.output_dir("beta_activation_glasser64k", "session_level"), "--session-out-root")
    a.corrected_out_root = paths.require_not_legacy(
        a.corrected_out_root or paths.output_dir("beta_activation_glasser64k", "task_averaged_level1_dms_demean"),
        "--corrected-out-root")
    return a


def parse_h5_path(h5_path: Path) -> tuple[str, str, str]:
    name = h5_path.name
    ses_m = re.search(r"(ses-\d+)", name)
    task_m = re.search(r"task-([^_]+)", name)
    run_m = re.search(r"(run-\d+)", name)
    if not (ses_m and task_m and run_m):
        raise ValueError(f"Cannot parse session/task/run from: {h5_path}")
    return ses_m.group(1), task_m.group(1), run_m.group(1)


def load_run_activation(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        betas = f["betas"][:].astype(np.float32)
        if betas.shape[0] != N_VERTICES:
            raise ValueError(f"{h5_path}: betas shape {betas.shape}, expected ({N_VERTICES}, K)")
        valid = (
            f["valid_mask"][:].astype(bool)
            if "valid_mask" in f
            else np.ones(N_VERTICES, dtype=bool)
        )

    act = np.nanmean(betas, axis=1).astype(np.float32)
    act[~valid] = np.nan
    return act


def nanmean_stack(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ValueError("empty stack")
    arr = np.stack(stack, axis=0)
    out = np.full(arr.shape[1:], np.nan, dtype=np.float32)
    valid_counts = np.sum(np.isfinite(arr), axis=0)
    has_data = valid_counts > 0
    out[has_data] = np.nanmean(arr[:, has_data], axis=0)
    return out


def session_global_mean(session_map: np.ndarray) -> float:
    finite = session_map[np.isfinite(session_map)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def collect_run_maps(
    subj: str,
    h5_root: Path,
    require_compliant: bool,
) -> tuple[dict[tuple[str, str], list[np.ndarray]], list[dict], list[dict]]:
    """Return run stacks keyed by (session, task), manifest rows, skipped rows."""
    by_session_task: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    run_manifest: list[dict[str, str]] = []
    skipped: list[dict] = []

    for h5_path in iter_run_beta_h5(h5_root):
        if subj not in h5_path.parts:
            continue
        if require_compliant:
            ok, diag = is_compliant_h5(h5_path)
            if not ok:
                skipped.append({"file": h5_path.name, "reasons": diag.get("fail_reasons")})
                continue

        ses, task, run = parse_h5_path(h5_path)
        act = load_run_activation(h5_path)
        by_session_task[(ses, task)].append(act)
        run_manifest.append(
            {"file": h5_path.name, "session": ses, "task": task, "run": run}
        )

    return by_session_task, run_manifest, skipped


def build_session_maps(
    by_session_task: dict[tuple[str, str], list[np.ndarray]],
) -> dict[tuple[str, str], np.ndarray]:
    return {
        key: nanmean_stack(stack)
        for key, stack in sorted(by_session_task.items())
    }


def export_session_maps(
    subj: str,
    session_maps: dict[tuple[str, str], np.ndarray],
    by_session_task: dict[tuple[str, str], list[np.ndarray]],
    session_out_root: Path,
) -> dict:
    out_subj = session_out_root / subj
    out_subj.mkdir(parents=True, exist_ok=True)

    sessions = sorted({ses for ses, _ in session_maps})
    manifest: dict = {
        "subject": subj,
        "n_vertices": N_VERTICES,
        "aggregation": "mean_trials_then_runs",
        "level": "session",
        "sessions": {},
    }

    for ses in sessions:
        ses_info: dict = {"tasks": {}}
        for task in TASKS:
            key = (ses, task)
            if key not in session_maps:
                arr = np.full(N_VERTICES, np.nan, dtype=np.float32)
                n_runs = 0
            else:
                arr = session_maps[key]
                n_runs = len(by_session_task.get(key, []))

            fname = f"{subj}_{ses}_task-{task}_beta_activation_vertex.npy"
            out_path = out_subj / fname
            np.save(out_path, arr)
            ses_info["tasks"][task] = {
                "path": str(out_path.relative_to(session_out_root)),
                "n_runs": n_runs,
                "n_finite_vertices": int(np.isfinite(arr).sum()),
            }
        manifest["sessions"][ses] = ses_info

    np.save(out_subj / f"{subj}_task_order.npy", np.array(TASKS, dtype=object))
    info_path = out_subj / f"{subj}_session_export_info.json"
    info_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def apply_level1_dms_demean(
    subj: str,
    session_maps: dict[tuple[str, str], np.ndarray],
    corrected_out_root: Path,
) -> dict:
    sessions = sorted({ses for ses, _ in session_maps})
    baseline_scalars: dict[str, float] = {}
    corrected_by_task: dict[str, list[np.ndarray]] = defaultdict(list)

    for ses in sessions:
        dms_key = (ses, BASELINE_TASK)
        if dms_key not in session_maps:
            baseline_scalars[ses] = float("nan")
            continue

        g_s = session_global_mean(session_maps[dms_key])
        baseline_scalars[ses] = g_s

        for task in TASKS:
            key = (ses, task)
            if key not in session_maps:
                continue
            corrected = session_maps[key].astype(np.float32) - np.float32(g_s)
            corrected_by_task[task].append(corrected)

    out_subj = corrected_out_root / subj
    out_subj.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "subject": subj,
        "n_vertices": N_VERTICES,
        "aggregation": "mean_trials_then_runs",
        "correction": "level1_dms_session_global_mean_subtraction",
        "baseline_task": BASELINE_TASK,
        "baseline_scalars": baseline_scalars,
        "tasks": {},
    }

    for task in TASKS:
        stack = corrected_by_task.get(task, [])
        if not stack:
            arr = np.full(N_VERTICES, np.nan, dtype=np.float32)
            n_sessions = 0
        else:
            arr = nanmean_stack(stack)
            n_sessions = len(stack)

        fname = f"{subj}_task-{task}_beta_activation_vertex_avg_level1.npy"
        out_path = out_subj / fname
        np.save(out_path, arr)
        manifest["tasks"][task] = {
            "path": str(out_path.relative_to(corrected_out_root)),
            "n_sessions": n_sessions,
            "n_finite_vertices": int(np.isfinite(arr).sum()),
        }

    np.save(out_subj / f"{subj}_task_order.npy", np.array(TASKS, dtype=object))
    info_path = out_subj / f"{subj}_export_info.json"
    info_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def process_subject(
    subj: str,
    h5_root: Path,
    session_out_root: Path,
    corrected_out_root: Path,
    require_compliant: bool,
    export_sessions: bool,
    apply_level1: bool,
) -> dict:
    by_session_task, run_manifest, skipped = collect_run_maps(
        subj, h5_root, require_compliant
    )
    session_maps = build_session_maps(by_session_task)

    result: dict = {
        "subject": subj,
        "n_runs_total": len(run_manifest),
        "runs": run_manifest,
        "skipped_noncompliant": skipped,
        "n_session_task_maps": len(session_maps),
    }

    if export_sessions:
        result["session_manifest"] = export_session_maps(
            subj, session_maps, by_session_task, session_out_root
        )

    if apply_level1:
        result["corrected_manifest"] = apply_level1_dms_demean(
            subj, session_maps, corrected_out_root
        )

    return result


def main():
    args = get_args()
    subjects = [args.subj] if args.subj else list(SUBJECTS)
    h5_root = args.h5_root.resolve()

    if args.export_sessions:
        args.session_out_root.mkdir(parents=True, exist_ok=True)
    if args.apply_level1:
        args.corrected_out_root.mkdir(parents=True, exist_ok=True)

    all_manifest: dict = {}
    for subj in subjects:
        print(f"Processing {subj}...")
        manifest = process_subject(
            subj,
            h5_root,
            args.session_out_root.resolve(),
            args.corrected_out_root.resolve(),
            args.require_compliant,
            args.export_sessions,
            args.apply_level1,
        )
        all_manifest[subj] = manifest
        print(f"  runs: {manifest['n_runs_total']}, session-task maps: {manifest['n_session_task_maps']}")
        if args.apply_level1 and "corrected_manifest" in manifest:
            for task, info in manifest["corrected_manifest"]["tasks"].items():
                print(f"  level1 task-{task}: {info['n_sessions']} sessions -> {info['path']}")

    manifest_path = args.corrected_out_root / "file_manifest.json"
    if args.apply_level1:
        manifest_path.write_text(json.dumps(all_manifest, indent=2) + "\n")
        print(f"Done. Session maps: {args.session_out_root}")
        print(f"Done. Level-1 corrected averages: {args.corrected_out_root}")
    elif args.export_sessions:
        (args.session_out_root / "file_manifest.json").write_text(
            json.dumps(all_manifest, indent=2) + "\n"
        )
        print(f"Done. Session maps: {args.session_out_root}")


if __name__ == "__main__":
    main()
