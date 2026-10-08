#!/usr/bin/env python3
"""Aggregate per-run vertex-level R2 maps from encoding-delay GLM betas H5 files."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import paths  # noqa: E402  TaskScape-WM inputs only; rejects the legacy tree

N_VERTICES = 64984
R2_METRICS = ("r2_full", "r2_task", "r2_noise")
# Input (TaskScape-WM): derivatives/glm/<sub>/<ses>/func/*_run-NN_betas.h5 (r2_* datasets; yhat/resid fallback).
# legacy equivalent: data/encoding_delay/trial_level_betas
# Output: <output_root>/r2_averaged_glasser64k/encoding_delay_ssmoothing_psc/<sub>/
# legacy equivalent: data/r2_averaged_glasser64k/encoding_delay_ssmoothing_psc


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subj", required=True, help="Subject ID, e.g. sub-03")
    p.add_argument(
        "--h5_root",
        default=None,
        help="Root containing <subj>/<ses>/func/*_betas.h5 (default: <dataset>/derivatives/glm)",
    )
    p.add_argument(
        "--out_root",
        default=None,
        help="Output root; files go under <out_root>/<subj>/ "
             "(default: <output_root>/r2_averaged_glasser64k/encoding_delay_ssmoothing_psc)",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(R2_METRICS),
        choices=list(R2_METRICS),
    )
    p.add_argument("--dataset_root", default=None, help="TaskScape-WM root (default: config.paths / $TASKSCAPE_DATASET_ROOT)")
    p.add_argument("--output_root", default=None, help="analysis output root (default: config.paths / $TASKSCAPE_OUTPUT_ROOT)")
    a = p.parse_args()
    paths.configure(a.dataset_root, a.output_root)
    paths.install_legacy_guard("aggregate_subject_r2_vertex_maps")
    a.h5_root = str(paths.require_not_legacy(a.h5_root or paths.glm_dir(), "--h5_root"))
    a.out_root = str(paths.require_not_legacy(
        a.out_root or paths.output_dir("r2_averaged_glasser64k", "encoding_delay_ssmoothing_psc"), "--out_root"))
    return a


def parse_h5_path(h5_path: Path) -> tuple[str, str, str]:
    """Return (session, task, run) from filename."""
    name = h5_path.name
    ses_m = re.search(r"(ses-\d+)", name)
    task_m = re.search(r"task-([^_]+)", name)
    run_m = re.search(r"(run-\d+)", name)
    if not (ses_m and task_m and run_m):
        raise ValueError(f"Cannot parse session/task/run from: {h5_path}")
    return ses_m.group(1), task_m.group(1), run_m.group(1)


def _fit_ols_noise_betas(X_noise: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vertex-wise OLS: returns betas_noise with shape (P, K_noise)."""
    betas, _, _, _ = np.linalg.lstsq(X_noise.astype(np.float64), Y.astype(np.float64), rcond=None)
    return betas.T.astype(np.float32)


def compute_vertex_r2_from_h5(f: h5py.File) -> dict[str, np.ndarray]:
    """Load or recompute r2_full, r2_noise, r2_task (64984,) from betas H5."""
    if all(k in f for k in R2_METRICS):
        return {k: f[k][:].astype(np.float32) for k in R2_METRICS}

    if "yhat" not in f:
        raise KeyError(f"Cannot compute R2: missing yhat and stored r2_* in {f.filename}")

    yhat = f["yhat"][:].astype(np.float64)
    if "resid" not in f:
        raise KeyError(f"resid missing in {f.filename}; cannot recompute R2 from yhat alone")
    resid = f["resid"][:].astype(np.float64)
    Y = yhat + resid

    SS_total = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    SS_total = np.where(SS_total < 1e-12, np.nan, SS_total)
    SS_res_full = np.sum(resid**2, axis=0)
    r2_full = (1.0 - SS_res_full / SS_total).astype(np.float32)

    if "design_matrix" not in f or "task_col_start" not in f.attrs:
        if all(k in f for k in ("r2_noise", "r2_task")):
            return {
                "r2_full": r2_full,
                "r2_noise": f["r2_noise"][:].astype(np.float32),
                "r2_task": f["r2_task"][:].astype(np.float32),
            }
        raise KeyError(f"Cannot compute r2_noise/r2_task without design_matrix in {f.filename}")

    X = f["design_matrix"][:].astype(np.float64)
    t0 = int(f.attrs["task_col_start"])
    X_noise = X[:, :t0]
    betas_noise = _fit_ols_noise_betas(X_noise, Y)
    yhat_noise = X_noise @ betas_noise.T
    SS_res_noise = np.sum((Y - yhat_noise) ** 2, axis=0)
    r2_noise = (1.0 - SS_res_noise / SS_total).astype(np.float32)
    r2_task = (r2_full - r2_noise).astype(np.float32)
    return {"r2_full": r2_full, "r2_noise": r2_noise, "r2_task": r2_task}


def load_r2_maps(h5_path: Path, metrics: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        valid = f["valid_mask"][:].astype(bool) if "valid_mask" in f else np.ones(N_VERTICES, dtype=bool)
        computed = compute_vertex_r2_from_h5(f)
        for metric in metrics:
            if metric not in computed:
                raise KeyError(f"{metric} missing for {h5_path}")
            arr = computed[metric]
            if arr.shape[0] != N_VERTICES:
                raise ValueError(f"{h5_path}: {metric} has shape {arr.shape}, expected ({N_VERTICES},)")
            arr = arr.copy()
            arr[~valid] = np.nan
            out[metric] = arr
    return out


def nanmean_stack(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ValueError("empty stack")
    return np.nanmean(np.stack(stack, axis=0), axis=0).astype(np.float32)


def save_map(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def main():
    args = get_args()
    subj_root = Path(args.h5_root) / args.subj
    h5_files = sorted(subj_root.rglob("*_betas.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files under {subj_root}")

    # Collect per-run maps grouped for three averaging schemes.
    by_task: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    by_session: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    global_runs: dict[str, list[np.ndarray]] = defaultdict(list)
    run_manifest: list[dict[str, str]] = []

    for h5_path in h5_files:
        ses, task, run = parse_h5_path(h5_path)
        maps = load_r2_maps(h5_path, args.metrics)
        run_manifest.append(
            {"file": h5_path.name, "session": ses, "task": task, "run": run}
        )
        for metric, arr in maps.items():
            by_task[task][metric].append(arr)
            by_session[ses][metric].append(arr)
            global_runs[metric].append(arr)

    out_subj = Path(args.out_root) / args.subj
    manifest: dict = {
        "subject": args.subj,
        "source": "encoding_delay trial-level GLM (spatial smoothed 4mm FWHM, PSC)",
        "h5_root": str(Path(args.h5_root).resolve()),
        "n_vertices": N_VERTICES,
        "metrics": args.metrics,
        "n_runs_total": len(h5_files),
        "runs": run_manifest,
        "outputs": {},
    }

    # Version 1: within each task, across runs and sessions.
    v1_dir = out_subj / "v1_avg_by_task_across_runs_and_sessions"
    manifest["outputs"]["v1_avg_by_task_across_runs_and_sessions"] = {}
    for task in sorted(by_task):
        manifest["outputs"]["v1_avg_by_task_across_runs_and_sessions"][task] = {}
        for metric in args.metrics:
            avg = nanmean_stack(by_task[task][metric])
            fname = f"{args.subj}_task-{task}_{metric}_vertex_avg.npy"
            out_path = v1_dir / fname
            save_map(out_path, avg)
            manifest["outputs"]["v1_avg_by_task_across_runs_and_sessions"][task][metric] = {
                "path": str(out_path.relative_to(out_subj)),
                "n_runs": len(by_task[task][metric]),
            }

    # Version 2: across all tasks, runs, and sessions.
    v2_dir = out_subj / "v2_avg_global_across_tasks_runs_and_sessions"
    manifest["outputs"]["v2_avg_global_across_tasks_runs_and_sessions"] = {}
    for metric in args.metrics:
        avg = nanmean_stack(global_runs[metric])
        fname = f"{args.subj}_{metric}_vertex_avg.npy"
        out_path = v2_dir / fname
        save_map(out_path, avg)
        manifest["outputs"]["v2_avg_global_across_tasks_runs_and_sessions"][metric] = {
            "path": str(out_path.relative_to(out_subj)),
            "n_runs": len(global_runs[metric]),
        }

    # Version 3: across all tasks within each session.
    v3_dir = out_subj / "v3_avg_by_session_across_tasks_and_runs"
    manifest["outputs"]["v3_avg_by_session_across_tasks_and_runs"] = {}
    for ses in sorted(by_session):
        manifest["outputs"]["v3_avg_by_session_across_tasks_and_runs"][ses] = {}
        for metric in args.metrics:
            avg = nanmean_stack(by_session[ses][metric])
            fname = f"{args.subj}_{ses}_{metric}_vertex_avg.npy"
            out_path = v3_dir / fname
            save_map(out_path, avg)
            manifest["outputs"]["v3_avg_by_session_across_tasks_and_runs"][ses][metric] = {
                "path": str(out_path.relative_to(out_subj)),
                "n_runs": len(by_session[ses][metric]),
            }

    manifest_path = out_subj / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(h5_files)} runs -> {out_subj}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
