#!/usr/bin/env python3
"""Pool adjusted R² across all BOLD tasks and export a publication-style distribution figure."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_SANITY_ROOT = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _SANITY_ROOT.parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
if str(_SANITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SANITY_ROOT))

# EXTERNAL style dependency: /project/def-pbellec/xuan/figure_style (separate repo; not legacy, not dataset)
_PROJECT_PARENT = Path(__file__).resolve().parents[3]
if str(_PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_PARENT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from glm_evaluation import (  # noqa: E402
    A4_WIDTH_MM,
    DIST_FIG_WIDTH_MM,
    plot_adjusted_r2_distribution_paper,
)
from utils import load_valid_vertex_mask  # noqa: E402  mask = <dataset>/derivatives/taskscape-surface/atlas/valid_vertex_mask.npy
from config import paths  # noqa: E402  TaskScape-WM inputs only; rejects the legacy tree

# Input: derivatives/glm/<sub>/<ses>/func/*_betas.h5 (r2_*_adj datasets).
# legacy equivalent: data/encoding_delay/trial_level_betas
# Output: <output_root>/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc
# legacy equivalent: scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc

DEFAULT_TASKS = ("1back", "ctxdm", "dms", "interdms")
DEFAULT_SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05", "sub-06")
ADJ_R2_KEYS = ("r2_full_adj", "r2_noise_adj", "r2_task_adj")

_STYLE_PATH = _PROJECT_PARENT / "figure_style" / "nature_style.mplstyle"
plt.style.use(str(_STYLE_PATH))


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--glm_root",
        default=None,
        help="Root containing <subj>/<ses>/func/*_betas.h5 (default: <dataset>/derivatives/glm)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="default: <output_root>/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc",
    )
    p.add_argument("--dataset_root", default=None, help="TaskScape-WM root (default: config.paths / $TASKSCAPE_DATASET_ROOT)")
    p.add_argument("--output_root", default=None, help="analysis output root (default: config.paths / $TASKSCAPE_OUTPUT_ROOT)")
    p.add_argument(
        "--subjects",
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="Subjects to include (default: all five CNeuroMod subjects).",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="BOLD task names to include (default: all four tasks).",
    )
    p.add_argument(
        "--dist_full_width_mm",
        type=float,
        default=DIST_FIG_WIDTH_MM,
        help="Figure width in mm (default: 1/2 A4 = 105 mm).",
    )
    p.add_argument(
        "--dist_hw_ratio",
        type=float,
        default=(2.0 / 3.0),
        help="Figure height/width ratio (default: 2:3).",
    )
    p.add_argument(
        "--journal",
        default="nature",
        choices=["nature", "pnas", "cell", "science", "plos", "neuron", "jneurosci"],
    )
    p.add_argument(
        "--dist_span",
        default="single",
        choices=["single", "1.5", "double"],
        help="Column span fallback when --dist_full_width_mm is not set.",
    )
    a = p.parse_args()
    paths.configure(a.dataset_root, a.output_root)
    paths.install_legacy_guard("plot_adjusted_r2_distribution_all_tasks")
    a.glm_root = str(paths.require_not_legacy(a.glm_root or paths.glm_dir(), "--glm_root"))
    a.output_dir = str(paths.require_not_legacy(
        a.output_dir or paths.output_dir("glm_fitting_check_results", "customized_glm", "encoding_delay_ssmoothing_psc"),
        "--output_dir"))
    return a


def _task_from_h5_name(name: str) -> str | None:
    m = re.search(r"task-([^_]+)", name)
    return m.group(1) if m else None


def _subject_from_h5_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return None


def collect_adjusted_r2(
    glm_root: Path,
    subjects: set[str],
    tasks: set[str],
    valid_mask: np.ndarray,
) -> tuple[dict[str, list[np.ndarray]], list[Path]]:
    """Load finite adjusted-R² values from all matching H5 files."""
    pooled: dict[str, list[np.ndarray]] = {k: [] for k in ADJ_R2_KEYS}
    used_files: list[Path] = []

    for h5_path in sorted(glm_root.glob("**/func/*_betas.h5")):
        subj = _subject_from_h5_path(h5_path)
        task = _task_from_h5_name(h5_path.name)
        if subj not in subjects or task not in tasks:
            continue

        with h5py.File(h5_path, "r") as f:
            if not all(k in f for k in ADJ_R2_KEYS):
                continue
            used_files.append(h5_path)
            for key in ADJ_R2_KEYS:
                vals = f[key][:].astype(np.float32)[valid_mask]
                pooled[key].append(vals[np.isfinite(vals)])

    return pooled, used_files


def write_summary_json(out_base, used_files, arrays: dict, extra: dict | None = None) -> None:
    """Reviewable numeric sidecar for the figure (same inputs as the plot): n, mean, sd, median, quantiles per metric."""
    import json
    qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    summ = {"n_runs": len(used_files), "runs": sorted(str(Path(f).name) for f in used_files)}
    for k, v in arrays.items():
        v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
        summ[k] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(np.median(v)),
                   **{f"q{int(q*100):02d}": float(np.quantile(v, q)) for q in qs}}
    if extra: summ.update(extra)
    Path(str(out_base) + "_summary.json").write_text(json.dumps(summ, indent=2))


def main():
    args = get_args()
    glm_root = Path(args.glm_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    subjects = set(args.subjects)
    tasks = set(args.tasks)
    valid_mask = load_valid_vertex_mask()

    pooled, used_files = collect_adjusted_r2(glm_root, subjects, tasks, valid_mask)
    if not used_files:
        raise SystemExit(
            f"No H5 files with adjusted R² found under {glm_root} "
            f"for subjects={sorted(subjects)} tasks={sorted(tasks)}"
        )

    r2_full = np.concatenate(pooled["r2_full_adj"])
    r2_noise = np.concatenate(pooled["r2_noise_adj"])
    r2_task = np.concatenate(pooled["r2_task_adj"])

    task_label = "all_tasks"
    out_stem = f"{task_label}_adjusted_r2_distribution_paper"
    title = "Adjusted R2 distributions (all tasks)"

    out_base = plot_adjusted_r2_distribution_paper(
        r2_full_adj=r2_full,
        r2_noise_adj=r2_noise,
        r2_task_adj=r2_task,
        outdir=outdir,
        out_stem=out_stem,
        title=title,
        journal=args.journal,
        span=args.dist_span,
        full_width_mm=args.dist_full_width_mm,
        hw_ratio=args.dist_hw_ratio,
    )

    write_summary_json(out_base, used_files, {"r2_full_adj": r2_full, "r2_noise_adj": r2_noise, "r2_task_adj": r2_task},
                       {"subjects": sorted(subjects), "tasks": sorted(tasks), "n_valid_vertices": int(np.sum(valid_mask)), "glm_root": str(glm_root)})
    print(f"Pooled {len(used_files)} runs across tasks={sorted(tasks)} subjects={sorted(subjects)}")
    print(f"Observations per metric: full={r2_full.size:,}, noise={r2_noise.size:,}, task={r2_task.size:,}")
    print(f"Figure width: {args.dist_full_width_mm:.1f} mm ({args.dist_full_width_mm / A4_WIDTH_MM:.0%} A4)")
    print(f"Figure height: {args.dist_full_width_mm * args.dist_hw_ratio:.1f} mm (H:W = {args.dist_hw_ratio:.3f})")
    print("Saved:")
    print(" -", out_base.with_suffix(".svg"))
    print(" -", out_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
