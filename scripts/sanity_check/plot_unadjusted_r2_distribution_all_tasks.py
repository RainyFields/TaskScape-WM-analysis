#!/usr/bin/env python3
"""Pool unadjusted R² across all BOLD tasks and export Figure 5c.

This is the unadjusted counterpart of plot_adjusted_r2_distribution_all_tasks.py.
It does not read or plot r2_*_adj, and it does not overwrite the adjusted figure.

The stored GLMs omit DCT drift columns (Nilearn names them drift_*; the GLM
keeps only cosine* and constant). Regenerate this figure if those GLMs are refit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

import h5py  # noqa: E402

from figure_style.figure_utils import apply_clean_spines, get_palette, mm_to_inches, save_figure  # noqa: E402
from glm_evaluation import A4_WIDTH_MM, DIST_FIG_WIDTH_MM  # noqa: E402
from utils import load_valid_vertex_mask  # noqa: E402  mask = <dataset>/derivatives/taskscape-surface/atlas/valid_vertex_mask.npy
from config import paths  # noqa: E402  TaskScape-WM inputs only; rejects the legacy tree

# Input: derivatives/glm/<sub>/<ses>/func/*_betas.h5 (r2_full/r2_noise/r2_task datasets).
# legacy equivalent: data/encoding_delay/trial_level_betas
# Output: <output_root>/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc/unadjusted_r2_distribution
# legacy equivalent: scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc/unadjusted_r2_distribution

DEFAULT_TASKS = ("1back", "ctxdm", "dms", "interdms")
DEFAULT_SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05", "sub-06")
UNADJ_R2_KEYS = ("r2_full", "r2_noise", "r2_task")
N_BINS = 80
# Same print size as the adjusted histogram (80% of half an A4 page, H:W = 2:3).
DEFAULT_HW_RATIO = 2.0 / 3.0

_STYLE_PATH = _PROJECT_PARENT / "figure_style" / "nature_style.mplstyle"
plt.style.use(str(_STYLE_PATH))
# Sans-serif math so subscripts match the Arial/DejaVu panel rather than Computer Modern.
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"


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
        help="default: <output_root>/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc/"
             "unadjusted_r2_distribution",
    )
    p.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--dist_full_width_mm", type=float, default=DIST_FIG_WIDTH_MM)
    p.add_argument("--dist_hw_ratio", type=float, default=DEFAULT_HW_RATIO)
    p.add_argument("--n_bins", type=int, default=N_BINS)
    p.add_argument("--dataset_root", default=None, help="TaskScape-WM root (default: config.paths / $TASKSCAPE_DATASET_ROOT)")
    p.add_argument("--output_root", default=None, help="analysis output root (default: config.paths / $TASKSCAPE_OUTPUT_ROOT)")
    a = p.parse_args()
    paths.configure(a.dataset_root, a.output_root)
    paths.install_legacy_guard("plot_unadjusted_r2_distribution_all_tasks")
    a.glm_root = str(paths.require_not_legacy(a.glm_root or paths.glm_dir(), "--glm_root"))
    a.output_dir = str(paths.require_not_legacy(
        a.output_dir or paths.output_dir("glm_fitting_check_results", "customized_glm",
                                         "encoding_delay_ssmoothing_psc", "unadjusted_r2_distribution"),
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


def collect_unadjusted_r2(
    glm_root: Path,
    subjects: set[str],
    tasks: set[str],
    valid_mask: np.ndarray,
) -> tuple[dict[str, list[np.ndarray]], list[Path], float]:
    """Load finite unadjusted R² from matching H5 files.

    Selection matches the adjusted histogram: global valid-vertex mask, then
    finite values, with no run or subject averaging. Also returns the maximum
    absolute departure of r2_task from r2_full - r2_noise on vertices finite
    in all three arrays.
    """
    pooled: dict[str, list[np.ndarray]] = {k: [] for k in UNADJ_R2_KEYS}
    used_files: list[Path] = []
    max_abs_diff = 0.0

    for h5_path in sorted(glm_root.glob("**/func/*_betas.h5")):
        subj = _subject_from_h5_path(h5_path)
        task = _task_from_h5_name(h5_path.name)
        if subj not in subjects or task not in tasks:
            continue

        with h5py.File(h5_path, "r") as f:
            if not all(k in f for k in UNADJ_R2_KEYS):
                continue
            used_files.append(h5_path)
            arrays = {key: f[key][:].astype(np.float64)[valid_mask] for key in UNADJ_R2_KEYS}
            finite_all = np.ones(arrays["r2_full"].shape[0], dtype=bool)
            for key in UNADJ_R2_KEYS:
                vals = arrays[key]
                finite = np.isfinite(vals)
                finite_all &= finite
                pooled[key].append(vals[finite].astype(np.float32))
            if np.any(finite_all):
                diff = arrays["r2_task"][finite_all] - (
                    arrays["r2_full"][finite_all] - arrays["r2_noise"][finite_all]
                )
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(diff))))

    return pooled, used_files, max_abs_diff


def _summarize(name: str, values: np.ndarray) -> str:
    n_neg = int(np.sum(values < 0))
    return (
        f"{name}: n={values.size:,} min={float(values.min()):.6g} "
        f"max={float(values.max()):.6g} mean={float(values.mean()):.6g} "
        f"n_negative={n_neg}"
    )


def plot_unadjusted_r2_distribution(
    r2_full: np.ndarray,
    r2_noise: np.ndarray,
    r2_task: np.ndarray,
    outdir: Path,
    out_stem: str,
    full_width_mm: float,
    hw_ratio: float,
    n_bins: int,
) -> Path:
    """Single-panel unadjusted R² histogram matching the adjusted figure's style."""
    panel_w_mm = float(full_width_mm)
    panel_h_mm = panel_w_mm * float(hw_ratio)
    figsize = (mm_to_inches(panel_w_mm), mm_to_inches(panel_h_mm))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    c_task, c_noise, c_full = get_palette("vigil")

    data_min = min(float(r2_full.min()), float(r2_noise.min()), float(r2_task.min()))
    data_max = max(float(r2_full.max()), float(r2_noise.max()), float(r2_task.max()))
    # Display range is [0, 1] unless a value falls outside it. Do not clip.
    lo = min(0.0, data_min)
    hi = max(1.0, data_max)
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    pad = 0.02 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)

    series = (
        (r2_full, c_full, r"$R^2_{\mathrm{full}}$"),
        (r2_noise, c_noise, r"$R^2_{\mathrm{noise}}$"),
        (r2_task, c_task, r"$R^2_{\mathrm{task}}$"),
    )
    for values, color, label in series:
        ax.hist(
            values,
            bins=edges,
            density=True,
            alpha=0.45,
            color=color,
            label=label,
            linewidth=0.0,
        )

    ax.set_xlabel(r"Unadjusted $R^2$")
    ax.set_ylabel("Density")
    # Upper left matched the adjusted histogram. These unadjusted densities
    # peak on the right, so the same corner stays clear of the curves.
    # 8 pt (rather than the 6 pt adjusted-figure legend) keeps the math
    # subscripts at about 5.6 pt on this 84 mm panel.
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.4, fontsize=8, labelspacing=0.35)
    apply_clean_spines(ax)
    ax.text(
        -0.02,
        1.04,
        "c",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    # Room for the panel letter above the axes and the y-axis label.
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.16)

    out_base = Path(outdir) / out_stem
    save_figure(fig, out_base, formats=("svg", "pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_base


def main():
    args = get_args()
    glm_root = Path(args.glm_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    valid_mask = load_valid_vertex_mask()
    pooled, used_files, max_abs_diff = collect_unadjusted_r2(
        glm_root, set(args.subjects), set(args.tasks), valid_mask
    )
    if not used_files:
        raise SystemExit(
            f"No H5 files with unadjusted R² found under {glm_root} "
            f"for subjects={sorted(args.subjects)} tasks={sorted(args.tasks)}"
        )

    r2_full = np.concatenate(pooled["r2_full"])
    r2_noise = np.concatenate(pooled["r2_noise"])
    r2_task = np.concatenate(pooled["r2_task"])

    out_base = plot_unadjusted_r2_distribution(
        r2_full=r2_full,
        r2_noise=r2_noise,
        r2_task=r2_task,
        outdir=outdir,
        out_stem="fig5c_unadjusted_r2_distribution",
        full_width_mm=args.dist_full_width_mm,
        hw_ratio=args.dist_hw_ratio,
        n_bins=args.n_bins,
    )

    print(f"Pooled {len(used_files)} runs across tasks={sorted(args.tasks)} subjects={sorted(args.subjects)}")
    print(f"Valid vertices in mask: {int(np.sum(valid_mask)):,}")
    print(_summarize("r2_full", r2_full))
    print(_summarize("r2_noise", r2_noise))
    print(_summarize("r2_task", r2_task))
    print(f"max |r2_task - (r2_full - r2_noise)| on jointly finite vertices = {max_abs_diff:.3e}")
    print(
        f"Figure size: {args.dist_full_width_mm:.1f} x "
        f"{args.dist_full_width_mm * args.dist_hw_ratio:.1f} mm "
        f"({args.dist_full_width_mm / A4_WIDTH_MM:.0%} of A4 width)"
    )
    print("Saved:")
    for ext in (".svg", ".pdf", ".png"):
        print(" -", out_base.with_suffix(ext))


if __name__ == "__main__":
    main()
