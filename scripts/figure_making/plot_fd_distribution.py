"""Generate 1x5 FD distribution panels for all subjects."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EXTERNAL style dependency: /project/def-pbellec/xuan/figure_style (separate repo; not legacy, not dataset)
workspace_root = Path(__file__).resolve().parents[3]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figure_style.figure_utils import get_figsize, apply_clean_spines
from config import paths  # noqa: E402  TaskScape-WM inputs only; rejects the legacy tree

subjects = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
# Set in main() after paths.configure():
#   data_root -> <dataset>/derivatives/fmriprep  (legacy equivalent: /project/def-pbellec/xuan/cneuromod.multfs.fmriprep)
#   save_root -> <output_root>/figures/fd_distribution  (legacy equivalent: scripts/figure_making/figures/fd_distribution)
data_root: Path | None = None
save_root: Path | None = None

FD_COLOR = "#5B6C8E"

plt.style.use(workspace_root / "figure_style" / "nature_style.mplstyle")


def load_subject_fd(subject):
    """Load per-frame FD values and per-run mean FD values."""
    frame_fd = []
    run_fd = []
    pattern = f"**/{subject}_ses-*_task-*_run-*_desc-confounds_timeseries.tsv"
    confound_files = sorted((data_root / subject).glob(pattern))

    for confound_file in confound_files:
        df = pd.read_csv(confound_file, sep="\t")
        if "framewise_displacement" not in df.columns:
            continue
        fd = df["framewise_displacement"].fillna(0.0).to_numpy(dtype=float)
        frame_fd.extend(fd.tolist())
        run_fd.append(float(np.mean(fd)))

    return np.asarray(frame_fd, dtype=float), np.asarray(run_fd, dtype=float)


def plot_fd_per_frame_panels(fd_by_subject):
    """Plot one row of 5 subject histograms for frame-level FD."""
    figsize, panel_w_mm, panel_h_mm = get_figsize("nature", "double", n_cols=5, n_rows=1)
    width_scale = 1.10
    figsize = (figsize[0] * width_scale, figsize[1])
    panel_w_mm = panel_w_mm * width_scale
    fig, axes = plt.subplots(1, len(subjects), figsize=figsize, sharey=True)

    bins = np.linspace(0.0, 1.4, 50)
    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        values = fd_by_subject[subject]["frame"]
        clipped = values[(values >= 0.0) & (values <= 1.4)]
        ax.hist(clipped, bins=bins, color=FD_COLOR, edgecolor=FD_COLOR, linewidth=0.4, alpha=0.8)
        if clipped.size:
            ax.axvline(np.mean(clipped), color="black", linewidth=0.7, linestyle="--")
        ax.set_xlim(0.0, 1.4)
        ax.set_title(subject.replace("sub-", "sub "), pad=1.5)
        if idx == 0:
            ax.set_ylabel("Frame count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        apply_clean_spines(ax)

    fig.supxlabel("FD per frame (mm)", y=0.03)

    fig.subplots_adjust(
        left=6.0 / panel_w_mm,
        right=1 - (1.5 / panel_w_mm),
        bottom=11.0 / panel_h_mm,
        top=1 - (3.0 / panel_h_mm),
        wspace=6.0 / panel_w_mm,
    )
    out_base = save_root / "fd_per_frame_5col"
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.0)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0, format="pdf")
    plt.close(fig)


def plot_mean_fd_per_run_panels(fd_by_subject):
    """Plot one row of 5 subject histograms for run-mean FD."""
    figsize, panel_w_mm, panel_h_mm = get_figsize("nature", "double", n_cols=5, n_rows=1)
    width_scale = 1.10
    figsize = (figsize[0] * width_scale, figsize[1])
    panel_w_mm = panel_w_mm * width_scale
    fig, axes = plt.subplots(1, len(subjects), figsize=figsize, sharey=True)

    bins = np.linspace(0.0, 0.20, 28)
    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        values = fd_by_subject[subject]["run"]
        clipped = values[(values >= 0.0) & (values <= 0.20)]
        ax.hist(clipped, bins=bins, color=FD_COLOR, edgecolor=FD_COLOR, linewidth=0.4, alpha=0.8)
        if clipped.size:
            ax.axvline(np.mean(clipped), color="black", linewidth=0.7, linestyle="--")
        ax.set_xlim(0.0, 0.20)
        ax.set_title(subject.replace("sub-", "sub "), pad=1.5)
        if idx == 0:
            ax.set_ylabel("Run count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        apply_clean_spines(ax)

    fig.supxlabel("Mean FD per run (mm)", y=0.03)

    fig.subplots_adjust(
        left=6.0 / panel_w_mm,
        right=1 - (1.5 / panel_w_mm),
        bottom=11.0 / panel_h_mm,
        top=1 - (3.0 / panel_h_mm),
        wspace=6.0 / panel_w_mm,
    )
    out_base = save_root / "mean_fd_per_run_5col"
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.0)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0, format="pdf")
    plt.close(fig)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", default=None, help="TaskScape-WM root (default: config.paths / $TASKSCAPE_DATASET_ROOT)")
    p.add_argument("--output_root", default=None, help="analysis output root (default: config.paths / $TASKSCAPE_OUTPUT_ROOT)")
    return p.parse_args()


def main():
    global data_root, save_root
    args = get_args()
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard("plot_fd_distribution")
    # Input: derivatives/fmriprep/<sub>/<ses>/func/*_desc-confounds_timeseries.tsv
    data_root = paths.fmriprep_dir()
    save_root = paths.output_dir("figures", "fd_distribution")

    fd_by_subject = {}
    for subject in subjects:
        frame_fd, run_fd = load_subject_fd(subject)
        fd_by_subject[subject] = {"frame": frame_fd, "run": run_fd}
        print(f"{subject}: {len(run_fd)} runs, {len(frame_fd)} frames")

    plot_fd_per_frame_panels(fd_by_subject)
    plot_mean_fd_per_run_panels(fd_by_subject)
    print(f"Saved figures in: {save_root}")


if __name__ == "__main__":
    main()
