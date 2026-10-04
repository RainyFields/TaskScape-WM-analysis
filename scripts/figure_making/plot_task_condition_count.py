"""Reverse cumulative count of task conditions by repetition threshold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

workspace_root = Path(__file__).resolve().parents[3]
figure_making_dir = Path(__file__).resolve().parent
for path in (workspace_root, figure_making_dir):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from figure_style.figure_utils import apply_clean_spines, mm_to_inches, save_figure
from plot_per_task_within_across_subj_similarity import SUBJECT_COLORS, SUBJECT_ORDER

DATA_DIR = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/"
    "trial_level_betas/grouped_betas/task_relevant_only"
)
OUTPUT_DIR = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/"
    "figures/task_condition_repetition"
)
OUTPUT_BASE = OUTPUT_DIR / "task_condition_reverse_cumulative_count"

A4_WIDTH_MM = 210.0
FIG_WIDTH_MM = A4_WIDTH_MM / 3.0
FIG_HEIGHT_MM = FIG_WIDTH_MM * 0.85
REFERENCE_REPETITIONS = 5

LABEL_FONTSIZE = 10
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 8
ANNOTATION_FONTSIZE = 7


def load_condition_counts(data_dir: Path) -> dict[str, list[int]]:
    """Load per-condition repetition counts for each subject."""
    counts_by_subject: dict[str, list[int]] = {}
    for json_path in sorted(data_dir.glob("*_condition_counts.json")):
        subject = json_path.name.split("_")[0]
        with json_path.open("r", encoding="utf-8") as handle:
            counts_by_subject[subject] = list(json.load(handle).values())
    return counts_by_subject


def reverse_cumulative_counts(
    counts: list[int],
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return the number of conditions with repetitions strictly above each threshold."""
    count_array = np.asarray(counts, dtype=int)
    return np.array([np.sum(count_array > threshold) for threshold in thresholds], dtype=int)


def plot_reverse_cumulative_task_condition_count() -> None:
    """Plot reverse cumulative condition counts versus repetition threshold."""
    counts_by_subject = load_condition_counts(DATA_DIR)
    if not counts_by_subject:
        raise FileNotFoundError(f"No condition count JSON files found in {DATA_DIR}")

    global_max = max(max(counts) for counts in counts_by_subject.values())
    thresholds = np.arange(1, global_max + 1, dtype=int)

    plt.style.use(workspace_root / "figure_style" / "nature_style.mplstyle")
    plt.rcParams.update(
        {
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "lines.linewidth": 1.2,
        }
    )

    fig, ax = plt.subplots(
        figsize=(mm_to_inches(FIG_WIDTH_MM), mm_to_inches(FIG_HEIGHT_MM))
    )

    subjects = [subj for subj in SUBJECT_ORDER if subj in counts_by_subject]

    for subject in subjects:
        cumulative = reverse_cumulative_counts(counts_by_subject[subject], thresholds)
        valid = cumulative >= 1
        ax.plot(
            thresholds[valid],
            cumulative[valid],
            color=SUBJECT_COLORS[subject],
            label=subject.replace("sub-", "sub "),
        )

    ax.set_yscale("log")
    ax.set_xlim(1, global_max)
    ax.set_xlabel("Number of repetitions")
    ax.set_ylabel("Number of conditions")
    ax.axvline(
        REFERENCE_REPETITIONS,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        zorder=1,
    )

    reference_conditions = 450
    ax.text(
        REFERENCE_REPETITIONS,
        -0.12,
        "5 repetitions",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
    )
    ax.text(
        REFERENCE_REPETITIONS + 3,
        reference_conditions,
        f"~{reference_conditions}",
        ha="left",
        va="center",
        fontsize=ANNOTATION_FONTSIZE,
    )

    ax.legend(title="Subject", loc="lower left", frameon=False)
    ax.grid(which="both", linestyle="--", alpha=0.25)
    apply_clean_spines(ax)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.98)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUTPUT_BASE, formats=("svg", "pdf"))
    plt.close(fig)

    print(f"Saved figure to {OUTPUT_BASE}.svg and {OUTPUT_BASE}.pdf")


if __name__ == "__main__":
    plot_reverse_cumulative_task_condition_count()
