import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
from matplotlib.patches import Patch

# --- 0. Configuration & Style Setup ---

workspace_root = Path(__file__).resolve().parents[3]   # EXTERNAL figure_style repo lives next to the analysis repo
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figure_style.figure_utils import get_figsize, save_figure, apply_clean_spines
from config import paths  # TaskScape-WM inputs / OUTPUT_ROOT only; rejects the legacy tree

import argparse
_ap = argparse.ArgumentParser(description="Single-column accuracy panels per task (figure_style).")
_ap.add_argument("--dataset_root", default=None)
_ap.add_argument("--output_root", default=None)
_args = _ap.parse_args()
paths.configure(_args.dataset_root, _args.output_root)
paths.install_legacy_guard("plot_accuracy")

# Define the tasks
tasks = ["1back", "dms", "interdms", "ctxdm"]
subjects = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]

# Keep color identity aligned with the response-rate figure.
task_palette = {
    "1back": "#0F4D92",
    "dms": "#9A4D8E",
    "interdms": "#42949E",
    "ctxdm": "#B64342",
}

style_path = workspace_root / "figure_style" / "nature_style.mplstyle"
plt.style.use(style_path)

# Base directory
# previous-analysis input: {task}_accuracy_results.csv written by scripts/accuracy_level.py to OUTPUT_ROOT/behavior
# (legacy equivalent: results/{task}_accuracy_results.csv)
results_dir = paths.output_dir("behavior")
# legacy equivalent: data/study_designs/
design_dir = paths.study_designs_dir()


def normalize_run(value):
    return f"{int(str(value)):02d}"


def get_subtask_from_block(task, block_name):
    block = str(block_name).lower().strip()
    if task == "1back":
        if "nback_loc" in block:
            return "1-BACK-LOC"
        if "nback_ctg" in block:
            return "1-BACK-CTG"
        if "nback_obj" in block:
            return "1-BACK-OBJ"
    if task == "interdms":
        if "interdms_loc_abab" in block:
            return "InterDMS-LOC-ABAB"
        if "interdms_loc_abba" in block:
            return "InterDMS-LOC-ABBA"
        if "interdms_obj_abab" in block:
            return "InterDMS-OBJ-ABAB"
        if "interdms_obj_abba" in block:
            return "InterDMS-OBJ-ABBA"
        if "interdms_ctg_abab" in block:
            return "InterDMS-CTG-ABAB"
        if "interdms_ctg_abba" in block:
            return "InterDMS-CTG-ABBA"
    if task == "ctxdm":
        if "ctxdm_col" in block:
            return "CTXDM-COL"
        if "ctxdm_lco" in block:
            return "CTXDM-LCO"
    return "unknown"


def build_subtask_lookup():
    lookup = {}
    for design_path in sorted(design_dir.glob("sub-*_design_design_with_converted.tsv")):
        design_df = pd.read_csv(design_path, sep="\t")
        design_df.columns = design_df.columns.str.strip()
        for _, row in design_df.iterrows():
            converted = str(row.get("converted_file_name", "")).strip()
            block_name = str(row.get("block_file_name", "")).strip()
            if not converted or not block_name:
                continue
            parts = converted.split("_")
            if len(parts) < 4:
                continue
            subject = parts[0]
            session = parts[1]
            task = parts[2].replace("task-", "")
            run = parts[3].replace("run-", "")
            subtask = get_subtask_from_block(task, block_name)
            lookup[(subject, session, task, run)] = subtask
    return lookup

# --- 1. Load and Combine Data ---
all_data = []
subtask_lookup = build_subtask_lookup()

for task in tasks:
    file_path = results_dir / f"{task}_accuracy_results.csv"
    
    if file_path.exists():
        df_temp = pd.read_csv(file_path)
        # Add columns for consistent joins and subtype annotation.
        df_temp["task"] = task
        df_temp["session"] = df_temp["session"].astype(str).str.strip()
        df_temp["run"] = df_temp["run"].apply(normalize_run)
        if task in {"1back", "ctxdm", "interdms"}:
            df_temp["subtask"] = df_temp.apply(
                lambda row: subtask_lookup.get(
                    (row["subject"], row["session"], row["task"], row["run"]),
                    "unknown",
                ),
                axis=1,
            )
        else:
            # dms has no subtype annotation in design files.
            df_temp["subtask"] = "DMS"
        all_data.append(df_temp)
    else:
        print(f"Warning: File not found for task '{task}' at {file_path}")

if not all_data:
    raise ValueError("No data loaded. Check paths.")

df_combined = pd.concat(all_data, ignore_index=True)

# --- 2. Plotting ---
# Compute one panel size from a 2-column layout and render one task per figure.
figsize, panel_w_mm, panel_h_mm = get_figsize(
    journal="nature",
    span="single",
    n_cols=1,
    n_rows=1,
)
width_scale = 6.0 / 7.0
height_scale = 0.5
figsize = (figsize[0] * width_scale, figsize[1] * height_scale)
panel_w_mm = panel_w_mm * width_scale
panel_h_mm = panel_h_mm * height_scale

# --- 3. Save Figure ---
save_dir = paths.output_dir("figures", "behavioral_performance")  # legacy equivalent: scripts/figure_making/figures/behavioral_performance/
os.makedirs(save_dir, exist_ok=True)

for task in tasks:
    task_df = df_combined[df_combined["task"] == task]
    if task_df.empty:
        print(f"Warning: No records found for {task}, skipping figure.")
        continue

    subtask_order = []
    if task == "1back":
        subtask_order = ["1-BACK-LOC", "1-BACK-CTG", "1-BACK-OBJ"]
    elif task == "interdms":
        subtask_order = [
            "InterDMS-LOC-ABAB",
            "InterDMS-LOC-ABBA",
            "InterDMS-OBJ-ABAB",
            "InterDMS-OBJ-ABBA",
            "InterDMS-CTG-ABAB",
            "InterDMS-CTG-ABBA",
        ]
    elif task == "ctxdm":
        subtask_order = ["CTXDM-COL", "CTXDM-LCO"]
    else:
        subtask_order = ["DMS"]
    subtask_order = [name for name in subtask_order if (task_df["subtask"] == name).any()]
    if not subtask_order:
        subtask_order = sorted(task_df["subtask"].dropna().unique().tolist())

    subtask_colors_by_task = {
        "1back": {
            "1-BACK-LOC": "#3775BA",
            "1-BACK-CTG": "#0F4D92",
            "1-BACK-OBJ": "#08386B",
        },
        "interdms": {
            "InterDMS-LOC-ABAB": "#7AB6BC",
            "InterDMS-LOC-ABBA": "#67AAB1",
            "InterDMS-OBJ-ABAB": "#58A1A8",
            "InterDMS-OBJ-ABBA": "#42949E",
            "InterDMS-CTG-ABAB": "#378089",
            "InterDMS-CTG-ABBA": "#2A6F77",
        },
        "ctxdm": {"CTXDM-COL": "#D26A69", "CTXDM-LCO": "#B64342"},
        "dms": {"DMS": "#9A4D8E"},
    }
    subtask_colors = subtask_colors_by_task[task]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    box_width = 0.12 if len(subtask_order) >= 5 else (0.18 if len(subtask_order) >= 3 else 0.25)
    intra_gap = 0.05
    inter_group_gap = 0.35
    positions_by_subtask = {subtask: [] for subtask in subtask_order}
    subject_centers = []
    x_cursor = 0.0
    for _ in subjects:
        group_start = x_cursor
        for sub_idx, subtask in enumerate(subtask_order):
            x_pos = x_cursor + sub_idx * (box_width + intra_gap)
            positions_by_subtask[subtask].append(x_pos)
        group_end = x_cursor + (len(subtask_order) - 1) * (box_width + intra_gap)
        subject_centers.append((group_start + group_end) / 2.0)
        x_cursor += len(subtask_order) * (box_width + intra_gap) + inter_group_gap

    for subtask in subtask_order:
        grouped_values = []
        for subject in subjects:
            values = (
                task_df.loc[
                    (task_df["subject"] == subject) & (task_df["subtask"] == subtask),
                    "accuracy",
                ]
                .dropna()
                .to_numpy()
            )
            grouped_values.append(values)

        bp = ax.boxplot(
            grouped_values,
            positions=positions_by_subtask[subtask],
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
            boxprops={"linewidth": 0.6},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#4D4D4D",
                "markeredgewidth": 0.5,
                "markersize": 2.0,
                "linestyle": "none",
                "alpha": 0.9,
            },
        )
        for box in bp["boxes"]:
            box.set_facecolor(subtask_colors[subtask])
            box.set_edgecolor("black")
            box.set_alpha(0.92)

        for pos, values in zip(positions_by_subtask[subtask], grouped_values):
            if values.size == 0:
                continue
            jitter = np.linspace(-0.1, 0.1, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.size, pos) + jitter,
                values,
                s=8,
                color=subtask_colors[subtask],
                alpha=0.55,
                linewidths=0.0,
                zorder=3,
            )

    ax.set_title(f"{task.upper()} accuracy", pad=1.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Subject")
    ax.set_xticks(subject_centers)
    ax.set_xticklabels(subjects)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    apply_clean_spines(ax)

    legend_handles = [
        Patch(facecolor=subtask_colors[subtask], edgecolor="black", label=subtask)
        for subtask in subtask_order
    ]
    ax.legend(
        handles=legend_handles,
        title="Task",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=1 if len(subtask_order) > 3 else len(subtask_order),
        handlelength=1.2,
    )

    left_margin_mm = 9.0
    right_margin_mm = 1.5
    bottom_margin_mm = 9.0
    top_margin_mm = 7.0
    fig.subplots_adjust(
        left=left_margin_mm / panel_w_mm,
        right=1 - (right_margin_mm / panel_w_mm),
        bottom=bottom_margin_mm / panel_h_mm,
        top=1 - (top_margin_mm / panel_h_mm),
    )

    out_base = save_dir / f"{task}_accuracy_singlecol"
    save_figure(fig, out_base, formats=("svg", "pdf"))
    plt.close(fig)
    print(f"Saved {task} figure:\n SVG: {out_base.with_suffix('.svg')}\n PDF: {out_base.with_suffix('.pdf')}")