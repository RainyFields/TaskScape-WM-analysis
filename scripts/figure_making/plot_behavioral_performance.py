"""
Behavioral Performance Analysis: Accuracy and Response Rate
Generates Nature-style publication figures for all tasks.
Reads from existing CSV results in /project/def-pbellec/xuan/fmri_dataset_project/results/
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import re

warnings.filterwarnings("ignore")

# ============================================================================
# Nature Scientific Data Figure Standards
# ============================================================================
def set_nature_rcparams():
    """Sets global rcParams for Nature-style plotting."""
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'savefig.dpi': 600,
        'savefig.transparent': True,
        'savefig.bbox': 'tight'
    }
    plt.rcParams.update(params)

set_nature_rcparams()

# ============================================================================
# Configuration
# ============================================================================
results_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results")
figure_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/behavioral_performance")
figure_dir.mkdir(parents=True, exist_ok=True)
design_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs")

subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06']
tasks = ['1back', 'dms', 'ctxdm', 'interdms']

# Colorblind-friendly palette
TASK_COLORS = {
    '1back': '#F1C40F',      # yellow family
    'dms': '#0072B2',        # blue (distinct from ctxdm)
    'ctxdm': '#16A085',      # teal family
    'interdms': '#6C3483',   # purple family
}

SUBTASK_COLORS = {
    "nback_loc": "#F1C40F",
    "nback_ctg": "#F7DC6F",
    "nback_obj": "#FCF3CF",
    "ctxdm_col": "#16A085",
    "ctxdm_lco": "#76D7C4",
    "interdms_loc_ABAB": "#6C3483",
    "interdms_loc_ABBA": "#8E44AD",
    "interdms_obj_ABAB": "#A569BD",
    "interdms_obj_ABBA": "#BB8FCE",
    "interdms_ctg_ABAB": "#D2B4DE",
    "interdms_ctg_ABBA": "#E8DAEF",
    "dms_loc": "#009E73",
}

GROUP_SUBTASKS = {
    "1back": ["nback_loc", "nback_ctg", "nback_obj"],
    "ctxdm": ["ctxdm_col", "ctxdm_lco"],
    "interdms": [
        "interdms_loc_ABAB", "interdms_loc_ABBA",
        "interdms_obj_ABAB", "interdms_obj_ABBA",
        "interdms_ctg_ABAB", "interdms_ctg_ABBA",
    ],
    "dms": ["dms_loc"],
}

# ============================================================================
# Load Data
# ============================================================================

print("="*80)
print("LOADING DATA FROM EXISTING CSVs")
print("="*80)

# Load accuracy data
accuracy_dfs = []
for task in tasks:
    csv_path = results_dir / f"{task}_accuracy_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        accuracy_dfs.append(df)
        print(f"  Loaded {task} accuracy: {len(df)} records")
    else:
        print(f"  WARNING: {csv_path} not found")

accuracy_df = pd.concat(accuracy_dfs, ignore_index=True) if accuracy_dfs else pd.DataFrame()

# Load response rate data
responserate_dfs = []
for task in tasks:
    csv_path = results_dir / f"{task}_responserate_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        responserate_dfs.append(df)
        print(f"  Loaded {task} response rate: {len(df)} records")
    else:
        print(f"  WARNING: {csv_path} not found")

responserate_df = pd.concat(responserate_dfs, ignore_index=True) if responserate_dfs else pd.DataFrame()


def build_fine_task_lookup():
    """Build run filename -> fine task mapping from design files."""
    mapping = {}
    for subj in subjects:
        design_file = design_dir / f"{subj}_design_design_with_converted.tsv"
        if not design_file.exists():
            continue
        df = pd.read_csv(design_file, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            conv = str(row["converted_file_name"]).strip()
            block = str(row["block_file_name"]).strip()
            fine_task = re.sub(r"_block_\d+$", "", block)
            mapping[conv] = fine_task
    return mapping


def add_fine_task_column(df, filename_to_fine_task):
    """Attach fine task label to each run row."""
    if df.empty:
        return df

    out = df.copy()

    def to_run_str(run_val):
        try:
            return f"{int(run_val):02d}"
        except Exception:
            run_txt = str(run_val).strip()
            return run_txt.zfill(2)

    out["converted_file_name"] = out.apply(
        lambda r: (
            f"{r['subject']}_{r['session']}_task-{r['task']}_run-{to_run_str(r['run'])}_events.tsv"
        ),
        axis=1,
    )
    out["fine_task"] = out["converted_file_name"].map(filename_to_fine_task)
    # DMS runs are not part of old design table; keep as a dedicated location task.
    out.loc[(out["task"] == "dms") & (out["fine_task"].isna()), "fine_task"] = "dms_loc"
    return out


fine_task_lookup = build_fine_task_lookup()
accuracy_df = add_fine_task_column(accuracy_df, fine_task_lookup)

# ============================================================================
# Summary Statistics
# ============================================================================

def print_summary_table(df, metric_name, metric_col):
    """Print formatted summary table."""
    print(f"\n{'='*100}")
    print(f"SUMMARY TABLE: {metric_name.upper()} (Mean ± Std)")
    print(f"{'='*100}")
    
    # Header
    print(f"\n{'Task':<12}", end="")
    for sub in subjects:
        print(f"{sub:<18}", end="")
    print(f"{'GROUP':<18}")
    print("-"*110)
    
    # Data rows
    for task in tasks:
        print(f"{task:<12}", end="")
        task_all = []
        for sub in subjects:
            subset = df[(df['task'] == task) & (df['subject'] == sub)]
            if len(subset) > 0:
                vals = subset[metric_col].dropna().values
                if len(vals) > 0:
                    mean_val = np.mean(vals) * 100
                    std_val = np.std(vals) * 100
                    print(f"{mean_val:.1f}±{std_val:.1f}%{'':<6}", end="")
                    task_all.extend(vals)
                else:
                    print(f"{'N/A':<18}", end="")
            else:
                print(f"{'N/A':<18}", end="")
        if task_all:
            print(f"{np.mean(task_all)*100:.1f}±{np.std(task_all)*100:.1f}%")
        else:
            print("N/A")

if not accuracy_df.empty:
    print_summary_table(accuracy_df, "Accuracy", "accuracy")
    
if not responserate_df.empty:
    print_summary_table(responserate_df, "Response Rate", "response_rate")

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_grouped_boxplot(df, metric_col, metric_label, output_name):
    """Plot boxplot grouped by subject with different tasks as hue."""
    
    # Nature standard: double column width
    width_mm = 183
    height_mm = 80
    fig, ax = plt.subplots(figsize=(width_mm/25.4, height_mm/25.4))
    
    n_tasks = len(tasks)
    n_subjects = len(subjects)
    
    box_width = 0.18
    group_gap = 0.15
    
    # X positions for each subject group
    subject_positions = np.arange(n_subjects)
    
    all_bplots = []
    
    for i, task in enumerate(tasks):
        positions = subject_positions + (i - (n_tasks - 1) / 2) * (box_width + 0.02)
        
        data_to_plot = []
        valid_positions = []
        
        for j, subj in enumerate(subjects):
            subset = df[(df['task'] == task) & (df['subject'] == subj)]
            if len(subset) > 0:
                vals = subset[metric_col].dropna().values * 100
                if len(vals) > 0:
                    data_to_plot.append(vals)
                    valid_positions.append(positions[j])
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, positions=valid_positions, widths=box_width,
                           patch_artist=True, showfliers=True, 
                           manage_ticks=False)
            
            for patch in bp['boxes']:
                patch.set_facecolor(TASK_COLORS[task])
                patch.set_alpha(0.7)
                patch.set_linewidth(0.5)
            
            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_linewidth(0.5)
                    item.set_color('black')
            
            for flier in bp['fliers']:
                flier.set(marker='o', markersize=1.5, markerfacecolor='gray',
                         markeredgecolor='none', alpha=0.5)
            
            all_bplots.append((bp, task))
    
    # X-axis
    ax.set_xticks(subject_positions)
    ax.set_xticklabels(subjects, fontsize=6)
    ax.set_xlabel('Subject', fontsize=7)
    ax.set_ylabel(f'{metric_label} (%)', fontsize=7)
    ax.set_title(f'Behavioral {metric_label} by Task', fontsize=8)
    
    # Y-axis limits
    ax.set_ylim(0, 105)
    
    # Legend
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=TASK_COLORS[t], alpha=0.7,
                                    edgecolor='black', linewidth=0.5) for t in tasks]
    ax.legend(legend_handles, tasks, loc='lower left', fontsize=5, frameon=True,
             edgecolor='black', framealpha=0.9, ncol=4)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    fig.savefig(figure_dir / f'{output_name}.png', dpi=600, facecolor='white', bbox_inches='tight')
    fig.savefig(figure_dir / f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_name}")


def plot_single_subject_boxplot(df, subject, metric_col, metric_label, output_name):
    """Plot boxplot for a single subject across all tasks."""
    
    # Nature standard: single column width
    width_mm = 89
    height_mm = 60
    fig, ax = plt.subplots(figsize=(width_mm/25.4, height_mm/25.4))
    
    data_to_plot = []
    colors = []
    valid_tasks = []
    
    for task in tasks:
        subset = df[(df['subject'] == subject) & (df['task'] == task)]
        if len(subset) > 0:
            vals = subset[metric_col].dropna().values * 100
            if len(vals) > 0:
                data_to_plot.append(vals)
                colors.append(TASK_COLORS[task])
                valid_tasks.append(task)
    
    if not data_to_plot:
        plt.close(fig)
        return
    
    bp = ax.boxplot(data_to_plot, positions=range(len(valid_tasks)), widths=0.6,
                   patch_artist=True, showfliers=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(0.5)
    
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(0.5)
            item.set_color('black')
    
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=1.5, markerfacecolor='gray',
                 markeredgecolor='none', alpha=0.5)
    
    ax.set_xticks(range(len(valid_tasks)))
    ax.set_xticklabels(valid_tasks, fontsize=6, rotation=30, ha='right')
    ax.set_xlabel('Task', fontsize=7)
    ax.set_ylabel(f'{metric_label} (%)', fontsize=7)
    ax.set_title(f'{subject}: {metric_label}', fontsize=8)
    
    ax.set_ylim(0, 105)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    fig.savefig(figure_dir / f'{output_name}.png', dpi=600, facecolor='white', bbox_inches='tight')
    fig.savefig(figure_dir / f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_task_summary(df, metric_col, metric_label, output_name):
    """Plot summary boxplot by task (all subjects pooled)."""
    
    # Nature standard: single column width
    width_mm = 89
    height_mm = 70
    fig, ax = plt.subplots(figsize=(width_mm/25.4, height_mm/25.4))
    
    data_to_plot = []
    colors = []
    valid_tasks = []
    
    for task in tasks:
        subset = df[df['task'] == task]
        if len(subset) > 0:
            vals = subset[metric_col].dropna().values * 100
            if len(vals) > 0:
                data_to_plot.append(vals)
                colors.append(TASK_COLORS[task])
                valid_tasks.append(task)
    
    if not data_to_plot:
        plt.close(fig)
        return
    
    bp = ax.boxplot(data_to_plot, positions=range(len(valid_tasks)), widths=0.6,
                   patch_artist=True, showfliers=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(0.5)
    
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(0.5)
            item.set_color('black')
    
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=1.5, markerfacecolor='gray',
                 markeredgecolor='none', alpha=0.5)
    
    ax.set_xticks(range(len(valid_tasks)))
    ax.set_xticklabels(valid_tasks, fontsize=6)
    ax.set_xlabel('Task', fontsize=7)
    ax.set_ylabel(f'{metric_label} (%)', fontsize=7)
    ax.set_title(f'Behavioral {metric_label}', fontsize=8)
    
    ax.set_ylim(0, 105)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    fig.savefig(figure_dir / f'{output_name}.png', dpi=600, facecolor='white', bbox_inches='tight')
    fig.savefig(figure_dir / f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_name}")


def plot_single_task_accuracy(df, task, output_name):
    """Plot one accuracy figure per task group across subjects."""
    width_mm = 89
    height_mm = 60
    fig, ax = plt.subplots(figsize=(width_mm/25.4, height_mm/25.4))

    data_to_plot = []
    valid_subjects = []
    for subj in subjects:
        subset = df[(df['task'] == task) & (df['subject'] == subj)]
        vals = subset['accuracy'].dropna().values * 100
        if len(vals) > 0:
            data_to_plot.append(vals)
            valid_subjects.append(subj)

    if not data_to_plot:
        plt.close(fig)
        return

    bp = ax.boxplot(data_to_plot, positions=range(len(valid_subjects)), widths=0.6,
                    patch_artist=True, showfliers=True)

    for patch in bp['boxes']:
        patch.set_facecolor(TASK_COLORS.get(task, '#999999'))
        patch.set_alpha(0.75)
        patch.set_linewidth(0.5)

    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(0.5)
            item.set_color('black')

    for flier in bp['fliers']:
        flier.set(marker='o', markersize=1.5, markerfacecolor='gray',
                  markeredgecolor='none', alpha=0.5)

    ax.set_xticks(range(len(valid_subjects)))
    ax.set_xticklabels(valid_subjects, fontsize=6)
    ax.set_xlabel('Subject', fontsize=7)
    ax.set_ylabel('Accuracy (%)', fontsize=7)
    ax.set_title(f'Accuracy: {task}', fontsize=8)
    ax.set_ylim(0, 105)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(figure_dir / f'{output_name}.png', dpi=600, facecolor='white', bbox_inches='tight')
    fig.savefig(figure_dir / f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_name}")


def plot_accuracy_within_group(df, group_name, output_name):
    """For one group, plot per-subtask accuracy with subject-grouped boxes."""
    subtasks = GROUP_SUBTASKS[group_name]
    group_df = df[df["fine_task"].isin(subtasks)].copy()
    if group_df.empty:
        return

    width_mm = 183
    height_mm = 80
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))

    n_subtasks = len(subtasks)
    subject_positions = np.arange(len(subjects))
    box_width = 0.14 if n_subtasks > 3 else 0.2

    for i, subtask in enumerate(subtasks):
        positions = subject_positions + (i - (n_subtasks - 1) / 2) * (box_width + 0.015)
        data_to_plot = []
        valid_positions = []
        for j, subj in enumerate(subjects):
            vals = group_df[
                (group_df["subject"] == subj) & (group_df["fine_task"] == subtask)
            ]["accuracy"].dropna().values * 100
            if len(vals) > 0:
                data_to_plot.append(vals)
                valid_positions.append(positions[j])

        if not data_to_plot:
            continue

        bp = ax.boxplot(
            data_to_plot,
            positions=valid_positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(SUBTASK_COLORS.get(subtask, "#999999"))
            patch.set_alpha(0.8)
            patch.set_linewidth(0.5)
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_linewidth(0.5)
                item.set_color("black")
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=1.5, markerfacecolor="gray",
                      markeredgecolor="none", alpha=0.5)

    ax.set_xticks(subject_positions)
    ax.set_xticklabels(subjects, fontsize=6)
    ax.set_xlabel("Subject", fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=7)
    ax.set_title(f"Accuracy within {group_name} task group", fontsize=8)
    ax.set_ylim(0, 105)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SUBTASK_COLORS.get(t, "#999999"),
                      edgecolor="black", linewidth=0.5, alpha=0.8)
        for t in subtasks
    ]
    ax.legend(
        legend_handles,
        subtasks,
        loc="lower left",
        fontsize=5,
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
        ncol=min(3, len(subtasks)),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    plt.tight_layout()

    fig.savefig(figure_dir / f"{output_name}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(figure_dir / f"{output_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_name}")


# ============================================================================
# Generate Figures
# ============================================================================

print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

# Accuracy figures
if not accuracy_df.empty:
    # Grouped by subject
    plot_grouped_boxplot(accuracy_df, 'accuracy', 'Accuracy', 'all_tasks_accuracy_grouped')
    
    # Summary by task
    plot_task_summary(accuracy_df, 'accuracy', 'Accuracy', 'all_tasks_accuracy_summary')
    
    # Per subject
    for subject in subjects:
        plot_single_subject_boxplot(accuracy_df, subject, 'accuracy', 'Accuracy',
                                    f'{subject}_accuracy_by_task')

    # One figure per task group (requested)
    for task in tasks:
        plot_single_task_accuracy(accuracy_df, task, f'accuracy_{task}_by_subject')

    # Per-group, per-subtask accuracy figures
    for group_name in ["1back", "ctxdm", "interdms", "dms"]:
        plot_accuracy_within_group(
            accuracy_df,
            group_name,
            f"accuracy_within_group_{group_name}",
        )

# Response rate figures
if not responserate_df.empty:
    # Grouped by subject
    plot_grouped_boxplot(responserate_df, 'response_rate', 'Response Rate', 
                         'all_tasks_response_rate_grouped')
    
    # Summary by task
    plot_task_summary(responserate_df, 'response_rate', 'Response Rate',
                      'all_tasks_response_rate_summary')
    
    # Per subject
    for subject in subjects:
        plot_single_subject_boxplot(responserate_df, subject, 'response_rate', 'Response Rate',
                                    f'{subject}_response_rate_by_task')

print(f"\nAll figures saved to: {figure_dir}")
