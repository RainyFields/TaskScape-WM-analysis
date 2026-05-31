import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

workspace_root = Path(__file__).resolve().parents[3]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from figure_style.figure_utils import get_figsize, save_figure, apply_clean_spines

# ==========================================
# PART 1: DEFINITIONS & CONFIGURATION
# ==========================================

root_path = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/behavior")
results_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results")
figure_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/behavioral_performance/")

subjects = ['01', '02', '03', '05', '06']
sessions = [f"{i:03}" for i in range(1, 17)]

# Tasks to process
tasks_to_process = ["1back", "dms", "ctxdm", "interdms"]

# Tasks to plot
tasks_to_plot = ["1back", "interdms", "ctxdm"]

# ==========================================
# PART 2: CALCULATION FUNCTIONS (STRICT RESPONSE RATE)
# ==========================================

def evaluate_responserate_1back(data):
    total_responses = 0
    total_trials = 0
    # Clean whitespace
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in data.iterrows():
        for i in range(1, 6):
            action_col = f"action{i + 1}"
            response_col = f"response_{i}"
            
            if action_col in row and response_col in row:
                response = row[response_col]
                # STRICT CHECK: Only count if response is exactly 'x' or 'b'
                if str(response) in ['x', 'b']:
                    total_responses += 1
                total_trials += 1

    return total_responses / total_trials if total_trials > 0 else float('nan')

def evaluate_responserate_dms(data):
    total_responses = 0
    total_trials = 0
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in data.iterrows():
        if 'action' in row and 'response_1' in row:
            response = row['response_1']
            # STRICT CHECK: Only count if response is exactly 'x' or 'b'
            if str(response) in ['x', 'b']:
                total_responses += 1
            total_trials += 1

    return total_responses / total_trials if total_trials > 0 else float('nan')

def evaluate_responserate_ctxdm(data):
    total_responses = 0
    total_trials = 0
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in data.iterrows():
        if 'action' in row and 'response_2' in row:
            response = row['response_2']
            # STRICT CHECK: Only count if response is exactly 'x' or 'b'
            if str(response) in ['x', 'b']:
                total_responses += 1
            total_trials += 1

    return total_responses / total_trials if total_trials > 0 else float('nan')

def evaluate_responserate_interdms(data):
    total_responses = 0
    total_trials = 0
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    for _, row in data.iterrows():
        for i in range(1, 3):
            action_col = f"action{i}"
            response_col = f"response_{i + 1}"

            if action_col in row and response_col in row:
                response = row[response_col]
                # STRICT CHECK: Only count if response is exactly 'x' or 'b'
                if str(response) in ['x', 'b']:
                    total_responses += 1
                total_trials += 1

    return total_responses / total_trials if total_trials > 0 else float('nan')

# ==========================================
# PART 3: EXECUTE CALCULATION & SAVE CSV
# ==========================================
print("--- Starting Response Rate Calculation (Strict Mode: x or b only) ---")

os.makedirs(results_dir, exist_ok=True)

for task_name in tasks_to_process:
    results = []
    print(f"Processing task: {task_name}")
    
    for sub in subjects:
        for ses in sessions:
            func_dir = root_path / f"sub-{sub}" / f"ses-{ses}" / "func"
            if not func_dir.exists():
                continue 
            
            pattern = f"sub-{sub}_ses-{ses}_task-{task_name}_run-*_events.tsv"
            matching_files = list(func_dir.glob(pattern))
            
            for file_path in matching_files:
                try:
                    data = pd.read_csv(file_path, sep='\t')
                    
                    rate = float('nan')
                    if task_name == "1back":
                        rate = evaluate_responserate_1back(data)
                    elif task_name == "dms":
                        rate = evaluate_responserate_dms(data)
                    elif task_name == "ctxdm":
                        rate = evaluate_responserate_ctxdm(data)
                    elif task_name == "interdms":
                        rate = evaluate_responserate_interdms(data)
                    
                    results.append({
                        "subject": f"sub-{sub}",
                        "session": f"ses-{ses}",
                        "task": task_name,
                        "run": file_path.name.split("_run-")[1].split("_")[0],
                        "response_rate": rate
                    })
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")

    results_df = pd.DataFrame(results)
    output_csv = results_dir / f"{task_name}_responserate_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

# ==========================================
# PART 4: PLOTTING (ZOOMED)
# ==========================================
print("\n--- Starting Plot Generation ---")

task_palette = {
    "1back": "#0F4D92",
    "interdms": "#42949E",
    "ctxdm": "#B64342",
}

style_path = workspace_root / "figure_style" / "nature_style.mplstyle"
plt.style.use(style_path)

all_data = []
for task in tasks_to_plot:
    file_path = results_dir / f"{task}_responserate_results.csv"
    if file_path.exists():
        df_temp = pd.read_csv(file_path)
        df_temp['task'] = task 
        all_data.append(df_temp)
    else:
        print(f"Warning: File not found for plotting: {file_path}")

if all_data:
    df_combined = pd.concat(all_data, ignore_index=True)

    figsize, panel_w_mm, panel_h_mm = get_figsize(
        journal="nature",
        span="single",
        n_cols=1,
        n_rows=1,
    )
    # Keep style typography unchanged while reducing physical canvas size
    # for multi-panel assembly in external tools (e.g., Inkscape).
    width_scale = 2.0 / 3.0
    height_scale = 0.5
    figsize = (figsize[0] * width_scale, figsize[1] * height_scale)
    panel_w_mm = panel_w_mm * width_scale
    panel_h_mm = panel_h_mm * height_scale

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    subject_order = [f"sub-{s}" for s in subjects]
    task_order = tasks_to_plot

    box_width = 0.18
    intra_gap = 0.06
    inter_group_gap = 0.42
    task_positions = {task: [] for task in task_order}
    subject_centers = []

    x_cursor = 0.0
    for subject in subject_order:
        group_start = x_cursor
        for task_idx, task in enumerate(task_order):
            x_pos = x_cursor + task_idx * (box_width + intra_gap)
            task_positions[task].append(x_pos)
        group_end = x_cursor + (len(task_order) - 1) * (box_width + intra_gap)
        subject_centers.append((group_start + group_end) / 2.0)
        x_cursor += len(task_order) * (box_width + intra_gap) + inter_group_gap

    for task in task_order:
        grouped_values = []
        for subject in subject_order:
            values = (
                df_combined.loc[
                    (df_combined["subject"] == subject) & (df_combined["task"] == task),
                    "response_rate",
                ]
                .dropna()
                .to_numpy()
            )
            grouped_values.append(values)

        bp = ax.boxplot(
            grouped_values,
            positions=task_positions[task],
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
                "markersize": 2.2,
                "linestyle": "none",
                "alpha": 0.9,
            },
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(task_palette[task])
            patch.set_edgecolor("black")
            patch.set_alpha(0.92)

    ax.set_title("Behavioral response rate", pad=2.0)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Response rate")
    ax.set_xlabel("Subject")
    ax.set_xticks(subject_centers)
    ax.set_xticklabels(subject_order)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    apply_clean_spines(ax)
    legend_handles = [Patch(facecolor=task_palette[t], edgecolor="black", label=t) for t in task_order]
    ax.legend(
        handles=legend_handles,
        title="Task",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        ncol=3,
        handlelength=1.2,
        columnspacing=1.0,
    )

    left_margin_mm = 7.0
    right_margin_mm = 1.5
    bottom_margin_mm = 9.0
    top_margin_mm = 4.0
    fig.subplots_adjust(
        left=left_margin_mm / panel_w_mm,
        right=1 - (right_margin_mm / panel_w_mm),
        bottom=bottom_margin_mm / panel_h_mm,
        top=1 - (top_margin_mm / panel_h_mm),
    )

    figure_base = figure_dir / "all_tasks_response_rate_singlecol"
    save_figure(fig, figure_base, formats=("svg", "pdf"))

    print(f"Single-column figure saved to:\n SVG: {figure_base.with_suffix('.svg')}\n PDF: {figure_base.with_suffix('.pdf')}")
else:
    print("No data available for plotting.")