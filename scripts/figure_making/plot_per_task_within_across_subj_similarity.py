#!/usr/bin/env python3
"""
Multi-Subject Similarity Analysis (Within vs. Across) with Statistical Testing.
 
Usage Example:
unit choose from: voxel, region and network
aggregation choose from: condition, task, task_group


python plot_per_task_within_across_subj_similarity.py \
  --input_dir .../grouped_betas/task_relevant_only \
  --output_dir .../results/similarity_analysis/with_dms \
  --atlas_path .../data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path .../data/cortex_parcel_network_assignments.txt \
  --aggregation task \
  --multipanel_a4 \
  --n_bootstrap 50
"""

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import nibabel as nib

# Add scripts directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_valid_vertex_mask, apply_mask_to_betas_dict

FIG_STYLE_ROOT = Path("/project/def-pbellec/xuan/figure_style")
sys.path.insert(0, str(FIG_STYLE_ROOT))
from figure_utils import mm_to_inches, save_figure, apply_clean_spines, add_panel_labels

STYLE_PATH = FIG_STYLE_ROOT / "nature_style.mplstyle"

# A4 page width; height slightly below the default aspect ratio.
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
FIG_HEIGHT_MM = 72.0

UNIT_PANEL_TITLES = {
    "voxel": "Vertices",
    "region": "Regions",
    "network": "Networks",
}
MULTIPANEL_UNITS = ("voxel", "region", "network")

# nature_style.mplstyle targets ~89 mm journal columns (6–7 pt). For a full A4-width
# standalone figure, scale typography up so labels stay readable on screen and in print.
A4_FONT_RC = {
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}

# 3x1 multipanel: wide shallow panels + legend column on the right.
MULTIPANEL_DATA_COL_MM = 168.0
MULTIPANEL_LEGEND_COL_MM = 42.0
MULTIPANEL_PANEL_HW_RATIO = 0.30  # panel height / panel width (low = shallow)
MULTIPANEL_ROW_GAP_MM = 1.0
MULTIPANEL_MARGIN_V_MM = 6.0

# Slightly above journal-column nature_style (6–7 pt) for A4-width standalone panels.
MULTIPANEL_FONT_RC = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


def apply_nature_style():
    """Nature Scientific Data typography (6 pt ticks, 7 pt labels, Arial)."""
    plt.style.use(STYLE_PATH)


def apply_multipanel_style():
    plt.style.use(STYLE_PATH)
    plt.rcParams.update(MULTIPANEL_FONT_RC)


def apply_a4_fullwidth_style():
    plt.style.use(STYLE_PATH)
    plt.rcParams.update(A4_FONT_RC)

# Task order: dms_loc → n-back → inter-dms → ctx-dm
TASK_ORDER = [
    "dms_loc",
    "nback_loc", "nback_ctg", "nback_obj",
    "interdms_loc_ABAB", "interdms_loc_ABBA",
    "interdms_obj_ABAB", "interdms_obj_ABBA",
    "interdms_ctg_ABAB", "interdms_ctg_ABBA",
    "ctxdm_col", "ctxdm_lco",
]
TASK_GROUP_ORDER = ["dms", "nback", "interdms", "ctxdm"]

TASK_TO_CATEGORY = {
    "dms_loc": "dms",
    "nback_loc": "nback", "nback_ctg": "nback", "nback_obj": "nback",
    "interdms_loc_ABAB": "interdms", "interdms_loc_ABBA": "interdms",
    "interdms_obj_ABAB": "interdms", "interdms_obj_ABBA": "interdms",
    "interdms_ctg_ABAB": "interdms", "interdms_ctg_ABBA": "interdms",
    "ctxdm_col": "ctxdm", "ctxdm_lco": "ctxdm",
}

# One color per task category (figures4papers palette)
TASK_CATEGORY_COLORS = {
    "dms": "#3775BA",
    "nback": "#FFD700",
    "interdms": "#9A4D8E",
    "ctxdm": "#42949E",
}
TASK_CATEGORY_LABELS = {
    "dms": "DMS",
    "nback": "n-back",
    "interdms": "Inter-DMS",
    "ctxdm": "ctxDM",
}

SUBJECT_ORDER = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
SUBJECT_COLORS = {
    "sub-01": "#6A98CB",
    "sub-02": "#0C2458",
    "sub-03": "#B64342",
    "sub-05": "#42949E",
    "sub-06": "#9A4D8E",
}
SUBJECT_MARKERS = {
    "sub-01": "o",
    "sub-02": "s",
    "sub-03": "^",
    "sub-05": "D",
    "sub-06": "v",
}
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]


def get_subject_marker(sub_id: str) -> str:
    if sub_id in SUBJECT_MARKERS:
        return SUBJECT_MARKERS[sub_id]
    idx = hash(sub_id) % len(MARKER_CYCLE)
    return MARKER_CYCLE[idx]

# ==========================================
# 1. HELPERS
# ==========================================

def load_glasser_labels(atlas_path: str):
    lab_img = nib.load(atlas_path)
    lab = np.asarray(lab_img.get_fdata()).squeeze()
    lab = lab.reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids

def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]
    def avg_by_region(A):
        out = np.full((A.shape[0], len(region_ids)), np.nan, dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx.size == 0: continue
            out[:, j] = np.nanmean(A[:, idx], axis=1)
        return out
    return avg_by_region

def load_region_to_network_map(assign_path: str, max_region_id: int):
    txt = Path(assign_path).read_text().strip().splitlines()
    arr = np.zeros(max_region_id + 1, dtype=int)
    for i, line in enumerate(txt, start=1):
        if i <= max_region_id:
            arr[i] = int(line.strip())
    return arr

def make_network_averager(region_ids: np.ndarray, region_to_net: np.ndarray, n_networks: int = 12):
    nets = [[] for _ in range(n_networks)]
    for j, rid in enumerate(region_ids):
        nid = region_to_net[rid]
        if nid >= 1 and nid <= n_networks:
            nets[nid - 1].append(j)
    def avg_by_network(R):
        out = np.full((R.shape[0], n_networks), np.nan, dtype=np.float32)
        for k, cols in enumerate(nets):
            if len(cols) == 0: continue
            out[:, k] = np.nanmean(R[:, cols], axis=1)
        return out
    return avg_by_network

def group_betas(raw_betas, aggregation_mode):
    """
    Groups raw dictionary keys.
    Note: For the box plot to be readable, we usually use 'task' or 'task_group'.
    """
    if aggregation_mode == 'task':
        targets = TASK_ORDER
    elif aggregation_mode == 'task_group':
        targets = TASK_GROUP_ORDER
    else: 
        # condition
        targets = list(raw_betas.keys())

    grouped = {k: [] for k in targets}
    
    if aggregation_mode == 'condition':
        grouped = raw_betas
    else:
        sorted_targets = sorted(targets, key=len, reverse=True)
        for raw_key, beta_list in raw_betas.items():
            for target in sorted_targets:
                if target in raw_key:
                    grouped[target].extend(beta_list)
                    break
    return grouped, targets

def order_tasks(valid_tasks, aggregation_mode):
    """Return tasks in the requested display order."""
    if aggregation_mode == "task":
        preferred = TASK_ORDER
    elif aggregation_mode == "task_group":
        preferred = TASK_GROUP_ORDER
    else:
        preferred = sorted(valid_tasks)
    ordered = [t for t in preferred if t in valid_tasks]
    extras = sorted(set(valid_tasks) - set(ordered))
    return ordered + extras


def get_task_color(task_name: str, aggregation_mode: str):
    if aggregation_mode == "task_group":
        return TASK_CATEGORY_COLORS.get(task_name, "#CFCECE")
    category = TASK_TO_CATEGORY.get(task_name)
    return TASK_CATEGORY_COLORS.get(category, "#CFCECE")


def annot_stat(star, x1, x2, y, h, ax):
    """Helper to draw significance bracket and stars."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c="k")
    ax.text(
        (x1 + x2) * 0.5, y + h, star,
        ha="center", va="bottom", color="k",
        fontsize=plt.rcParams["font.size"],
    )


def apply_unit_transform(mat, unit, avg_by_region, avg_by_network):
    if unit == "voxel":
        return mat
    mat = avg_by_region(mat)
    if unit == "network":
        return avg_by_network(mat)
    return mat


def setup_spatial_averagers(unit, atlas_path, network_assignments_path, valid_mask):
    avg_by_region = None
    avg_by_network = None
    if unit in ("region", "network"):
        labels_vec, region_ids = load_glasser_labels(atlas_path)
        if valid_mask is not None:
            labels_vec = labels_vec[valid_mask]
        avg_by_region = make_region_averager(labels_vec, region_ids)
        if unit == "network":
            if not network_assignments_path:
                raise SystemExit("Need --network_assignments_path for network unit.")
            max_rid = int(region_ids.max())
            r2n = load_region_to_network_map(network_assignments_path, max_rid)
            avg_by_network = make_network_averager(region_ids, r2n)
    return avg_by_region, avg_by_network


def load_subject_task_matrix(
    pickle_path,
    task,
    aggregation,
    min_reps,
    valid_mask,
    *,
    max_reps=None,
    rng=None,
):
    with open(pickle_path, "rb") as f:
        raw = pickle.load(f)
    if valid_mask is not None:
        raw = apply_mask_to_betas_dict(raw, valid_mask)
    grouped, _ = group_betas(raw, aggregation)
    if task not in grouped or len(grouped[task]) < min_reps:
        return None
    mat = np.stack([np.asarray(x).reshape(-1) for x in grouped[task]], axis=0)
    if max_reps is not None and mat.shape[0] > max_reps:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(mat.shape[0], size=max_reps, replace=False)
        mat = mat[idx]
    return mat


def scan_subject_task_index(input_dir, aggregation, min_reps, _valid_mask=None):
    # Mask is applied in load_subject_task_matrix (scan only counts repetitions).
    files = sorted(Path(input_dir).glob("*task_condition_betas.pkl"))
    if not files:
        raise SystemExit(f"No pickle files found in {input_dir}. Check path.")

    if aggregation == "task":
        targets = TASK_ORDER
    elif aggregation == "task_group":
        targets = TASK_GROUP_ORDER
    else:
        targets = None

    subject_files = {}
    valid_tasks = set()
    for fp in files:
        sub_id = fp.name.split("_")[0]
        with open(fp, "rb") as f:
            raw = pickle.load(f)
        if aggregation == "condition":
            tasks_ok = {k for k, v in raw.items() if len(v) >= min_reps}
        else:
            counts = {k: 0 for k in targets}
            sorted_targets = sorted(targets, key=len, reverse=True)
            for raw_key, beta_list in raw.items():
                for target in sorted_targets:
                    if target in raw_key:
                        counts[target] += len(beta_list)
                        break
            tasks_ok = {t for t, n in counts.items() if n >= min_reps}
        if tasks_ok:
            subject_files[sub_id] = fp
            valid_tasks |= tasks_ok
        del raw

    sorted_tasks = order_tasks(valid_tasks, aggregation)
    return subject_files, sorted_tasks


def load_task_mats_vertex(
    subject_files,
    task,
    aggregation,
    min_reps,
    valid_mask,
    *,
    max_reps=None,
    rng=None,
):
    mats = {}
    for sub_id, fp in subject_files.items():
        mat = load_subject_task_matrix(
            fp, task, aggregation, min_reps, valid_mask,
            max_reps=max_reps, rng=rng,
        )
        if mat is not None:
            mats[sub_id] = mat
    return mats


def run_bootstrap_for_unit(
    subject_files,
    sorted_tasks,
    unit,
    avg_by_region,
    avg_by_network,
    n_bootstrap,
    rng,
    aggregation,
    min_reps,
    valid_mask,
    *,
    max_reps_per_task=None,
):
    plot_data = []
    stats_data = {}
    subject_stats = {}

    for t in sorted_tasks:
        vertex_mats = load_task_mats_vertex(
            subject_files, t, aggregation, min_reps, valid_mask,
            max_reps=max_reps_per_task if unit == "voxel" else None,
            rng=rng,
        )
        subs_with_task = sorted(vertex_mats.keys())
        if len(subs_with_task) < 2:
            del vertex_mats
            continue

        unit_mats = {
            s: apply_unit_transform(vertex_mats[s], unit, avg_by_region, avg_by_network)
            for s in subs_with_task
        }
        del vertex_mats

        task_within = []
        task_across = []
        subject_stats[t] = {s: ([], []) for s in subs_with_task}

        for _ in range(n_bootstrap):
            h1_list, h2_list = [], []
            for s in subs_with_task:
                mat = unit_mats[s]
                n_reps = mat.shape[0]
                perm = rng.permutation(n_reps)
                mid = n_reps // 2
                h1_list.append(mat[perm[:mid]].mean(axis=0))
                h2_list.append(mat[perm[mid:]].mean(axis=0))

            m1 = np.stack(h1_list, axis=0)
            m2 = np.stack(h2_list, axis=0)
            m1_z = (m1 - m1.mean(axis=1, keepdims=True)) / (m1.std(axis=1, keepdims=True) + 1e-8)
            m2_z = (m2 - m2.mean(axis=1, keepdims=True)) / (m2.std(axis=1, keepdims=True) + 1e-8)
            r = np.dot(m1_z, m2_z.T) / m1.shape[1]

            within_mean = np.mean(np.diag(r))
            off_mask = ~np.eye(r.shape[0], dtype=bool)
            across_mean = np.mean(r[off_mask])
            task_within.append(within_mean)
            task_across.append(across_mean)

            for si, s in enumerate(subs_with_task):
                subject_stats[t][s][0].append(r[si, si])
                subject_stats[t][s][1].append(np.mean(np.delete(r[si, :], si)))

            plot_data.append({"Task": t, "Type": "Within", "Similarity": within_mean})
            plot_data.append({"Task": t, "Type": "Across", "Similarity": across_mean})

        stats_data[t] = (np.array(task_within), np.array(task_across))
        for s in subs_with_task:
            subject_stats[t][s] = (
                np.array(subject_stats[t][s][0]),
                np.array(subject_stats[t][s][1]),
            )

        del unit_mats

    return pd.DataFrame(plot_data), stats_data, subject_stats


def subjects_in_plot_from_stats(subject_stats, sorted_tasks):
    subjects = [
        s for s in SUBJECT_ORDER if any(s in subject_stats.get(t, {}) for t in sorted_tasks)
    ]
    if subjects:
        return subjects
    return sorted(
        {s for t in subject_stats for s in subject_stats[t]},
        key=lambda x: (SUBJECT_ORDER.index(x) if x in SUBJECT_ORDER else 99, x),
    )


def draw_similarity_panel(
    ax,
    df,
    sorted_tasks,
    subject_stats,
    stats_data,
    aggregation,
    *,
    title,
    show_xlabel=True,
    show_ylabel=True,
    show_xticklabels=True,
    annotate_stats=True,
    scatter_size=14,
):
    x_positions = np.arange(len(sorted_tasks))
    offset = 0.18
    width = 0.30

    for i, task in enumerate(sorted_tasks):
        w = df[(df["Task"] == task) & (df["Type"] == "Within")]["Similarity"].values
        a = df[(df["Task"] == task) & (df["Type"] == "Across")]["Similarity"].values
        c = get_task_color(task, aggregation)

        if len(w) > 0:
            bp_w = ax.boxplot(
                [w], positions=[i - offset], widths=width, patch_artist=True,
                showfliers=False, manage_ticks=False,
            )
            bp_w["boxes"][0].set_facecolor(c)
            bp_w["boxes"][0].set_alpha(0.85)
            bp_w["boxes"][0].set_linewidth(0.6)
            for e in ["whiskers", "caps", "medians"]:
                for line in bp_w[e]:
                    line.set_color("black")
                    line.set_linewidth(0.6)

        if len(a) > 0:
            bp_a = ax.boxplot(
                [a], positions=[i + offset], widths=width, patch_artist=True,
                showfliers=False, manage_ticks=False,
            )
            bp_a["boxes"][0].set_facecolor(c)
            bp_a["boxes"][0].set_alpha(0.35)
            bp_a["boxes"][0].set_linewidth(0.6)
            for e in ["whiskers", "caps", "medians"]:
                for line in bp_a[e]:
                    line.set_color("black")
                    line.set_linewidth(0.6)

    jitter_scale = width * 0.12
    for i, task in enumerate(sorted_tasks):
        if task not in subject_stats:
            continue
        n_subs = len(subject_stats[task])
        jitter = (
            np.linspace(-jitter_scale, jitter_scale, n_subs)
            if n_subs > 1 else np.array([0.0])
        )
        for j, sub_id in enumerate(
            sorted(
                subject_stats[task].keys(),
                key=lambda x: (SUBJECT_ORDER.index(x) if x in SUBJECT_ORDER else 99, x),
            )
        ):
            w_arr, a_arr = subject_stats[task][sub_id]
            marker = get_subject_marker(sub_id)
            ax.scatter(
                i - offset + jitter[j], np.mean(w_arr),
                marker=marker, s=scatter_size, facecolors="white", edgecolors="black",
                linewidths=0.5, zorder=5,
            )
            ax.scatter(
                i + offset + jitter[j], np.mean(a_arr),
                marker=marker, s=scatter_size, facecolors="white", edgecolors="black",
                linewidths=0.5, zorder=5,
            )

    if annotate_stats and len(df) > 0:
        y_max = df["Similarity"].max()
        y_range = y_max - df["Similarity"].min()
        h = y_range * 0.05 if y_range > 0 else 0.02
        for i, t in enumerate(sorted_tasks):
            if t not in stats_data:
                continue
            w_arr, a_arr = stats_data[t]
            _, pval = ttest_rel(w_arr, a_arr)
            if pval < 0.001:
                star = "***"
            elif pval < 0.01:
                star = "**"
            elif pval < 0.05:
                star = "*"
            else:
                star = "ns"
            curr_y = df[df["Task"] == t]["Similarity"].max() + (h * 0.5)
            annot_stat(star, i - offset, i + offset, curr_y, h * 0.4, ax)

    ax.set_title(title, pad=3)
    ax.set_xlabel("Task" if show_xlabel else "")
    ax.set_ylabel("Similarity (Pearson r)" if show_ylabel else "")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_tasks, rotation=35, ha="right")
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    apply_clean_spines(ax)
    return ax


def build_legend_handles(aggregation, sorted_tasks, subjects_in_plot):
    if aggregation == "task_group":
        legend_categories = [c for c in TASK_GROUP_ORDER if c in sorted_tasks]
    else:
        legend_categories = [
            c for c in TASK_GROUP_ORDER
            if any(TASK_TO_CATEGORY.get(t) == c for t in sorted_tasks)
        ]
    category_patches = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=TASK_CATEGORY_COLORS[c],
            edgecolor="black",
            linewidth=0.5,
        )
        for c in legend_categories
    ]
    category_labels = [TASK_CATEGORY_LABELS[c] for c in legend_categories]
    within_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#4D4D4D", alpha=0.85,
        edgecolor="black", linewidth=0.5,
    )
    across_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#4D4D4D", alpha=0.35,
        edgecolor="black", linewidth=0.5,
    )
    subject_handles = [
        plt.Line2D(
            [0], [0], marker=get_subject_marker(s), color="w",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.5, markersize=4, linestyle="None",
            label=s.replace("sub-", "Sub "),
        )
        for s in subjects_in_plot
    ]
    return (
        category_patches,
        category_labels,
        [within_patch, across_patch],
        ["Within-subject", "Across-subject"],
        subject_handles,
    )


def draw_legend_panel(ax, aggregation, sorted_tasks, subjects_in_plot):
    ax.axis("off")
    cat_patches, cat_labels, cmp_patches, cmp_labels, sub_handles = build_legend_handles(
        aggregation, sorted_tasks, subjects_in_plot
    )
    leg_task = ax.legend(
        cat_patches, cat_labels,
        title="Task", loc="upper left", bbox_to_anchor=(0.0, 1.0),
        frameon=False, ncol=1, borderaxespad=0.0,
    )
    ax.add_artist(leg_task)
    leg_cmp = ax.legend(
        cmp_patches, cmp_labels,
        title="Comparison", loc="upper left", bbox_to_anchor=(0.0, 0.58),
        frameon=False, ncol=1, borderaxespad=0.0,
    )
    ax.add_artist(leg_cmp)
    ax.legend(
        sub_handles,
        [h.get_label() for h in sub_handles],
        title="Subject", loc="upper left", bbox_to_anchor=(0.0, 0.22),
        frameon=False, ncol=1, handletextpad=0.4, borderaxespad=0.0,
    )


def add_side_legends(ax, aggregation, sorted_tasks, subjects_in_plot):
    cat_patches, cat_labels, cmp_patches, cmp_labels, sub_handles = build_legend_handles(
        aggregation, sorted_tasks, subjects_in_plot
    )
    leg_task = ax.legend(
        cat_patches, cat_labels,
        title="Task", loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, borderaxespad=0.0,
    )
    ax.add_artist(leg_task)
    leg_cmp = ax.legend(
        cmp_patches, cmp_labels,
        title="Comparison", loc="upper left", bbox_to_anchor=(1.01, 0.45),
        frameon=False, borderaxespad=0.0,
    )
    ax.add_artist(leg_cmp)
    ax.legend(
        sub_handles,
        [h.get_label() for h in sub_handles],
        title="Subject", loc="upper left", bbox_to_anchor=(1.01, 0.08),
        frameon=False, borderaxespad=0.0, handletextpad=0.4,
    )


def multipanel_shared_ylim(results_by_unit, padding_frac=0.14):
    vals = []
    for unit in MULTIPANEL_UNITS:
        df = results_by_unit[unit][0]
        if len(df) > 0:
            vals.extend(df["Similarity"].tolist())
    if not vals:
        return 0.0, 1.0
    ymin, ymax = float(np.min(vals)), float(np.max(vals))
    pad = max((ymax - ymin) * padding_frac, 0.03)
    return ymin - pad * 0.25, ymax + pad


def plot_multipanel_a4(results_by_unit, sorted_tasks, aggregation, out_base):
    apply_multipanel_style()

    panel_h_mm = MULTIPANEL_DATA_COL_MM * MULTIPANEL_PANEL_HW_RATIO
    fig_h_mm = (
        3 * panel_h_mm
        + 2 * MULTIPANEL_ROW_GAP_MM
        + MULTIPANEL_MARGIN_V_MM
    )
    fig = plt.figure(
        figsize=(mm_to_inches(A4_WIDTH_MM), mm_to_inches(fig_h_mm)),
    )
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[MULTIPANEL_DATA_COL_MM, MULTIPANEL_LEGEND_COL_MM],
        height_ratios=[1, 1, 1],
        wspace=0.08,
        hspace=0.10,
    )

    ax_voxel = fig.add_subplot(gs[0, 0])
    ax_region = fig.add_subplot(gs[1, 0])
    ax_network = fig.add_subplot(gs[2, 0])
    ax_legend = fig.add_subplot(gs[:, 1])

    ylo, yhi = multipanel_shared_ylim(results_by_unit)
    data_axes = [
        (ax_voxel, "voxel", False),
        (ax_region, "region", False),
        (ax_network, "network", True),
    ]
    for ax, unit, show_xlabel in data_axes:
        df, stats_data, subject_stats = results_by_unit[unit]
        draw_similarity_panel(
            ax, df, sorted_tasks, subject_stats, stats_data, aggregation,
            title=UNIT_PANEL_TITLES[unit],
            show_ylabel=True,
            show_xlabel=show_xlabel,
            show_xticklabels=show_xlabel,
            scatter_size=8,
        )
        ax.set_ylim(ylo, yhi)

    add_panel_labels([ax_voxel, ax_region, ax_network], x=-0.11, y=1.08, fontsize=10)
    subject_stats_any = results_by_unit["network"][2]
    draw_legend_panel(
        ax_legend,
        aggregation,
        sorted_tasks,
        subjects_in_plot_from_stats(subject_stats_any, sorted_tasks),
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.20)
    save_figure(fig, out_base, formats=("svg", "pdf", "png"))
    plt.close(fig)

# ==========================================
# 2. MAIN
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    # Changed to directory input to handle multiple subjects
    ap.add_argument("--input_dir", required=True, help="Folder containing sub-XX_task_condition_betas.pkl files")
    ap.add_argument("--output_dir", required=True)
    
    # Analysis params
    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="region")
    ap.add_argument(
        "--multipanel_a4",
        action="store_true",
        help="3x1 A4-width figure: vertices, regions, networks + legend column on the right.",
    )
    ap.add_argument(
        "--replot_multipanel",
        action="store_true",
        help="Reuse cached multipanel bootstrap results (layout-only replot).",
    )
    ap.add_argument("--aggregation", choices=["condition", "task", "task_group"], default="task",
                    help="Use 'task' for the 11-task boxplot. 'condition' will make the plot very crowded.")
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument(
        "--max_reps_per_task",
        type=int,
        default=300,
        help="Cap repetitions per subject-task at vertex level (limits RAM). "
        "Set 0 to use all repetitions.",
    )
    ap.add_argument("--min_reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # Atlas params
    ap.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--network_assignments_path", help="Required if unit=network")
    
    # Valid vertex mask (excludes medial wall / zero-variance vertices)
    ap.add_argument("--apply_vertex_mask", type=int, default=1,
                    help="Apply valid vertex mask to exclude zero-variance vertices (default: 1)")
    ap.add_argument("--vertex_mask_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/valid_vertex_mask.npy",
                    help="Path to valid vertex mask")
    
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = None
    if args.apply_vertex_mask:
        print(f"Loading valid vertex mask from {args.vertex_mask_path}...")
        valid_mask = load_valid_vertex_mask(args.vertex_mask_path)
        n_valid = np.sum(valid_mask)
        n_total = len(valid_mask)
        print(f"  Mask: {n_valid:,}/{n_total:,} vertices ({n_valid/n_total*100:.1f}% valid)")

    atlas_unit = "network" if args.multipanel_a4 else args.unit
    avg_by_region, avg_by_network = None, None
    if atlas_unit in ("region", "network"):
        print(f"Loading Atlas from {args.atlas_path}...")
        avg_by_region, avg_by_network = setup_spatial_averagers(
            atlas_unit, args.atlas_path, args.network_assignments_path, valid_mask
        )

    subject_files, sorted_tasks = scan_subject_task_index(
        args.input_dir, args.aggregation, args.min_reps, valid_mask
    )
    print(f"Indexed {len(subject_files)} subjects. Aggregation: {args.aggregation}")
    if len(subject_files) < 2:
        raise SystemExit("Error: Need at least 2 subjects to calculate Across-Subject similarity.")

    max_reps = args.max_reps_per_task if args.max_reps_per_task > 0 else None
    if max_reps is not None:
        print(f"Vertex-level repetitions capped at {max_reps} per subject-task.")

    if args.multipanel_a4:
        cache_path = out_dir / f"similarity_multipanel_cache_{args.aggregation}.pkl"
        if args.replot_multipanel:
            if not cache_path.exists():
                raise SystemExit(f"No multipanel cache at {cache_path}. Run without --replot_multipanel first.")
            print(f"Loading multipanel cache from {cache_path}...")
            with open(cache_path, "rb") as f:
                results_by_unit, sorted_tasks = pickle.load(f)
        else:
            results_by_unit = {}
            for unit in MULTIPANEL_UNITS:
                print(f"Running {args.n_bootstrap} bootstraps per task ({unit})...")
                rng = np.random.default_rng(args.seed + hash(unit) % 10000)
                df, stats_data, subject_stats = run_bootstrap_for_unit(
                    subject_files,
                    sorted_tasks,
                    unit,
                    avg_by_region,
                    avg_by_network,
                    args.n_bootstrap,
                    rng,
                    args.aggregation,
                    args.min_reps,
                    valid_mask,
                    max_reps_per_task=max_reps,
                )
                results_by_unit[unit] = (df, stats_data, subject_stats)
                df.to_csv(
                    out_dir / f"similarity_bootstrap_results_{args.aggregation}_{unit}.csv",
                    index=False,
                )
            with open(cache_path, "wb") as f:
                pickle.dump((results_by_unit, sorted_tasks), f)
            print(f"Saved multipanel cache: {cache_path}")

        out_base = out_dir / f"similarity_boxplot_multipanel_a4_{args.aggregation}"
        plot_multipanel_a4(results_by_unit, sorted_tasks, args.aggregation, out_base)
        print(f"✅ SVG saved: {out_base.with_suffix('.svg')}")
        print(f"✅ PDF saved: {out_base.with_suffix('.pdf')}")
        print(f"✅ PNG saved: {out_base.with_suffix('.png')}")
        return

    rng = np.random.default_rng(args.seed)
    print(f"Running {args.n_bootstrap} bootstraps per task ({args.unit})...")
    df, stats_data, subject_stats = run_bootstrap_for_unit(
        subject_files,
        sorted_tasks,
        args.unit,
        avg_by_region if args.unit in ("region", "network") else None,
        avg_by_network if args.unit == "network" else None,
        args.n_bootstrap,
        rng,
        args.aggregation,
        args.min_reps,
        valid_mask,
        max_reps_per_task=max_reps if args.unit == "voxel" else None,
    )
    df.to_csv(out_dir / f"similarity_bootstrap_results_{args.aggregation}.csv", index=False)

    apply_a4_fullwidth_style()
    fig, ax = plt.subplots(
        figsize=(mm_to_inches(A4_WIDTH_MM), mm_to_inches(FIG_HEIGHT_MM))
    )
    draw_similarity_panel(
        ax, df, sorted_tasks, subject_stats, stats_data, args.aggregation,
        title=f"Spatial pattern similarity ({args.unit})",
        show_xlabel=True,
        show_ylabel=True,
        scatter_size=14,
    )
    add_side_legends(
        ax, args.aggregation, sorted_tasks,
        subjects_in_plot_from_stats(subject_stats, sorted_tasks),
    )
    fig.subplots_adjust(right=0.82)

    out_base = out_dir / f"similarity_boxplot_with_stats_{args.aggregation}_{args.unit}"
    save_figure(fig, out_base, formats=("svg", "pdf", "png"))
    print(f"✅ SVG saved: {out_base.with_suffix('.svg')}")
    print(f"✅ PDF saved: {out_base.with_suffix('.pdf')}")
    print(f"✅ PNG saved: {out_base.with_suffix('.png')}")
    print(f"✅ CSV saved: {out_dir / f'similarity_bootstrap_results_{args.aggregation}.csv'}")
    plt.close(fig)

if __name__ == "__main__":
    main()