#!/usr/bin/env python3
"""
Multi-Subject Similarity Analysis (Within vs. Across) with Statistical Testing.
 
Usage Example:
unit choose from: voxel, region and network
aggregation choose from: condition, task, task_group


python plot_per_task_within_across_subj_similarity.py \
  --input_dir /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/similarity_analysis \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --unit network \
  --aggregation task_group \
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
from figure_utils import mm_to_inches, save_figure, apply_clean_spines

STYLE_PATH = FIG_STYLE_ROOT / "nature_style.mplstyle"

# A4 page width; height slightly below the default aspect ratio.
A4_WIDTH_MM = 210.0
FIG_HEIGHT_MM = 72.0

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
    ap.add_argument("--aggregation", choices=["condition", "task", "task_group"], default="task",
                    help="Use 'task' for the 11-task boxplot. 'condition' will make the plot very crowded.")
    ap.add_argument("--n_bootstrap", type=int, default=50)
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
    rng = np.random.default_rng(args.seed)
    
    # -----------------------------
    # A. Load Valid Vertex Mask
    # -----------------------------
    valid_mask = None
    if args.apply_vertex_mask:
        print(f"Loading valid vertex mask from {args.vertex_mask_path}...")
        valid_mask = load_valid_vertex_mask(args.vertex_mask_path)
        n_valid = np.sum(valid_mask)
        n_total = len(valid_mask)
        print(f"  Mask: {n_valid:,}/{n_total:,} vertices ({n_valid/n_total*100:.1f}% valid)")
    
    # -----------------------------
    # B. Setup Spatial Averagers
    # -----------------------------
    avg_by_region = None
    avg_by_network = None
    
    if args.unit in ("region", "network"):
        print(f"Loading Atlas from {args.atlas_path}...")
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        # Apply vertex mask to atlas labels if mask was applied
        if valid_mask is not None:
            labels_vec = labels_vec[valid_mask]
        avg_by_region = make_region_averager(labels_vec, region_ids)
        
        if args.unit == "network":
            if not args.network_assignments_path:
                raise SystemExit("Need --network_assignments_path for network unit.")
            max_rid = int(region_ids.max())
            r2n = load_region_to_network_map(args.network_assignments_path, max_rid)
            avg_by_network = make_network_averager(region_ids, r2n)

    # -----------------------------
    # C. Load All Subjects
    # -----------------------------
    input_dir = Path(args.input_dir)
    # Automatically matching your file pattern
    files = sorted(list(input_dir.glob("*task_condition_betas.pkl")))
    
    if not files:
        raise SystemExit(f"No pickle files found in {input_dir}. Check path.")

    print(f"Found {len(files)} subject files.")
    
    all_data = {}
    valid_tasks = set()

    for fp in files:
        # Extract subject ID from filename (assuming sub-XX_...)
        sub_id = fp.name.split('_')[0] 
        
        with open(fp, "rb") as f:
            raw = pickle.load(f)
        
        # Apply valid vertex mask to exclude zero-variance vertices
        if valid_mask is not None:
            raw = apply_mask_to_betas_dict(raw, valid_mask)
        
        grouped, possible_tasks = group_betas(raw, args.aggregation)
        
        sub_data = {}
        for t in possible_tasks:
            if t in grouped and len(grouped[t]) >= args.min_reps:
                # Stack repetitions
                mat = np.stack([np.asarray(x).reshape(-1) for x in grouped[t]], axis=0)
                
                # Apply Spatial Transform
                if avg_by_region:
                    mat = avg_by_region(mat)
                if avg_by_network:
                    mat = avg_by_network(mat)
                
                sub_data[t] = mat
                valid_tasks.add(t)
        
        if sub_data:
            all_data[sub_id] = sub_data

    sorted_tasks = order_tasks(valid_tasks, args.aggregation)
    print(f"Loaded {len(all_data)} subjects. Analysis Aggregation: {args.aggregation}")
    if len(all_data) < 2:
        raise SystemExit("Error: Need at least 2 subjects to calculate Across-Subject similarity.")

    # -----------------------------
    # C. Bootstrap (Split-Everything)
    # -----------------------------
    # We will store flattened data for the DataFrame later
    # Structure: [{'Task': t, 'Type': 'Within', 'Score': 0.85}, ...]
    plot_data = []
    
    # Store arrays for statistical testing: stats_data[task] = (within_array, across_array)
    stats_data = {}

    print(f"Running {args.n_bootstrap} bootstraps per task...")

    for t in sorted_tasks:
        subs_with_task = [s for s in all_data.keys() if t in all_data[s]]
        if len(subs_with_task) < 2:
            continue

        task_within = []
        task_across = []

        for b in range(args.n_bootstrap):
            H1_list = []
            H2_list = []
            
            # Generate split-halves for every subject
            for s in subs_with_task:
                mat = all_data[s][t]
                n_reps = mat.shape[0]
                perm = rng.permutation(n_reps)
                mid = n_reps // 2
                
                h1 = mat[perm[:mid]].mean(axis=0)
                h2 = mat[perm[mid:]].mean(axis=0)
                H1_list.append(h1)
                H2_list.append(h2)
            
            M1 = np.stack(H1_list, axis=0)
            M2 = np.stack(H2_list, axis=0)
            
            # Z-score rows (spatial pattern)
            M1_z = (M1 - M1.mean(axis=1, keepdims=True)) / (M1.std(axis=1, keepdims=True) + 1e-8)
            M2_z = (M2 - M2.mean(axis=1, keepdims=True)) / (M2.std(axis=1, keepdims=True) + 1e-8)
            
            # Correlation Matrix: (n_subs x n_subs)
            R = np.dot(M1_z, M2_z.T) / M1.shape[1]
            
            # Extract Metrics
            within_mean = np.mean(np.diag(R))
            
            mask = ~np.eye(R.shape[0], dtype=bool)
            across_mean = np.mean(R[mask])
            
            task_within.append(within_mean)
            task_across.append(across_mean)

            # Append to plot data
            plot_data.append({'Task': t, 'Type': 'Within', 'Similarity': within_mean})
            plot_data.append({'Task': t, 'Type': 'Across', 'Similarity': across_mean})

        # Save for stats
        stats_data[t] = (np.array(task_within), np.array(task_across))

    # -----------------------------
    # D. Stats & Plotting
    # -----------------------------
    df = pd.DataFrame(plot_data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw stats
    df.to_csv(out_dir / f"similarity_bootstrap_results_{args.aggregation}.csv", index=False)

    apply_a4_fullwidth_style()
    fig, ax = plt.subplots(
        figsize=(mm_to_inches(A4_WIDTH_MM), mm_to_inches(FIG_HEIGHT_MM))
    )

    # Draw side-by-side boxplots per task with task-category color coding.
    x_positions = np.arange(len(sorted_tasks))
    offset = 0.18
    width = 0.30

    for i, task in enumerate(sorted_tasks):
        w = df[(df["Task"] == task) & (df["Type"] == "Within")]["Similarity"].values
        a = df[(df["Task"] == task) & (df["Type"] == "Across")]["Similarity"].values
        c = get_task_color(task, args.aggregation)

        if len(w) > 0:
            bp_w = ax.boxplot(
                [w], positions=[i - offset], widths=width, patch_artist=True,
                showfliers=False, manage_ticks=False
            )
            bp_w["boxes"][0].set_facecolor(c)
            bp_w["boxes"][0].set_alpha(0.85)
            bp_w["boxes"][0].set_linewidth(0.8)
            for e in ["whiskers", "caps", "medians"]:
                for line in bp_w[e]:
                    line.set_color("black")
                    line.set_linewidth(0.8)

        if len(a) > 0:
            bp_a = ax.boxplot(
                [a], positions=[i + offset], widths=width, patch_artist=True,
                showfliers=False, manage_ticks=False
            )
            bp_a["boxes"][0].set_facecolor(c)
            bp_a["boxes"][0].set_alpha(0.35)
            bp_a["boxes"][0].set_linewidth(0.8)
            for e in ["whiskers", "caps", "medians"]:
                for line in bp_a[e]:
                    line.set_color("black")
                    line.set_linewidth(0.8)
    
    # Calculate and Annotate Stats
    # Get y-limits to place stars correctly
    y_max = df['Similarity'].max()
    y_range = y_max - df['Similarity'].min()
    h = y_range * 0.05
    
    for i, t in enumerate(sorted_tasks):
        if t not in stats_data: continue
        
        w_arr, a_arr = stats_data[t]
        # Paired T-test on the bootstrap distributions
        stat, pval = ttest_rel(w_arr, a_arr)
        
        # Determine significance marker
        if pval < 0.001: star = "***"
        elif pval < 0.01: star = "**"
        elif pval < 0.05: star = "*"
        else: star = "ns"
        
        # Find position
        # Boxplot places categories at 0, 1, 2... 
        # Hue offsets them. We want the center of the pair.
        x_center = i
        
        # Current max y for this specific task to place bracket just above
        curr_y = df[df['Task'] == t]['Similarity'].max() + (h * 0.5)
        
        annot_stat(star, x_center - offset, x_center + offset, curr_y, h * 0.4, ax)

    ax.set_title(f"Spatial pattern similarity ({args.unit})", pad=4)
    ax.set_xlabel("Task")
    ax.set_ylabel("Similarity (Pearson r)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_tasks, rotation=35, ha="right")
    apply_clean_spines(ax)

    # Task-category legend (one color per category)
    if args.aggregation == "task_group":
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

    # Within/Across encoding via alpha on the same category color
    within_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#4D4D4D", alpha=0.85,
        edgecolor="black", linewidth=0.5,
    )
    across_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#4D4D4D", alpha=0.35,
        edgecolor="black", linewidth=0.5,
    )

    legend_task = ax.legend(
        category_patches, category_labels,
        title="Task", loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, borderaxespad=0.0,
    )
    ax.add_artist(legend_task)
    ax.legend(
        [within_patch, across_patch], ["Within-subject", "Across-subject"],
        title="Comparison", loc="upper left", bbox_to_anchor=(1.01, 0.45),
        frameon=False, borderaxespad=0.0,
    )

    fig.subplots_adjust(right=0.82)

    base_name = f"similarity_boxplot_with_stats_{args.aggregation}_{args.unit}"
    out_base = out_dir / base_name
    save_figure(fig, out_base, formats=("svg", "pdf", "png"))

    print(f"✅ SVG saved: {out_base.with_suffix('.svg')}")
    print(f"✅ PDF saved: {out_base.with_suffix('.pdf')}")
    print(f"✅ PNG saved: {out_base.with_suffix('.png')}")
    print(f"✅ CSV saved: {out_dir / f'similarity_bootstrap_results_{args.aggregation}.csv'}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()