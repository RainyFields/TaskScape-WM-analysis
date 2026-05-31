#!/usr/bin/env python3
"""
Noise ceiling with three granularities: voxel, Glasser region, or network (12 networks).
Now aggregates raw conditions into 11 specific Task Conditions before calculation.

Usage Example:
python noise_ceiling_tradition_task_level.py \
  --subject sub-03 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-01_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/traditional_way/task_level \
  --unit voxel --n_bootstrap 10 --min_reps 4
"""

#!/usr/bin/env python3
"""
Noise ceiling with three granularities: voxel, Glasser region, or network (12 networks).
Supports three levels of task aggregation:
  1. condition: Raw keys from input file (highest granularity).
  2. task: The 11 specific tasks (intermediate).
  3. task_group: 3 broad groups (nback, interdms, ctxdm).

Usage Example:

python noise_ceiling_final_version.py \
  --subject sub-03 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-01_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/final_version \
  --unit network \
  --aggregation condition \
  --n_bootstrap 10
"""

import argparse
import json 
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nibabel as nib

from utils import load_valid_vertex_mask, apply_mask_to_betas_dict

# ============================================================================
# GLOBAL MATPLOTLIB SETTINGS FOR ACADEMIC PUBLICATION
# ============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']

# Font sizes (strict academic standards)
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# Line and spine settings
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['lines.linewidth'] = 2.5

# Tick settings
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# PDF font embedding
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# Output DPI
OUTPUT_DPI = 300

# ---------- helpers ----------

def colwise_correlation(X, Y):
    """Column-wise Pearson r between matrices X and Y (same shape)."""
    assert X.shape == Y.shape, f"Shape mismatch: {X.shape} vs {Y.shape}"
    n = X.shape[1]
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        x = X[:, i]
        y = Y[:, i]
        # guard against constants or very short vectors
        if len(x) < 2: 
            continue
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        r, _ = pearsonr(x, y)
        out[i] = r
    return out


def spearman_brown(r, k=2.0):
    """Apply Spearman–Brown prophecy to a correlation r for test-length factor k."""
    return (k * r) / (1.0 + (k - 1.0) * r)


def load_glasser_labels(atlas_path: str):
    """Load Glasser .dlabel.nii/.nii labels -> (labels_vector, region_ids_sorted)."""
    lab_img = nib.load(atlas_path)
    lab = np.asarray(lab_img.get_fdata()).squeeze()
    lab = lab.reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids


def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    """Return a function avg_by_region(trials x features) -> (trials x n_regions)."""
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]

    def avg_by_region(A):
        out = np.full((A.shape[0], len(region_ids)), np.nan, dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx.size == 0:
                continue
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
            if len(cols) == 0:
                continue
            out[:, k] = np.nanmean(R[:, cols], axis=1)
        return out
    return avg_by_network


def zscore_columns(X, eps=1e-8):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def group_betas(raw_betas, aggregation_mode):
    """
    Groups raw dictionary {key: [beta,...]} into {target: [beta,...]} based on mode.
    Modes:
      - 'condition': Identity mapping (uses raw keys).
      - 'task': Maps to 11 specific tasks.
      - 'task_group': Maps to 3 broad groups (nback, interdms, ctxdm).
    """
    
    # 1. Define targets based on mode
    if aggregation_mode == 'task':
        targets = [
            'nback_loc', 'nback_ctg', 'nback_obj',
            'dms_loc',
            'ctxdm_col', 'ctxdm_lco',
            'interdms_loc_ABAB', 'interdms_loc_ABBA',
            'interdms_obj_ABAB', 'interdms_obj_ABBA',
            'interdms_ctg_ABAB', 'interdms_ctg_ABBA',
        ]
    elif aggregation_mode == 'task_group':
        targets = ['nback', 'dms', 'interdms', 'ctxdm']
    else: # 'condition'
        # Targets are just the keys present in the file
        targets = list(raw_betas.keys())

    # Initialize container
    grouped_betas = {k: [] for k in targets}
    
    # 2. Perform Grouping
    mapped_count = 0
    
    if aggregation_mode == 'condition':
        # Direct assignment
        grouped_betas = raw_betas
        mapped_count = len(raw_betas)
    else:
        # Substring matching for 'task' and 'task_group'
        # Sort by descending length to avoid collisions (e.g., 'dms' matching 'interdms').
        sorted_targets = sorted(targets, key=len, reverse=True)
        for raw_key, beta_list in raw_betas.items():
            matched = False
            for target in sorted_targets:
                # E.g. if target='nback' and raw_key='sub-01_task-nback_loc', it matches
                if target in raw_key:
                    grouped_betas[target].extend(beta_list)
                    matched = True
                    mapped_count += 1
                    break
    
    return grouped_betas, targets


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument("--min_reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # Unit selection
    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="voxel",
                    help="Spatial granularity of NC computation")

    # Aggregation Level
    ap.add_argument("--aggregation", choices=["condition", "task", "task_group"], default="task",
                    help="Level of task aggregation.")

    # Glasser options
    ap.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--network_assignments_path")
    ap.add_argument("--apply_spearman_brown", type=int, default=1)
    ap.add_argument("--standardize_halves", type=int, default=0)
    ap.add_argument("--network_mapping_json", default="")
    
    # Valid vertex mask (excludes medial wall / zero-variance vertices)
    ap.add_argument("--apply_vertex_mask", type=int, default=1,
                    help="Apply valid vertex mask to exclude zero-variance vertices (default: 1)")
    ap.add_argument("--vertex_mask_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/valid_vertex_mask.npy",
                    help="Path to valid vertex mask")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ==========================================
    # 1. LOAD DATA & GROUP
    # ==========================================
    input_path = Path(args.input_pickle)
    with open(input_path, "rb") as f:
        raw_betas = pickle.load(f)

    # Apply valid vertex mask to exclude zero-variance vertices
    valid_mask = None
    if args.apply_vertex_mask:
        print(f"Loading valid vertex mask from {args.vertex_mask_path}...")
        valid_mask = load_valid_vertex_mask(args.vertex_mask_path)
        n_valid = np.sum(valid_mask)
        n_total = len(valid_mask)
        print(f"  Applying mask: {n_valid:,}/{n_total:,} vertices ({n_valid/n_total*100:.1f}% valid)")
        raw_betas = apply_mask_to_betas_dict(raw_betas, valid_mask)

    print(f"--- Grouping Mode: {args.aggregation.upper()} ---")
    condition_betas, all_possible_targets = group_betas(raw_betas, args.aggregation)

    # Filter out conditions/tasks/groups that didn't get enough data
    valid_conditions = [k for k in all_possible_targets 
                        if k in condition_betas and len(condition_betas[k]) >= args.min_reps]

    if not valid_conditions:
        raise SystemExit(f"No valid conditions with >= {args.min_reps} reps found for mode {args.aggregation}.")
    
    print(f"Valid items for calculation: {len(valid_conditions)}")
    # Print a few examples if there are too many
    if len(valid_conditions) > 20:
        print(f"  (First 5): {valid_conditions[:5]} ...")
    else:
        for vc in valid_conditions:
            print(f"  - {vc}: {len(condition_betas[vc])} repetitions")

    # Warn if using 'task_group' (N=3)
    if args.aggregation == 'task_group' and len(valid_conditions) <= 3:
        print("\nWARNING: Aggregation is 'task_group' (N<=3). Pearson correlation on 3 points is statistically unstable.\n")

    # infer feature length
    first_vec = np.asarray(condition_betas[valid_conditions[0]][0]).reshape(-1)
    n_features = first_vec.size 

    # ==========================================
    # 2. PREP UNIT TRANSFORMS
    # ==========================================
    avg_by_region = None
    avg_by_network = None
    region_ids = None

    if args.unit in ("region", "network"):
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        # Apply vertex mask to atlas labels if mask was applied to betas
        if valid_mask is not None:
            labels_vec = labels_vec[valid_mask]
        if labels_vec.size != n_features:
            raise SystemExit(f"Atlas size ({labels_vec.size}) != beta size ({n_features}).")
        avg_by_region = make_region_averager(labels_vec, region_ids)

    if args.unit == "network":
        if not args.network_assignments_path:
            raise SystemExit("--network_assignments_path is required for --unit=network")
        max_region_id = int(region_ids.max())
        region_to_net = load_region_to_network_map(args.network_assignments_path, max_region_id)
        avg_by_network = make_network_averager(region_ids, region_to_net, n_networks=12)

        # network name mapping (optional)
        if args.network_mapping_json and Path(args.network_mapping_json).exists():
            with open(args.network_mapping_json, "r") as f:
                network_name_map = {int(k): v for k, v in json.load(f).items()}
        else:
            network_name_map = {i: f"Net-{i}" for i in range(1,13)}

    # unit meta
    if args.unit == "voxel":
        n_units = n_features
        unit_name = "voxel"
    elif args.unit == "region":
        n_units = len(region_ids)
        unit_name = "region"
    else:
        n_units = 12
        unit_name = "network"

    # ==========================================
    # 3. BOOTSTRAP SPLIT-HALF NC
    # ==========================================
    boot_nc = []

    for b in range(args.n_bootstrap):
        half1_rows = []
        half2_rows = []

        # Loop over the valid items (conditions OR tasks OR groups)
        for cond in valid_conditions:
            reps = np.stack([np.asarray(v).reshape(-1) for v in condition_betas[cond]], axis=0)
            
            if avg_by_region:
                reps = avg_by_region(reps)
            if avg_by_network:
                reps = avg_by_network(reps)

            # split reps
            perm = rng.permutation(reps.shape[0])
            mid = reps.shape[0] // 2
            
            A = reps[perm[:mid]].mean(axis=0)
            B = reps[perm[mid:]].mean(axis=0)

            if A.size == 0 or B.size == 0:
                continue

            half1_rows.append(A)
            half2_rows.append(B)

        if not half1_rows or not half2_rows:
            continue

        H1 = np.stack(half1_rows, axis=0)
        H2 = np.stack(half2_rows, axis=0)

        if args.standardize_halves:
            H1 = zscore_columns(H1)
            H2 = zscore_columns(H2)

        r_cols = colwise_correlation(H1, H2)

        if args.apply_spearman_brown:
            r_cols = np.clip(r_cols, -0.9999, 0.9999)
            r_cols = spearman_brown(r_cols, k=2.0)

        boot_nc.append(r_cols)

    if not boot_nc:
        raise SystemExit("No bootstrap iterations produced valid splits.")

    noise_ceiling = np.nanmean(np.stack(boot_nc, axis=0), axis=0)
    noise_ceiling_clipped = np.clip(noise_ceiling, -1, 1)

    # ==========================================
    # 4. CALCULATE STATISTICS (including negative ratio)
    # ==========================================
    valid_nc = noise_ceiling_clipped[np.isfinite(noise_ceiling_clipped)]
    
    n_total_units = len(noise_ceiling_clipped)
    n_nan = np.sum(~np.isfinite(noise_ceiling_clipped))
    n_valid = len(valid_nc)
    n_negative = np.sum(valid_nc < 0)
    n_positive = np.sum(valid_nc >= 0)
    
    ratio_negative = n_negative / n_valid if n_valid > 0 else 0
    ratio_positive = n_positive / n_valid if n_valid > 0 else 0
    
    mean_nc = np.mean(valid_nc)
    median_nc = np.median(valid_nc)
    std_nc = np.std(valid_nc)
    min_nc = np.min(valid_nc)
    max_nc = np.max(valid_nc)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("NOISE CEILING STATISTICS")
    print("=" * 60)
    print(f"Subject:            {args.subject}")
    print(f"Aggregation:        {args.aggregation}")
    print(f"Spatial Unit:       {unit_name}")
    print("-" * 60)
    print(f"Total units:        {n_total_units:,}")
    print(f"Valid (non-NaN):    {n_valid:,}")
    print(f"NaN units:          {n_nan:,}")
    print("-" * 60)
    print(f"Negative (< 0):     {n_negative:,} ({ratio_negative*100:.2f}%)")
    print(f"Positive (>= 0):    {n_positive:,} ({ratio_positive*100:.2f}%)")
    print("-" * 60)
    print(f"Mean NC:            {mean_nc:.4f}")
    print(f"Median NC:          {median_nc:.4f}")
    print(f"Std NC:             {std_nc:.4f}")
    print(f"Min NC:             {min_nc:.4f}")
    print(f"Max NC:             {max_nc:.4f}")
    print("=" * 60 + "\n")
    
    # Compile statistics dictionary
    nc_stats = {
        "subject": args.subject,
        "aggregation": args.aggregation,
        "unit": unit_name,
        "apply_vertex_mask": bool(args.apply_vertex_mask),
        "apply_spearman_brown": bool(args.apply_spearman_brown),
        "n_bootstrap": args.n_bootstrap,
        "n_total_units": int(n_total_units),
        "n_valid_units": int(n_valid),
        "n_nan_units": int(n_nan),
        "n_negative": int(n_negative),
        "n_positive": int(n_positive),
        "ratio_negative": float(ratio_negative),
        "ratio_positive": float(ratio_positive),
        "positive_only_saved": True,
        "mean_nc": float(mean_nc),
        "median_nc": float(median_nc),
        "std_nc": float(std_nc),
        "min_nc": float(min_nc),
        "max_nc": float(max_nc),
    }

    # ==========================================
    # 5. SAVE & PLOT
    # ==========================================
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Construct base filename with aggregation info
    base_name = f"{args.subject}_agg-{args.aggregation}_unit-{unit_name}"

    # --- ENHANCED PLOT (Publication Quality with Negative Highlighting) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bins spanning the full range
    bin_edges = np.linspace(-1, 1, 61)
    
    # Split data into negative and positive
    negative_vals = valid_nc[valid_nc < 0]
    positive_vals = valid_nc[valid_nc >= 0]
    
    # Plot negative values in red
    if len(negative_vals) > 0:
        ax.hist(negative_vals, bins=bin_edges, color='#e74c3c', alpha=0.8, 
                edgecolor='#c0392b', linewidth=1.5,
                label=f'Negative (n={len(negative_vals):,}, {ratio_negative*100:.1f}%)')
    
    # Plot positive values in green
    if len(positive_vals) > 0:
        ax.hist(positive_vals, bins=bin_edges, color='#2ecc71', alpha=0.8, 
                edgecolor='#27ae60', linewidth=1.5,
                label=f'Positive (n={len(positive_vals):,}, {ratio_positive*100:.1f}%)')
    
    # Add vertical lines for statistics
    ax.axvline(0, color='black', linewidth=2.5, linestyle='-', label='Zero', alpha=0.7)
    ax.axvline(mean_nc, color='#3498db', linewidth=3, linestyle='--', 
               label=f'Mean = {mean_nc:.3f}')
    ax.axvline(median_nc, color='#9b59b6', linewidth=3, linestyle=':', 
               label=f'Median = {median_nc:.3f}')
    
    # Title and labels
    ax.set_title(f"Noise Ceiling Distribution: {args.subject}\n(Aggregation: {args.aggregation}, Unit: {unit_name})",
                 fontweight='bold', pad=15)
    ax.set_xlabel("Noise Ceiling (Pearson r)")
    ax.set_ylabel(f"Number of {unit_name}s")
    
    # Legend
    ax.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=14,
              framealpha=0.95)
    
    # Statistics annotation box
    stats_text = (f"Total: {n_valid:,}\n"
                  f"Negative: {n_negative:,} ({ratio_negative*100:.1f}%)\n"
                  f"Mean: {mean_nc:.3f}\n"
                  f"Median: {median_nc:.3f}\n"
                  f"Std: {std_nc:.3f}")
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', 
                      linewidth=2, alpha=0.95))
    
    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', length=10, width=2.0, direction='out')
    ax.tick_params(axis='both', which='minor', length=5, width=1.5, direction='out')
    ax.minorticks_on()
    
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, alpha=0.3, linewidth=1.0)
    plt.tight_layout()

    # Save PNG
    png_path = outdir / f"{base_name}_dist.png"
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Save PDF
    pdf_path = outdir / f"{base_name}_dist.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close(fig)

    # --- POSITIVE-ONLY PLOT (Publication Quality) ---
    if len(positive_vals) > 0:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        # Create bins for positive range only
        bin_edges_pos = np.linspace(0, 1, 51)
        
        # Plot positive values
        ax2.hist(positive_vals, bins=bin_edges_pos, color='#2ecc71', alpha=0.8, 
                 edgecolor='#27ae60', linewidth=1.5)
        
        # Statistics for positive values only
        mean_pos = np.mean(positive_vals)
        median_pos = np.median(positive_vals)
        std_pos = np.std(positive_vals)
        
        # Add vertical lines for statistics
        ax2.axvline(mean_pos, color='#3498db', linewidth=3, linestyle='--', 
                    label=f'Mean = {mean_pos:.3f}')
        ax2.axvline(median_pos, color='#9b59b6', linewidth=3, linestyle=':', 
                    label=f'Median = {median_pos:.3f}')
        
        # Title and labels
        ax2.set_title(f"Noise Ceiling Distribution (Positive Only): {args.subject}\n(Aggregation: {args.aggregation}, Unit: {unit_name})",
                      fontweight='bold', pad=15)
        ax2.set_xlabel("Noise Ceiling (Pearson r)")
        ax2.set_ylabel(f"Number of {unit_name}s")
        
        # Legend
        ax2.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=14,
                   framealpha=0.95)
        
        # Statistics annotation box
        stats_text_pos = (f"N (positive): {len(positive_vals):,}\n"
                          f"Mean: {mean_pos:.3f}\n"
                          f"Median: {median_pos:.3f}\n"
                          f"Std: {std_pos:.3f}\n"
                          f"Min: {np.min(positive_vals):.3f}\n"
                          f"Max: {np.max(positive_vals):.3f}")
        ax2.text(0.98, 0.97, stats_text_pos, transform=ax2.transAxes, fontsize=13,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', 
                           linewidth=2, alpha=0.95))
        
        # Thicken spines
        for spine in ax2.spines.values():
            spine.set_linewidth(2.0)
        
        # Tick parameters
        ax2.tick_params(axis='both', which='major', length=10, width=2.0, direction='out')
        ax2.tick_params(axis='both', which='minor', length=5, width=1.5, direction='out')
        ax2.minorticks_on()
        
        ax2.set_xlim(-0.02, 1.02)
        ax2.grid(True, alpha=0.3, linewidth=1.0)
        plt.tight_layout()

        # Save positive-only PNG
        png_path_pos = outdir / f"{base_name}_dist_positive.png"
        fig2.savefig(png_path_pos, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        # Save positive-only PDF
        pdf_path_pos = outdir / f"{base_name}_dist_positive.pdf"
        fig2.savefig(pdf_path_pos, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close(fig2)
        
        print(f"PNG (positive only) saved: {png_path_pos}")
        print(f"PDF (positive only) saved: {pdf_path_pos}")

    # --- SAVE FILES ---
    # For voxel-level data, expand back to full 64k domain with NaN for invalid vertices
    # This allows proper surface visualization (invalid vertices appear grey)
    if args.unit == "voxel" and valid_mask is not None:
        n_total_vertices = len(valid_mask)  # Should be 64984
        noise_ceiling_full = np.full(n_total_vertices, np.nan, dtype=np.float32)
        noise_ceiling_full[valid_mask] = noise_ceiling_clipped
        nc_to_save = noise_ceiling_full
        print(f"Expanded voxel data to full {n_total_vertices:,} vertices (invalid = NaN for grey on surface)")
    else:
        nc_to_save = noise_ceiling_clipped
    
    # Save signed data for QC/debug
    signed_npy_path = outdir / f"{base_name}_signed.npy"
    np.save(signed_npy_path, nc_to_save)
    signed_pkl_path = outdir / f"{base_name}_signed.pkl"
    with open(signed_pkl_path, "wb") as f:
        pickle.dump(nc_to_save, f)

    # Save positive-only data for visualization (negative/zero -> NaN)
    nc_positive = nc_to_save.copy()
    nc_positive[~np.isfinite(nc_positive)] = np.nan
    nc_positive[nc_positive <= 0] = np.nan

    npy_save_path = outdir / f"{base_name}.npy"
    np.save(npy_save_path, nc_positive)
    nc_save_path = outdir / f"{base_name}.pkl"
    with open(nc_save_path, "wb") as f:
        pickle.dump(nc_positive, f)

    # Save statistics JSON (always)
    stats_json_path = outdir / f"{base_name}_stats.json"
    with open(stats_json_path, "w") as f:
        json.dump(nc_stats, f, indent=2)

    print(f"PNG saved: {png_path}")
    print(f"PDF saved: {pdf_path}")
    print(f"Data saved (positive-only): {npy_save_path}")
    print(f"Data saved (signed): {signed_npy_path}")
    print(f"Stats saved: {stats_json_path}")

    # Additional network-level summary if unit is network
    if args.unit == "network":
        net_stats = []
        for k in range(12):
            val = float(noise_ceiling_clipped[k])
            net_stats.append({
                "network_id": k + 1,
                "name": network_name_map.get(k + 1, str(k+1)),
                "noise_ceiling": val
            })
        
        # Add network-specific info to stats
        nc_stats["networks"] = net_stats
            
        network_json_path = outdir / f"{base_name}_network_summary.json"
        with open(network_json_path, "w") as f:
            json.dump({
                "subject": args.subject, 
                "aggregation": args.aggregation,
                "unit": args.unit,
                "stats": nc_stats,
                "networks": net_stats
            }, f, indent=2)
        print(f"Network summary saved: {network_json_path}")

if __name__ == "__main__":
    main()