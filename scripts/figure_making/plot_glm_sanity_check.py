#!/usr/bin/env python3
"""
GLM Sanity Check Visualization
Generates publication-quality figures showing:
1. Raw BOLD signals (10 random parcels) - time series without any fit
2. BOLD signals with GLM fit overlay - observed vs fitted comparison
3. Design matrix decomposed into confounds, drifts, and task regressors
4. R² distribution across all parcels

Usage:
    python plot_glm_sanity_check.py
"""

import argparse
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import paths  # noqa: E402  TaskScape-WM inputs only; rejects the legacy tree

# ============================================================================
# GLOBAL MATPLOTLIB SETTINGS - Nature Scientific Data Standards
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
# CONFIGURATION
# ============================================================================
SUBJ = "sub-03"
TR = 1.49  # seconds
TMASK = 1  # frames dropped at start
N_VOXELS_SAMPLE = 10  # number of voxels to visualize

# Paths (set in main() after paths.configure()):
#   BETA_ROOT  -> <dataset>/derivatives/glm/<SUBJ>           (legacy equivalent: data/encoding_delay/trial_level_betas/<SUBJ>)
#   BOLD       -> derivatives/taskscape-surface/<sub>/<ses>/func/*_space-fsLR_den-32k_bold.dtseries.nii (unsmoothed, as before)
#                 (legacy equivalent: data/glasser_resampled/<SUBJ>/<ses>/*_space-Glasser64k_bold.dtseries.nii)
#   OUTPUT_DIR -> <output_root>/figures/glm_sanity_figures  (legacy equivalent: scripts/figure_making/results/glm_sanity_figures)
BETA_ROOT: Path | None = None
OUTPUT_DIR: Path | None = None


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", default=None, help="TaskScape-WM root (default: config.paths / $TASKSCAPE_DATASET_ROOT)")
    p.add_argument("--output_root", default=None, help="analysis output root (default: config.paths / $TASKSCAPE_OUTPUT_ROOT)")
    return p.parse_args()


def find_first_run(subj: str):
    """Find the first available run for a subject."""
    beta_dir = BETA_ROOT
    if not beta_dir.exists():
        raise FileNotFoundError(f"Beta directory not found: {beta_dir}")
    
    # Find first session
    sessions = sorted([d for d in beta_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")])
    if not sessions:
        raise FileNotFoundError(f"No sessions found in {beta_dir}")
    
    ses = sessions[0]
    func_dir = ses / "func"
    
    # Find first h5 file
    h5_files = sorted(func_dir.glob("*_betas.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No beta files found in {func_dir}")
    
    return h5_files[0]


def load_glm_data(h5_path: Path):
    """Load GLM outputs from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        betas_full = f["betas_full"][:]  # (P x K)
        design_matrix = f["design_matrix"][:]  # (T_kept x K)
        col_names = [n.decode() if isinstance(n, bytes) else n for n in f["design_col_names"][:]]
        
        # Get attributes
        task_col_start = f.attrs["task_col_start"]
        task_col_end = f.attrs["task_col_end"]
        high_pass_sec = f.attrs.get("high_pass_sec", 128.0)
        estimator = f.attrs.get("estimator", "unknown")
        
        # Load yhat if available, otherwise compute
        if "yhat" in f:
            yhat = f["yhat"][:]
        else:
            yhat = design_matrix @ betas_full.T  # (T x P)
    
    return {
        "betas_full": betas_full,
        "design_matrix": design_matrix,
        "col_names": col_names,
        "task_col_start": task_col_start,
        "task_col_end": task_col_end,
        "high_pass_sec": high_pass_sec,
        "estimator": estimator,
        "yhat": yhat,
    }


def load_bold_data(h5_path: Path):
    """Load original BOLD data corresponding to the GLM run."""
    # Parse filename to get timeseries path
    # e.g., sub-03_ses-001_task-ctxdm_run-1_betas.h5
    fname = h5_path.stem.replace("_betas", "")
    parts = fname.split("_")
    subj = parts[0]
    ses = parts[1]
    task = parts[2].replace("task-", "")
    run = parts[3]

    # Unsmoothed surface BOLD (same choice as the legacy script); paths.run_label() normalises run-1 / run-01.
    ts_path = paths.surface_bold(subj, ses, task, run, smoothed=False)

    if not ts_path.exists():
        raise FileNotFoundError(f"Timeseries not found: {ts_path}")
    
    Y = nib.load(str(ts_path)).get_fdata(dtype=np.float32)  # (T x P)
    
    # Apply tmask (drop first frame)
    Y = Y[TMASK:, :]
    
    return Y, ts_path


def decompose_design_matrix(design_matrix, col_names, task_col_start):
    """Decompose design matrix into confounds, drifts, and task components."""
    # Find where drifts start (cosine or constant columns)
    drift_start = None
    for i, name in enumerate(col_names):
        if name.startswith("cosine") or name == "constant":
            drift_start = i
            break
    
    if drift_start is None:
        # Assume confounds are first, then drifts before task
        drift_start = task_col_start - 10  # rough estimate
        drift_start = max(0, drift_start)
    
    # Decompose
    confounds = design_matrix[:, :drift_start]
    drifts = design_matrix[:, drift_start:task_col_start]
    task = design_matrix[:, task_col_start:]
    
    confound_names = col_names[:drift_start]
    drift_names = col_names[drift_start:task_col_start]
    task_names = col_names[task_col_start:]
    
    return {
        "confounds": confounds,
        "drifts": drifts,
        "task": task,
        "confound_names": confound_names,
        "drift_names": drift_names,
        "task_names": task_names,
    }


def compute_r2(Y, yhat):
    """Compute R² for each voxel."""
    ss_res = np.sum((Y - yhat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    return r2


def plot_raw_bold_signals(Y, time_vec, voxel_indices, output_dir, run_name):
    """Plot raw BOLD signals for sampled voxels (without GLM fit)."""
    n_voxels = len(voxel_indices)
    
    # Nature standard: single column = 89mm, double = 183mm
    width_mm = 183
    height_mm = 15 * n_voxels + 20
    fig, axes = plt.subplots(n_voxels, 1, figsize=(width_mm/25.4, height_mm/25.4), sharex=True)
    
    if n_voxels == 1:
        axes = [axes]
    
    for i, (ax, vox_idx) in enumerate(zip(axes, voxel_indices)):
        # Plot raw BOLD signal - black only
        ax.plot(time_vec, Y[:, vox_idx], color='black', linewidth=0.5, alpha=0.9)
        
        # Add mean and std info
        mean_val = np.mean(Y[:, vox_idx])
        std_val = np.std(Y[:, vox_idx])
        
        ax.set_ylabel(f"Parcel {vox_idx}", fontsize=6)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.5)
        
        # Add stats annotation
        ax.text(0.98, 0.95, f"mean={mean_val:.1f}, std={std_val:.1f}", 
                transform=ax.transAxes, fontsize=5, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                         linewidth=0.5, alpha=0.8))
        
        ax.set_xlim(time_vec[0], time_vec[-1])
    
    axes[-1].set_xlabel("Time (s)", fontsize=7)
    fig.suptitle(f"Raw BOLD signal - {run_name}", fontsize=8, y=1.01)
    
    # Add common y-axis label
    fig.text(0.02, 0.5, 'BOLD signal (a.u.)', va='center', rotation='vertical', fontsize=7)
    
    plt.tight_layout()
    fig.subplots_adjust(left=0.08)
    
    # Save
    fig.savefig(output_dir / f"raw_bold_signals_{run_name}.png", dpi=600, 
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"raw_bold_signals_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved raw BOLD signals figure")
    plt.close(fig)


def plot_bold_signals(Y, yhat, time_vec, voxel_indices, output_dir, run_name):
    """Plot BOLD signals for sampled voxels with fitted overlay."""
    n_voxels = len(voxel_indices)
    
    # Nature standard dimensions
    width_mm = 183
    height_mm = 15 * n_voxels + 20
    fig, axes = plt.subplots(n_voxels, 1, figsize=(width_mm/25.4, height_mm/25.4), sharex=True)
    
    if n_voxels == 1:
        axes = [axes]
    
    for i, (ax, vox_idx) in enumerate(zip(axes, voxel_indices)):
        # Plot actual BOLD - solid black
        ax.plot(time_vec, Y[:, vox_idx], color='black', linewidth=0.5, 
                label='Observed', alpha=0.8)
        # Plot fitted - dashed black
        ax.plot(time_vec, yhat[:, vox_idx], color='black', linewidth=0.5, 
                label='GLM fit', linestyle='--', alpha=0.9)
        
        # Compute R² for this voxel
        r2_vox = compute_r2(Y[:, vox_idx:vox_idx+1], yhat[:, vox_idx:vox_idx+1])[0]
        
        ax.set_ylabel(f"Parcel {vox_idx}\n(R²={r2_vox:.3f})", fontsize=6)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(0.5)
        
        if i == 0:
            ax.legend(loc='upper right', frameon=True, edgecolor='black', 
                     fontsize=5, framealpha=0.9)
        
        ax.set_xlim(time_vec[0], time_vec[-1])
    
    axes[-1].set_xlabel("Time (s)", fontsize=7)
    fig.suptitle(f"BOLD signal and GLM fit - {run_name}", fontsize=8, y=1.01)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / f"bold_signals_{run_name}.png", dpi=600, 
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"bold_signals_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved BOLD signals figure")
    plt.close(fig)


def plot_design_matrix(decomposed, time_vec, output_dir, run_name):
    """Plot full design matrix with task regressors and confounds combined."""
    # Combine all components: confounds, drifts, task
    full_matrix = np.hstack([
        decomposed["confounds"],
        decomposed["drifts"],
        decomposed["task"]
    ])
    
    n_confounds = decomposed["confounds"].shape[1]
    n_drifts = decomposed["drifts"].shape[1]
    n_task = decomposed["task"].shape[1]
    n_total = full_matrix.shape[1]
    
    # Nature standard: single column width
    width_mm = 89
    height_mm = 70
    fig, ax = plt.subplots(figsize=(width_mm/25.4, height_mm/25.4))
    
    # Normalize for visualization
    matrix_norm = (full_matrix - full_matrix.min(axis=0)) / (full_matrix.max(axis=0) - full_matrix.min(axis=0) + 1e-10)
    
    # Use grayscale colormap (black only)
    im = ax.imshow(matrix_norm, aspect='auto', cmap='Greys', 
                   extent=[0, n_total, time_vec[-1], time_vec[0]])
    
    # Add vertical lines to separate components
    ax.axvline(n_confounds, color='black', linewidth=0.5, linestyle='--', alpha=0.7)
    ax.axvline(n_confounds + n_drifts, color='black', linewidth=0.5, linestyle='--', alpha=0.7)
    
    # Add section labels at bottom
    ax.text(n_confounds/2, time_vec[-1] + 5, f'Confounds\n({n_confounds})', 
            ha='center', va='top', fontsize=5)
    ax.text(n_confounds + n_drifts/2, time_vec[-1] + 5, f'Drift\n({n_drifts})', 
            ha='center', va='top', fontsize=5)
    ax.text(n_confounds + n_drifts + n_task/2, time_vec[-1] + 5, f'Task\n({n_task})', 
            ha='center', va='top', fontsize=5)
    
    ax.set_xlabel("Regressor index", fontsize=7)
    ax.set_ylabel("Time (s)", fontsize=7)
    ax.set_title(f"Design matrix - {run_name}", fontsize=8)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label('Normalized', fontsize=6)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / f"design_matrix_{run_name}.png", dpi=600,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"design_matrix_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved design matrix figure")
    plt.close(fig)


def identify_flat_voxels(Y, threshold=1e-6):
    """Identify voxels with flat/no signal (zero variance over time)."""
    variance = np.var(Y, axis=0)
    flat_mask = variance < threshold  # True = flat voxel (to exclude)
    return flat_mask, variance


def plot_voxel_quality_pie(flat_mask, output_dir, run_name):
    """Plot pie chart showing percentage of valid vs. flat voxels."""
    n_total = len(flat_mask)
    n_flat = np.sum(flat_mask)
    n_valid = n_total - n_flat
    
    pct_flat = 100 * n_flat / n_total
    pct_valid = 100 * n_valid / n_total
    
    # Nature standard: single column
    width_mm = 60
    height_mm = 60
    fig, ax = plt.subplots(figsize=(width_mm/25.4, height_mm/25.4))
    
    # Pie chart - grayscale only
    sizes = [n_valid, n_flat]
    labels = [f'Valid\n({n_valid}, {pct_valid:.1f}%)', 
              f'Flat signal\n({n_flat}, {pct_flat:.1f}%)']
    colors = ['white', 'black']
    explode = (0, 0.05)
    
    wedges, texts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                           startangle=90, wedgeprops=dict(linewidth=0.5, edgecolor='black'))
    
    for text in texts:
        text.set_fontsize(5)
    
    ax.set_title(f'Voxel quality - {run_name}', fontsize=7, pad=5)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / f"voxel_quality_pie_{run_name}.png", dpi=600,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"voxel_quality_pie_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved voxel quality pie chart")
    plt.close(fig)
    
    return n_valid, n_flat, pct_valid, pct_flat


def plot_r2_distribution(r2, flat_mask, output_dir, run_name):
    """Plot R² distribution across all voxels, excluding flat-signal voxels."""
    # Filter out flat voxels
    valid_mask = ~flat_mask
    r2_valid = r2[valid_mask]
    
    n_total = len(r2)
    n_valid = len(r2_valid)
    n_excluded = n_total - n_valid
    
    # Nature standard: single column
    width_mm = 89
    height_mm = 60
    fig, ax = plt.subplots(figsize=(width_mm/25.4, height_mm/25.4))
    
    # Histogram - black only (excluding flat voxels)
    n, bins, patches = ax.hist(r2_valid, bins=50, color='black', edgecolor='black', 
                                linewidth=0.3, alpha=0.7)
    
    # Statistics (on valid voxels only)
    mean_r2 = np.mean(r2_valid)
    median_r2 = np.median(r2_valid)
    
    # Add vertical lines for mean and median - black with different line styles
    ax.axvline(mean_r2, color='black', linewidth=0.8, linestyle='--', 
               label=f'Mean = {mean_r2:.3f}')
    ax.axvline(median_r2, color='black', linewidth=0.8, linestyle=':', 
               label=f'Median = {median_r2:.3f}')
    
    ax.set_xlabel("R²", fontsize=7)
    ax.set_ylabel("Number of parcels", fontsize=7)
    ax.set_title(f"GLM goodness of fit - {run_name}\n(excl. {n_excluded} flat-signal voxels)", 
                 fontsize=7, pad=5)
    
    ax.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=5,
              framealpha=0.9)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    # Add text box with statistics
    stats_text = f"N valid = {n_valid}\nMin = {r2_valid.min():.3f}\nMax = {r2_valid.max():.3f}"
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=5,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    
    ax.set_xlim(-0.1, 1.0)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / f"r2_distribution_{run_name}.png", dpi=600,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"r2_distribution_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved R² distribution figure (excluding flat voxels)")
    plt.close(fig)
    
    return mean_r2, median_r2


def main():
    """Generate all GLM sanity check figures."""
    global BETA_ROOT, OUTPUT_DIR
    args = get_args()
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard("plot_glm_sanity_check")
    BETA_ROOT = paths.glm_dir() / SUBJ
    OUTPUT_DIR = paths.output_dir("figures", "glm_sanity_figures")
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find first available run
    print(f"Looking for GLM data for {SUBJ}...")
    h5_path = find_first_run(SUBJ)
    print(f"Found: {h5_path}")
    
    # Extract run name for labeling
    run_name = h5_path.stem.replace("_betas", "")
    
    # Load GLM data
    print("Loading GLM outputs...")
    glm_data = load_glm_data(h5_path)
    
    # Load original BOLD data
    print("Loading original BOLD data...")
    Y, ts_path = load_bold_data(h5_path)
    print(f"BOLD data shape: {Y.shape} (T x P)")
    
    # Verify dimensions match
    T_glm = glm_data["design_matrix"].shape[0]
    if Y.shape[0] != T_glm:
        print(f"Warning: BOLD T={Y.shape[0]} vs Design T={T_glm}, truncating...")
        Y = Y[:T_glm, :]
    
    # Compute fitted values if not in file
    yhat = glm_data["yhat"]
    if yhat.shape != Y.shape:
        print("Computing fitted values from design matrix and betas...")
        yhat = glm_data["design_matrix"] @ glm_data["betas_full"].T
    
    # Create time vector
    time_vec = np.arange(Y.shape[0]) * TR
    
    # Select random voxels for visualization
    np.random.seed(42)  # Reproducibility
    n_parcels = Y.shape[1]
    voxel_indices = np.random.choice(n_parcels, size=min(N_VOXELS_SAMPLE, n_parcels), replace=False)
    voxel_indices = np.sort(voxel_indices)
    print(f"Selected parcels for visualization: {voxel_indices}")
    
    # Decompose design matrix
    print("Decomposing design matrix...")
    decomposed = decompose_design_matrix(
        glm_data["design_matrix"], 
        glm_data["col_names"],
        glm_data["task_col_start"]
    )
    print(f"  Confounds: {decomposed['confounds'].shape[1]} regressors")
    print(f"  Drifts: {decomposed['drifts'].shape[1]} regressors")
    print(f"  Task: {decomposed['task'].shape[1]} regressors")
    
    # Identify flat-signal voxels (no variance over time)
    print("Identifying flat-signal voxels...")
    flat_mask, variance = identify_flat_voxels(Y)
    n_flat = np.sum(flat_mask)
    print(f"  Found {n_flat} flat-signal voxels ({100*n_flat/len(flat_mask):.2f}%)")
    
    # Compute R²
    print("Computing R² for all parcels...")
    r2 = compute_r2(Y, yhat)
    print(f"  Mean R² (all) = {r2.mean():.4f}")
    print(f"  Mean R² (valid only) = {r2[~flat_mask].mean():.4f}")
    print(f"  Median R² (valid only) = {np.median(r2[~flat_mask]):.4f}")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # 1. Raw BOLD signals (without fit)
    plot_raw_bold_signals(Y, time_vec, voxel_indices, OUTPUT_DIR, run_name)
    
    # 2. BOLD signals with GLM fit overlay
    plot_bold_signals(Y, yhat, time_vec, voxel_indices, OUTPUT_DIR, run_name)
    
    # 3. Design matrix decomposition
    plot_design_matrix(decomposed, time_vec, OUTPUT_DIR, run_name)
    
    # 4. Voxel quality pie chart
    plot_voxel_quality_pie(flat_mask, OUTPUT_DIR, run_name)
    
    # 5. R² distribution (excluding flat voxels)
    plot_r2_distribution(r2, flat_mask, OUTPUT_DIR, run_name)
    
    # Save valid voxel mask (1 = valid, 0 = flat/excluded)
    valid_mask = (~flat_mask).astype(np.int8)
    mask_path = OUTPUT_DIR / f"valid_voxel_mask_{run_name}.npy"
    np.save(mask_path, valid_mask)
    print(f"Saved valid voxel mask to {mask_path}")
    
    print(f"\n✅ All figures saved to: {OUTPUT_DIR}")
    print(f"   - raw_bold_signals_{run_name}.png/pdf")
    print(f"   - bold_signals_{run_name}.png/pdf")
    print(f"   - design_matrix_{run_name}.png/pdf")
    print(f"   - voxel_quality_pie_{run_name}.png/pdf")
    print(f"   - r2_distribution_{run_name}.png/pdf")
    print(f"   - valid_voxel_mask_{run_name}.npy")


if __name__ == "__main__":
    main()
