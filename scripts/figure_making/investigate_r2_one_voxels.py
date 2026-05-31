#!/usr/bin/env python3
"""
Investigate Voxels with R² = 1 (Flat-Line Voxels)

This script identifies and visualizes voxels where R² equals or approaches 1,
which typically indicates flat-line signals (no variance) that need to be filtered.

Generates:
1. Time series plots of 4 sample flat-line voxels
2. Comparison with 4 normal voxels
3. Statistics on the ratio of problematic voxels

Usage:
    python investigate_r2_one_voxels.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib

# ============================================================================
# GLOBAL MATPLOTLIB SETTINGS FOR ACADEMIC PUBLICATION
# ============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']

# Font sizes
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 16

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

# PDF/SVG font embedding
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# ============================================================================
# CONFIGURATION
# ============================================================================
SUBJ = "sub-03"
TR = 1.49
TMASK = 1
OUTPUT_DPI = 300
R2_THRESHOLD = 0.999  # Threshold for "suspicious" R² values

# Paths
DATA_ROOT = Path("/project/def-pbellec/xuan/fmri_dataset_project/data")
FMRI_ROOT = DATA_ROOT / "glasser_resampled" / SUBJ
BETA_ROOT = DATA_ROOT / "encoding_delay/trial_level_betas" / SUBJ
OUTPUT_DIR = Path("/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/results/r2_investigation")


def find_first_run(subj: str):
    """Find the first available run for a subject."""
    beta_dir = BETA_ROOT
    sessions = sorted([d for d in beta_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")])
    if not sessions:
        raise FileNotFoundError(f"No sessions found in {beta_dir}")
    
    ses = sessions[0]
    func_dir = ses / "func"
    h5_files = sorted(func_dir.glob("*_betas.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No beta files found in {func_dir}")
    
    return h5_files[0]


def load_glm_data(h5_path: Path):
    """Load GLM outputs from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        betas_full = f["betas_full"][:]
        design_matrix = f["design_matrix"][:]
        col_names = [n.decode() if isinstance(n, bytes) else n for n in f["design_col_names"][:]]
        
        if "yhat" in f:
            yhat = f["yhat"][:]
        else:
            yhat = design_matrix @ betas_full.T
    
    return {
        "betas_full": betas_full,
        "design_matrix": design_matrix,
        "col_names": col_names,
        "yhat": yhat,
    }


def load_bold_data(h5_path: Path):
    """Load original BOLD data corresponding to the GLM run."""
    fname = h5_path.stem.replace("_betas", "")
    parts = fname.split("_")
    subj = parts[0]
    ses = parts[1]
    task = parts[2].replace("task-", "")
    run = parts[3]
    
    ts_path = FMRI_ROOT / ses / f"{subj}_{ses}_task-{task}_{run}_space-Glasser64k_bold.dtseries.nii"
    
    if not ts_path.exists():
        run_num = run.replace("run-", "")
        alt_run = f"run-{int(run_num):02d}" if len(run_num) == 1 else f"run-{int(run_num)}"
        ts_path = FMRI_ROOT / ses / f"{subj}_{ses}_task-{task}_{alt_run}_space-Glasser64k_bold.dtseries.nii"
    
    if not ts_path.exists():
        raise FileNotFoundError(f"Timeseries not found: {ts_path}")
    
    Y = nib.load(str(ts_path)).get_fdata(dtype=np.float32)
    Y = Y[TMASK:, :]
    
    return Y, ts_path


def compute_r2(Y, yhat):
    """Compute R² for each voxel."""
    ss_res = np.sum((Y - yhat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    return r2


def compute_voxel_stats(Y):
    """Compute statistics for each voxel."""
    mean_val = np.mean(Y, axis=0)
    std_val = np.std(Y, axis=0)
    min_val = np.min(Y, axis=0)
    max_val = np.max(Y, axis=0)
    range_val = max_val - min_val
    
    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
    }


def plot_flat_line_voxels(Y, yhat, r2, time_vec, flat_indices, output_dir, run_name):
    """Plot time series for flat-line voxels (R² ≈ 1)."""
    n_samples = min(4, len(flat_indices))
    if n_samples == 0:
        print("No flat-line voxels found!")
        return
    
    # Select 4 sample indices
    sample_indices = flat_indices[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]
    
    for i, (ax, vox_idx) in enumerate(zip(axes, sample_indices)):
        # Plot observed BOLD
        ax.plot(time_vec, Y[:, vox_idx], color='#1f77b4', linewidth=2.5, 
                label='Observed BOLD', alpha=0.9)
        # Plot GLM fit
        ax.plot(time_vec, yhat[:, vox_idx], color='#d62728', linewidth=2.5, 
                label='GLM Fit', linestyle='--', alpha=0.9)
        
        # Statistics
        mean_val = np.mean(Y[:, vox_idx])
        std_val = np.std(Y[:, vox_idx])
        r2_val = r2[vox_idx]
        
        ax.set_ylabel(f"Vertex {vox_idx}", fontsize=14)
        
        # Add stats annotation
        stats_text = f"R² = {r2_val:.6f}\nmean = {mean_val:.2f}\nstd = {std_val:.6f}"
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='red', 
                          alpha=0.9, linewidth=2))
        
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=10, width=2.0)
        ax.tick_params(axis='both', which='minor', length=5, width=1.5)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        if i == 0:
            ax.legend(loc='upper left', frameon=True, edgecolor='black')
        
        ax.set_xlim(time_vec[0], time_vec[-1])
    
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Flat-Line Voxels (R² ≥ {R2_THRESHOLD}) - {run_name}", 
                 fontsize=22, fontweight='bold', y=1.02, color='red')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / f"flat_line_voxels_{run_name}.png", dpi=OUTPUT_DPI,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"flat_line_voxels_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved flat-line voxels figure")
    plt.close(fig)


def plot_normal_voxels(Y, yhat, r2, time_vec, normal_indices, output_dir, run_name):
    """Plot time series for normal voxels (for comparison)."""
    n_samples = min(4, len(normal_indices))
    if n_samples == 0:
        print("No normal voxels found for comparison, skipping...")
        return
    
    sample_indices = normal_indices[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]
    
    for i, (ax, vox_idx) in enumerate(zip(axes, sample_indices)):
        ax.plot(time_vec, Y[:, vox_idx], color='#1f77b4', linewidth=2.5, 
                label='Observed BOLD', alpha=0.9)
        ax.plot(time_vec, yhat[:, vox_idx], color='#2ecc71', linewidth=2.5, 
                label='GLM Fit', linestyle='--', alpha=0.9)
        
        mean_val = np.mean(Y[:, vox_idx])
        std_val = np.std(Y[:, vox_idx])
        r2_val = r2[vox_idx]
        
        ax.set_ylabel(f"Vertex {vox_idx}", fontsize=14)
        
        stats_text = f"R² = {r2_val:.4f}\nmean = {mean_val:.2f}\nstd = {std_val:.2f}"
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', 
                          alpha=0.9, linewidth=2))
        
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=10, width=2.0)
        ax.tick_params(axis='both', which='minor', length=5, width=1.5)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        if i == 0:
            ax.legend(loc='upper left', frameon=True, edgecolor='black')
        
        ax.set_xlim(time_vec[0], time_vec[-1])
    
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Normal Voxels (for comparison) - {run_name}", 
                 fontsize=22, fontweight='bold', y=1.02, color='green')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / f"normal_voxels_{run_name}.png", dpi=OUTPUT_DPI,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"normal_voxels_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved normal voxels figure")
    plt.close(fig)


def plot_r2_distribution_detailed(r2, output_dir, run_name):
    """Plot detailed R² distribution highlighting problematic voxels."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: Full distribution
    ax1 = axes[0]
    n, bins, patches = ax1.hist(r2, bins=100, color='#3498db', edgecolor='#2980b9', 
                                 linewidth=1.5, alpha=0.7)
    
    # Color problematic bins
    for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
        if bin_left >= R2_THRESHOLD:
            patch.set_facecolor('#e74c3c')
            patch.set_edgecolor('#c0392b')
    
    ax1.axvline(R2_THRESHOLD, color='red', linewidth=3, linestyle='--', 
                label=f'Threshold = {R2_THRESHOLD}')
    ax1.axvline(np.mean(r2), color='#2ecc71', linewidth=3, linestyle='-', 
                label=f'Mean = {np.mean(r2):.4f}')
    
    ax1.set_xlabel("R²")
    ax1.set_ylabel("Number of Vertices")
    ax1.set_title("Full R² Distribution", fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, edgecolor='black')
    
    for spine in ax1.spines.values():
        spine.set_linewidth(2.0)
    ax1.tick_params(axis='both', which='major', length=10, width=2.0)
    ax1.minorticks_on()
    
    # Right: Zoomed in on high R² region
    ax2 = axes[1]
    high_r2_mask = r2 >= 0.9
    high_r2 = r2[high_r2_mask]
    
    if len(high_r2) > 0:
        n2, bins2, patches2 = ax2.hist(high_r2, bins=50, color='#e74c3c', 
                                        edgecolor='#c0392b', linewidth=1.5, alpha=0.7)
        ax2.axvline(R2_THRESHOLD, color='darkred', linewidth=3, linestyle='--',
                    label=f'Threshold = {R2_THRESHOLD}')
        ax2.axvline(1.0, color='black', linewidth=2, linestyle=':',
                    label='R² = 1.0')
    
    ax2.set_xlabel("R²")
    ax2.set_ylabel("Number of Vertices")
    ax2.set_title("High R² Region (R² ≥ 0.9)", fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, edgecolor='black')
    ax2.set_xlim(0.9, 1.01)
    
    for spine in ax2.spines.values():
        spine.set_linewidth(2.0)
    ax2.tick_params(axis='both', which='major', length=10, width=2.0)
    ax2.minorticks_on()
    
    # Add statistics text box
    n_total = len(r2)
    n_problematic = np.sum(r2 >= R2_THRESHOLD)
    n_exact_one = np.sum(r2 >= 0.9999)
    ratio_problematic = n_problematic / n_total * 100
    
    stats_text = (f"Total vertices: {n_total:,}\n"
                  f"R² ≥ {R2_THRESHOLD}: {n_problematic:,} ({ratio_problematic:.2f}%)\n"
                  f"R² ≥ 0.9999: {n_exact_one:,} ({n_exact_one/n_total*100:.2f}%)")
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', 
                       linewidth=2, alpha=0.9))
    
    fig.suptitle(f"R² Distribution Analysis - {run_name}", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    
    fig.savefig(output_dir / f"r2_distribution_detailed_{run_name}.png", dpi=OUTPUT_DPI,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"r2_distribution_detailed_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved detailed R² distribution figure")
    plt.close(fig)


def plot_std_vs_r2(r2, voxel_stats, output_dir, run_name):
    """Plot relationship between voxel std and R²."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    std_vals = voxel_stats["std"]
    
    # Color by R² threshold
    colors = np.where(r2 >= R2_THRESHOLD, '#e74c3c', '#3498db')
    alphas = np.where(r2 >= R2_THRESHOLD, 0.9, 0.3)
    
    # Plot normal voxels first (behind)
    normal_mask = r2 < R2_THRESHOLD
    ax.scatter(std_vals[normal_mask], r2[normal_mask], c='#3498db', alpha=0.3, 
               s=10, label=f'Normal (R² < {R2_THRESHOLD})')
    
    # Plot problematic voxels on top
    problem_mask = r2 >= R2_THRESHOLD
    ax.scatter(std_vals[problem_mask], r2[problem_mask], c='#e74c3c', alpha=0.9, 
               s=50, marker='x', linewidths=2, label=f'Problematic (R² ≥ {R2_THRESHOLD})')
    
    ax.axhline(R2_THRESHOLD, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_xlabel("Signal Standard Deviation")
    ax.set_ylabel("R²")
    ax.set_title(f"Signal Variance vs R² - {run_name}", fontweight='bold')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis='both', which='major', length=10, width=2.0)
    ax.minorticks_on()
    
    # Add text annotation
    n_low_std = np.sum(std_vals < 1.0)
    n_problem_low_std = np.sum((r2 >= R2_THRESHOLD) & (std_vals < 1.0))
    
    info_text = (f"Low std (<1.0): {n_low_std:,}\n"
                 f"Problematic & low std: {n_problem_low_std:,}")
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=14,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    fig.savefig(output_dir / f"std_vs_r2_{run_name}.png", dpi=OUTPUT_DPI,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"std_vs_r2_{run_name}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"Saved std vs R² figure")
    plt.close(fig)


def main():
    """Main function to investigate R² = 1 voxels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("INVESTIGATING VOXELS WITH R² = 1 (FLAT-LINE VOXELS)")
    print("=" * 70)
    
    # Find and load data
    print(f"\nLooking for GLM data for {SUBJ}...")
    h5_path = find_first_run(SUBJ)
    print(f"Found: {h5_path}")
    
    run_name = h5_path.stem.replace("_betas", "")
    
    print("\nLoading GLM outputs...")
    glm_data = load_glm_data(h5_path)
    
    print("Loading original BOLD data...")
    Y, ts_path = load_bold_data(h5_path)
    print(f"BOLD data shape: {Y.shape} (T x P)")
    
    # Ensure dimensions match
    T_glm = glm_data["design_matrix"].shape[0]
    if Y.shape[0] != T_glm:
        Y = Y[:T_glm, :]
    
    yhat = glm_data["yhat"]
    if yhat.shape != Y.shape:
        yhat = glm_data["design_matrix"] @ glm_data["betas_full"].T
    
    time_vec = np.arange(Y.shape[0]) * TR
    
    # Compute R² and voxel statistics
    print("\nComputing R² and voxel statistics...")
    r2 = compute_r2(Y, yhat)
    voxel_stats = compute_voxel_stats(Y)
    
    # Identify problematic voxels
    flat_mask = r2 >= R2_THRESHOLD
    flat_indices = np.where(flat_mask)[0]
    n_flat = len(flat_indices)
    n_total = len(r2)
    
    # Also identify voxels with very low variance
    low_std_mask = voxel_stats["std"] < 1.0
    n_low_std = np.sum(low_std_mask)
    
    # Combined: flat AND low std
    combined_mask = flat_mask | low_std_mask
    n_combined = np.sum(combined_mask)
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total vertices:                    {n_total:,}")
    print(f"Vertices with R² ≥ {R2_THRESHOLD}:        {n_flat:,} ({n_flat/n_total*100:.4f}%)")
    print(f"Vertices with R² ≥ 0.9999:         {np.sum(r2 >= 0.9999):,} ({np.sum(r2 >= 0.9999)/n_total*100:.4f}%)")
    print(f"Vertices with R² = 1.0 (exact):    {np.sum(r2 == 1.0):,} ({np.sum(r2 == 1.0)/n_total*100:.4f}%)")
    print(f"Vertices with std < 1.0:           {n_low_std:,} ({n_low_std/n_total*100:.4f}%)")
    print(f"Vertices with std < 0.1:           {np.sum(voxel_stats['std'] < 0.1):,}")
    print(f"Problematic (R²≥{R2_THRESHOLD} OR std<1): {n_combined:,} ({n_combined/n_total*100:.4f}%)")
    print("=" * 70)
    
    # Show some example flat voxels
    if n_flat > 0:
        print(f"\nExample flat-line voxels (first 10):")
        for i, idx in enumerate(flat_indices[:10]):
            print(f"  Vertex {idx}: R²={r2[idx]:.6f}, std={voxel_stats['std'][idx]:.6f}, "
                  f"mean={voxel_stats['mean'][idx]:.2f}, range={voxel_stats['range'][idx]:.6f}")
    
    # Identify normal voxels for comparison (R² between 0.1 and 0.7, with variance)
    normal_mask = (r2 >= 0.1) & (r2 <= 0.7) & (voxel_stats["std"] > 1.0)
    normal_indices = np.where(normal_mask)[0]
    
    # If still no voxels, relax further
    if len(normal_indices) == 0:
        normal_mask = (r2 < R2_THRESHOLD) & (voxel_stats["std"] > 0.1)
        normal_indices = np.where(normal_mask)[0]
    
    # Random sample for normal voxels
    np.random.seed(42)
    if len(normal_indices) > 4:
        normal_indices = np.random.choice(normal_indices, size=4, replace=False)
    
    print(f"\nNormal voxels for comparison ({len(normal_indices)} found):")
    for idx in normal_indices:
        print(f"  Vertex {idx}: R²={r2[idx]:.4f}, std={voxel_stats['std'][idx]:.2f}")
    
    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    # 1. Flat-line voxels
    print("\n1. Plotting flat-line voxels...")
    plot_flat_line_voxels(Y, yhat, r2, time_vec, flat_indices, OUTPUT_DIR, run_name)
    
    # 2. Normal voxels for comparison
    print("\n2. Plotting normal voxels for comparison...")
    plot_normal_voxels(Y, yhat, r2, time_vec, normal_indices, OUTPUT_DIR, run_name)
    
    # 3. Detailed R² distribution
    print("\n3. Plotting detailed R² distribution...")
    plot_r2_distribution_detailed(r2, OUTPUT_DIR, run_name)
    
    # 4. Std vs R² scatter plot
    print("\n4. Plotting std vs R² relationship...")
    plot_std_vs_r2(r2, voxel_stats, OUTPUT_DIR, run_name)
    
    print("\n" + "=" * 70)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Return statistics for further use
    return {
        "n_total": n_total,
        "n_flat": n_flat,
        "ratio_flat": n_flat / n_total,
        "flat_indices": flat_indices,
        "r2": r2,
        "voxel_stats": voxel_stats,
    }


if __name__ == "__main__":
    results = main()
