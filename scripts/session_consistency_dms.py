"""
Session-to-session consistency analysis for DMS task.
Calculates inter-session correlation of beta patterns for each condition,
then averages across conditions to generate a heatmap.

Uses region-level analysis (360 Glasser regions) for efficiency.
"""

import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================
# Nature-style figure settings
# ============================================================
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


def load_glasser_labels(atlas_path: str):
    """Load Glasser .dlabel.nii labels -> (labels_vector, region_ids_sorted)."""
    img = nib.load(atlas_path)
    lab = np.asarray(img.dataobj).flatten()
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids


def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    """Return a function avg_by_region(features,) -> (n_regions,)."""
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]
    
    def avg_by_region(A):
        """A: (n_features,) -> (n_regions,)"""
        out = np.full(len(region_ids), np.nan, dtype=np.float32)
        for i, idx in enumerate(idxs):
            if len(idx) > 0:
                vals = A[idx]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    out[i] = np.mean(valid)
        return out
    
    return avg_by_region


def load_session_mean_betas(subject, betas_dir, avg_by_region):
    """
    Load session-level MEAN beta patterns (averaged across all encoding trials).
    
    This provides a more stable estimate by averaging all 16 trials per session,
    reducing noise compared to single-trial correlations.
    
    Returns:
        dict: {session: mean_beta_pattern (n_regions,)}
        list: sorted sessions
    """
    session_mean_betas = {}
    subject_dir = betas_dir / subject
    
    # Find all DMS sessions
    sessions = sorted([d.name for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith('ses-')])
    
    for ses in sessions:
        design_path = subject_dir / ses / 'func' / f'{subject}_{ses}_task-dms_run-1_design.csv'
        betas_path = subject_dir / ses / 'func' / f'{subject}_{ses}_task-dms_run-1_betas.h5'
        
        if not design_path.exists() or not betas_path.exists():
            continue
        
        print(f"  Loading {ses}...")
        
        # Load design info
        design_df = pd.read_csv(design_path)
        enc_df = design_df[design_df['type'] == 'encoding'].copy()
        
        # Load betas
        with h5py.File(betas_path, 'r') as f:
            betas = f['betas'][:]  # shape: (n_voxels, n_trials)
        
        # Collect all encoding trial betas for this session
        session_trial_betas = []
        
        for trial_num in enc_df['trialNumber'].unique():
            trial_df = enc_df[enc_df['trialNumber'] == trial_num]
            # Use stim_order == 1 to get the encoding onset (first stimulus)
            s1 = trial_df[trial_df['stim_order'] == 1].iloc[0]
            trial_idx = int(s1['trial_index'])
            beta_voxel = betas[:, trial_idx]
            
            # Convert to region-level
            beta_region = avg_by_region(beta_voxel)
            session_trial_betas.append(beta_region)
        
        # Average across all trials in this session
        session_trial_betas = np.array(session_trial_betas)  # (n_trials, n_regions)
        mean_beta = np.nanmean(session_trial_betas, axis=0)  # (n_regions,)
        
        session_mean_betas[ses] = mean_beta
        print(f"    -> {len(session_trial_betas)} trials averaged")
    
    return session_mean_betas, sessions


def compute_session_correlation_matrix(session_mean_betas, sessions):
    """
    Compute session-to-session correlation matrix from mean session patterns.
    
    Returns:
        np.ndarray: Correlation matrix (n_sessions x n_sessions)
    """
    n_sessions = len(sessions)
    corr_matrix = np.full((n_sessions, n_sessions), np.nan)
    
    for i, ses_i in enumerate(sessions):
        if ses_i not in session_mean_betas:
            continue
        for j, ses_j in enumerate(sessions):
            if ses_j not in session_mean_betas:
                continue
            
            beta_i = session_mean_betas[ses_i]
            beta_j = session_mean_betas[ses_j]
            
            # Remove NaN values
            valid_mask = np.isfinite(beta_i) & np.isfinite(beta_j)
            if valid_mask.sum() > 10:  # Require minimum valid regions
                corr = np.corrcoef(beta_i[valid_mask], beta_j[valid_mask])[0, 1]
                corr_matrix[i, j] = corr
    
    return corr_matrix


def plot_session_heatmap(corr_matrix, sessions, output_path, subject):
    """Generate Nature-style heatmap of session correlations."""
    set_nature_rcparams()
    
    # Figure dimensions (single column width)
    width_mm = 89
    height_mm = 80
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    
    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=True)
    
    # Create heatmap with turbo colormap
    im = ax.imshow(corr_matrix, cmap='turbo', aspect='equal', 
                   vmin=0, vmax=1)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel('Pearson r', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    
    # Session labels (simplified)
    session_labels = [s.replace('ses-', '') for s in sessions]
    
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_yticks(np.arange(len(sessions)))
    ax.set_xticklabels(session_labels, fontsize=6, rotation=45, ha='right')
    ax.set_yticklabels(session_labels, fontsize=6)
    
    ax.set_xlabel('Session', fontsize=7)
    ax.set_ylabel('Session', fontsize=7)
    ax.set_title(f'{subject}: Session-to-session consistency (DMS)', fontsize=8)
    
    # Add correlation values in cells
    if len(sessions) <= 16:
        for i in range(len(sessions)):
            for j in range(len(sessions)):
                val = corr_matrix[i, j]
                if np.isfinite(val):
                    text_color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=4, color=text_color)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Save figures
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(f"{output_path}.pdf", format='pdf')
    fig.savefig(f"{output_path}.png", format='png', dpi=600, facecolor='white')
    plt.close(fig)
    
    print(f"Exported {output_path} (.pdf, .png)")


def main():
    # Configuration
    subjects = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
    betas_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas")
    atlas_path = "/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii"
    output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/session_consistency")
    
    # Load Glasser atlas for region averaging
    print("Loading Glasser atlas...")
    labels_vec, region_ids = load_glasser_labels(atlas_path)
    avg_by_region = make_region_averager(labels_vec, region_ids)
    print(f"Atlas loaded: {len(region_ids)} regions")
    
    # Collect all results for group summary
    all_off_diag = []
    
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Processing {subject}")
        print('='*60)
        
        print(f"Loading betas for {subject}...")
        session_mean_betas, sessions = load_session_mean_betas(subject, betas_dir, avg_by_region)
        
        if len(session_mean_betas) == 0:
            print(f"  No DMS data found for {subject}, skipping...")
            continue
        
        print(f"\nFound {len(session_mean_betas)} sessions with data")
        
        print("Computing session-to-session correlations...")
        corr_matrix = compute_session_correlation_matrix(session_mean_betas, sessions)
        
        # Print summary statistics
        off_diag_mask = ~np.eye(len(sessions), dtype=bool)
        off_diag_values = corr_matrix[off_diag_mask]
        off_diag_values = off_diag_values[np.isfinite(off_diag_values)]
        
        print(f"\nOff-diagonal correlation statistics:")
        print(f"  Mean: {np.mean(off_diag_values):.4f}")
        print(f"  Std:  {np.std(off_diag_values):.4f}")
        print(f"  Min:  {np.min(off_diag_values):.4f}")
        print(f"  Max:  {np.max(off_diag_values):.4f}")
        
        all_off_diag.extend(off_diag_values.tolist())
        
        # Plot heatmap
        output_path = output_dir / f"{subject}_dms_session_consistency_region"
        plot_session_heatmap(corr_matrix, sessions, output_path, subject)
        
        # Save correlation matrix
        np.save(output_dir / f"{subject}_dms_session_corr_matrix_region.npy", corr_matrix)
    
    # Group summary
    print(f"\n{'='*60}")
    print("GROUP SUMMARY (all subjects)")
    print('='*60)
    all_off_diag = np.array(all_off_diag)
    print(f"  Mean: {np.mean(all_off_diag):.4f}")
    print(f"  Std:  {np.std(all_off_diag):.4f}")
    print(f"  Min:  {np.min(all_off_diag):.4f}")
    print(f"  Max:  {np.max(all_off_diag):.4f}")


if __name__ == "__main__":
    main()
