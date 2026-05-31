"""
Split-half consistency analysis for DMS task.
For each condition, split 16 repetitions into halves, correlate patterns,
and generate a boxplot showing consistency across conditions.
"""

import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import combinations
import random

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


def get_condition_label(s1_loc, s1_obj, s2_loc, s2_obj):
    """Generate condition label from stimulus properties."""
    obj1_id = int(s1_obj) % 4
    obj2_id = int(s2_obj) % 4
    return f"loc{int(s1_loc)}_obj{obj1_id}*loc{int(s2_loc)}_obj{obj2_id}"


def load_condition_betas(subject, betas_dir, avg_by_region):
    """
    Load all DMS betas organized by condition.
    
    Returns:
        dict: {condition: list of (n_regions,) beta patterns across sessions}
    """
    condition_betas = defaultdict(list)
    subject_dir = betas_dir / subject
    
    sessions = sorted([d.name for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith('ses-')])
    
    for ses in sessions:
        design_path = subject_dir / ses / 'func' / f'{subject}_{ses}_task-dms_run-1_design.csv'
        betas_path = subject_dir / ses / 'func' / f'{subject}_{ses}_task-dms_run-1_betas.h5'
        
        if not design_path.exists() or not betas_path.exists():
            continue
        
        design_df = pd.read_csv(design_path)
        enc_df = design_df[design_df['type'] == 'encoding'].copy()
        
        with h5py.File(betas_path, 'r') as f:
            betas = f['betas'][:]
        
        for trial_num in enc_df['trialNumber'].unique():
            trial_df = enc_df[enc_df['trialNumber'] == trial_num]
            s1 = trial_df[trial_df['stim_order'] == 1].iloc[0]
            s2 = trial_df[trial_df['stim_order'] == 2].iloc[0]
            
            condition = get_condition_label(
                s1['location'], s1['object'],
                s2['location'], s2['object']
            )
            
            trial_idx = int(s1['trial_index'])
            beta_voxel = betas[:, trial_idx]
            beta_region = avg_by_region(beta_voxel)
            
            condition_betas[condition].append(beta_region)
    
    return condition_betas


def compute_split_half_correlations(condition_betas, n_splits=100, seed=42):
    """
    Compute split-half correlations for each condition.
    
    For each condition with N repetitions, randomly split into two halves,
    average each half, and correlate. Repeat n_splits times.
    
    Returns:
        dict: {condition: list of correlation values}
        list: chance level correlations (from shuffled region labels)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    condition_correlations = {}
    chance_correlations = []
    
    for condition, betas_list in condition_betas.items():
        n_reps = len(betas_list)
        betas_array = np.array(betas_list)  # (n_reps, n_regions)
        
        correlations = []
        
        for split_i in range(n_splits):
            # Random split into two halves
            indices = list(range(n_reps))
            random.shuffle(indices)
            half1_idx = indices[:n_reps // 2]
            half2_idx = indices[n_reps // 2:]
            
            # Average each half
            half1_mean = np.nanmean(betas_array[half1_idx, :], axis=0)
            half2_mean = np.nanmean(betas_array[half2_idx, :], axis=0)
            
            # Correlate
            valid_mask = np.isfinite(half1_mean) & np.isfinite(half2_mean)
            if valid_mask.sum() > 10:
                corr = np.corrcoef(half1_mean[valid_mask], half2_mean[valid_mask])[0, 1]
                correlations.append(corr)
                
                # Compute chance level by shuffling region labels
                shuffled_half2 = half2_mean[valid_mask].copy()
                np.random.shuffle(shuffled_half2)
                chance_corr = np.corrcoef(half1_mean[valid_mask], shuffled_half2)[0, 1]
                chance_correlations.append(chance_corr)
        
        condition_correlations[condition] = correlations
    
    return condition_correlations, chance_correlations


def plot_split_half_boxplot(condition_correlations, chance_correlations, output_path, subject):
    """Generate Nature-style boxplot of split-half correlations with chance level."""
    set_nature_rcparams()
    
    # Figure dimensions (double column width for many conditions)
    width_mm = 183
    height_mm = 80
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    
    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=True)
    
    # Sort conditions for consistent ordering
    conditions = sorted(condition_correlations.keys())
    data = [condition_correlations[c] for c in conditions]
    
    # Simplified labels (shorter)
    labels = [c.replace('loc', 'L').replace('obj', 'O').replace('*', '\n') for c in conditions]
    
    # Create boxplot
    bp = ax.boxplot(data, positions=range(len(conditions)), widths=0.6,
                    patch_artist=True, showfliers=True)
    
    # Style the boxplot
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(conditions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(0.5)
    
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(0.5)
            item.set_color('black')
    
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=2, markerfacecolor='gray', 
                  markeredgecolor='none', alpha=0.5)
    
    # Compute chance level statistics
    chance_mean = np.mean(chance_correlations)
    chance_std = np.std(chance_correlations)
    chance_95 = np.percentile(chance_correlations, 97.5)
    
    # Add chance level shading (mean ± 2 std, approximately 95% CI)
    ax.axhspan(chance_mean - 2*chance_std, chance_mean + 2*chance_std, 
               color='gray', alpha=0.2, label=f'Chance 95% CI')
    ax.axhline(y=chance_mean, color='gray', linestyle=':', linewidth=0.8,
               label=f'Chance mean: {chance_mean:.3f}')
    
    # Add overall mean line
    overall_mean = np.mean([v for vals in data for v in vals])
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=0.8, 
               label=f'Overall mean: {overall_mean:.3f}')
    
    # Labels and title
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha='right')
    ax.set_xlabel('Task condition', fontsize=7)
    ax.set_ylabel('Split-half correlation (r)', fontsize=7)
    ax.set_title(f'{subject}: DMS split-half consistency by condition', fontsize=8)
    
    # Set y-axis limits
    ax.set_ylim(-0.3, 1.0)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=5, frameon=False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(f"{output_path}.pdf", format='pdf')
    fig.savefig(f"{output_path}.png", format='png', dpi=600, facecolor='white')
    plt.close(fig)
    
    print(f"Exported {output_path} (.pdf, .png)")
    
    return overall_mean, chance_mean


def main():
    # Configuration
    subjects = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
    betas_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas")
    atlas_path = "/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii"
    output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/split_half_consistency")
    n_splits = 100
    
    # Load Glasser atlas
    print("Loading Glasser atlas...")
    labels_vec, region_ids = load_glasser_labels(atlas_path)
    avg_by_region = make_region_averager(labels_vec, region_ids)
    print(f"Atlas loaded: {len(region_ids)} regions")
    
    all_results = {}
    
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Processing {subject}")
        print('='*60)
        
        print("Loading condition betas...")
        condition_betas = load_condition_betas(subject, betas_dir, avg_by_region)
        
        if len(condition_betas) == 0:
            print(f"  No DMS data found for {subject}, skipping...")
            continue
        
        print(f"Found {len(condition_betas)} conditions")
        for cond, betas in condition_betas.items():
            print(f"  {cond}: {len(betas)} repetitions")
        
        print(f"\nComputing split-half correlations ({n_splits} splits)...")
        condition_correlations, chance_correlations = compute_split_half_correlations(
            condition_betas, n_splits=n_splits
        )
        
        # Print summary
        print("\nSplit-half correlation statistics by condition:")
        for cond in sorted(condition_correlations.keys()):
            corrs = condition_correlations[cond]
            print(f"  {cond}: mean={np.mean(corrs):.3f}, std={np.std(corrs):.3f}")
        
        chance_mean = np.mean(chance_correlations)
        print(f"\nChance level: mean={chance_mean:.3f}, std={np.std(chance_correlations):.3f}")
        
        # Plot
        output_path = output_dir / f"{subject}_dms_split_half_boxplot"
        overall_mean, chance_mean = plot_split_half_boxplot(
            condition_correlations, chance_correlations, output_path, subject
        )
        
        all_results[subject] = {
            'condition_correlations': condition_correlations,
            'overall_mean': overall_mean,
            'chance_mean': chance_mean
        }
        
        # Save data
        np.savez(
            output_dir / f"{subject}_dms_split_half_data.npz",
            **{cond: np.array(corrs) for cond, corrs in condition_correlations.items()}
        )
    
    # Group summary
    print(f"\n{'='*60}")
    print("GROUP SUMMARY")
    print('='*60)
    for subj, results in all_results.items():
        print(f"  {subj}: overall mean r = {results['overall_mean']:.3f}, chance = {results['chance_mean']:.3f}")
    
    group_mean = np.mean([r['overall_mean'] for r in all_results.values()])
    group_chance = np.mean([r['chance_mean'] for r in all_results.values()])
    print(f"\n  Group average: {group_mean:.3f}")
    print(f"  Group chance:  {group_chance:.3f}")


if __name__ == "__main__":
    main()
