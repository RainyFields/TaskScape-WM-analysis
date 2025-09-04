#!/usr/bin/env python3
"""
GLM Model Evaluation Script (vertex-level; Glasser surf dtseries)

Features:
- Consistency check: saved task betas vs OLS-recomputed task betas (per sampled vertex)
- R² distribution over:
    (a) sampled vertices (for plots & fast stats)
    (b) a reduced "all-vertices" subset controlled by --max_vertices_r2 (for global distribution)
- Side-by-side panels per interesting vertex: Pred vs Actual and Residuals, with R² in title
- Interesting vertices = top-3 by R² + 3 random remaining (up to 6 total)

Usage example:
python glm_evaluation.py \
  --subj sub-01 --task ctxdm --ses ses-001 --run run-01 \
  --glm_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas \
  --fmri_root /project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_delay \
  --max_vertices 5000 \
  --max_vertices_r2 20000 \
  --seed 123
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


# -----------------------
# CLI
# -----------------------
def get_args():
    p = argparse.ArgumentParser(description="Evaluate GLM fitting quality at vertex level (Glasser dtseries).")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--task", default="ctxdm")
    p.add_argument("--ses", default="ses-001")
    p.add_argument("--run", default="run-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1)
    p.add_argument("--glm_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas")
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled")
    p.add_argument("--output_dir", default="/project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/glm_fitting_check_results/customized_glm")
    p.add_argument("--max_vertices", type=int, default=5000,
                   help="Randomly sample up to this many vertices for plots and per-vertex diagnostics.")
    p.add_argument("--seed", type=int, default=0)

    # R² over a reduced 'all-vertices' subset
    p.add_argument("--compute_all_r2", action="store_true", default=True,
                   help="Compute R² distribution over a subset of all vertices (controlled by --max_vertices_r2).")
    p.add_argument("--max_vertices_r2", type=int, default=20000,
                   help="Limit the number of vertices for the 'all-vertices' R² distribution. "
                        "If <=0, skip. If None, use all nonzero-variance vertices (may be slow).")
    p.add_argument("--chunk_size", type=int, default=5000,
                   help="Chunk size for OLS and predictions when computing R² distribution to limit memory.")
    return p.parse_args()


# -----------------------
# I/O helpers
# -----------------------
def load_glm_data(glm_file):
    data = {}
    with h5py.File(glm_file, 'r') as f:
        # NEW: prefer full betas if present
        data['has_betas_full'] = 'betas_full' in f
        if data['has_betas_full']:
            data['betas_full'] = f['betas_full'][:]  # (V x K)

        data['betas'] = f['betas'][:]                 # (V x K_task)
        data['design_matrix'] = f['design_matrix'][:] # (T_kept x K)
        data['sigma2'] = f['sigma2'][:]               # (V,)
        data['XtX_inv'] = f['XtX_inv'][:]             # (K x K)
        data['design_col_names'] = [n.decode('utf-8') if isinstance(n, bytes) else n
                                    for n in f['design_col_names'][:]]
        data['task_regressor_names'] = [n.decode('utf-8') if isinstance(n, bytes) else n
                                        for n in f.attrs['task_regressor_names']]
        data['task_col_start'] = int(f.attrs['task_col_start'])
        data['task_col_end'] = int(f.attrs['task_col_end'])
        data['dof'] = int(f.attrs['dof'])
        data['rho_ar1'] = float(f.attrs['rho_ar1'])
        data['tmask_dropped'] = int(f.attrs['tmask_dropped'])
    return data


def load_fmri_data(fmri_file, tmask):
    Y = nib.load(str(fmri_file)).get_fdata(dtype=np.float32)  # (T x V)
    keep = np.ones((Y.shape[0],), dtype=bool)
    keep[:tmask] = False
    return Y[keep, :]  # (T_kept x V)


# -----------------------
# Core computations
# -----------------------
def compute_predictions(X, betas_selected):
    return X @ betas_selected.T  # (T_kept x V_sel)


def compute_residuals(Y_actual, Y_pred):
    return Y_actual - Y_pred


def compute_r2_scores(Y_actual, Y_pred):
    V = Y_actual.shape[1]
    r2 = np.empty(V, dtype=np.float32)
    for v in range(V):
        try:
            r2[v] = r2_score(Y_actual[:, v], Y_pred[:, v])
        except Exception:
            r2[v] = np.nan
    return r2


# -----------------------
# Plotting
# -----------------------
def plot_predictions_vs_actual(
    Y_actual, Y_pred, vertex_list, output_dir, subj, task, ses, run,
    intercept_betas=None  # NEW
):
    n = min(len(vertex_list), 6)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n))
    if n == 1:
        axes = [axes]
    t = np.arange(Y_actual.shape[0])
    for i in range(n):
        v = vertex_list[i]  # local index within sampled vertices
        axes[i].plot(t, Y_actual[:, v], label='Actual', alpha=0.85)
        axes[i].plot(t, Y_pred[:, v], label='Predicted', alpha=0.85)

        # NEW: include β₀ in the title if provided
        if intercept_betas is not None:
            axes[i].set_title(f'Vertex {v}: Actual vs Predicted · β₀={float(intercept_betas[v]):.3f}')
        else:
            axes[i].set_title(f'Vertex {v}: Actual vs Predicted')

        axes[i].set_xlabel('Time (frames after tmask)')
        axes[i].set_ylabel('Signal')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_pred_vs_actual.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out

def plot_residual_analysis(residuals, vertex_list, output_dir, subj, task, ses, run):
    n = min(3, len(vertex_list))
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)
    t = np.arange(residuals.shape[0])
    for i in range(n):
        v = vertex_list[i]
        # residual trace
        axes[0, i].plot(t, residuals[:, v], alpha=0.9)
        axes[0, i].set_title(f'Vertex {v}: Residuals')
        axes[0, i].set_xlabel('Time (frames after tmask)')
        axes[0, i].set_ylabel('Residual')
        axes[0, i].grid(True, alpha=0.3)
        # histogram + normal fit
        vals = residuals[:, v]
        axes[1, i].hist(vals, bins=30, alpha=0.7, density=True)
        mu, sd = np.mean(vals), np.std(vals)
        x = np.linspace(vals.min(), vals.max(), 100)
        y = stats.norm.pdf(x, mu, sd) if sd > 0 else np.zeros_like(x)
        axes[1, i].plot(x, y, 'r--', label='Normal fit')
        axes[1, i].set_title(f'Vertex {v}: Residual Distribution')
        axes[1, i].set_xlabel('Residual')
        axes[1, i].set_ylabel('Density')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_residuals.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def plot_r2_distribution(r2_scores, output_dir, subj, task, ses, run, tag="sampled"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    finite = r2_scores[~np.isnan(r2_scores)]
    axes[0].hist(finite, bins=50, alpha=0.8, density=True)
    if finite.size > 0:
        axes[0].axvline(np.nanmean(finite), linestyle='--', label=f'Mean: {np.nanmean(finite):.3f}')
        axes[0].axvline(np.nanmedian(finite), linestyle='--', label=f'Median: {np.nanmedian(finite):.3f}')
    axes[0].set_xlabel('R²')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'R² Distribution ({tag})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(finite, vert=True)
    axes[1].set_ylabel('R²')
    axes[1].set_title(f'R² Box Plot ({tag})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_r2_distribution_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def plot_vertex_panels(
    Y_actual, Y_pred, residuals, r2_scores, vertex_list, output_dir, subj, task, ses, run,
    intercept_betas=None  # NEW
):
    n = len(vertex_list)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2*n))
    if n == 1:
        axes = np.array(axes).reshape(1, 2)

    t = np.arange(Y_actual.shape[0])
    for i, v in enumerate(vertex_list):  # v is local index
        # left: pred vs actual
        axL = axes[i, 0]
        axL.plot(t, Y_actual[:, v], label='Actual', alpha=0.85)
        axL.plot(t, Y_pred[:, v], label='Predicted', alpha=0.85)
        axL.set_xlabel('Time (frames after tmask)')
        axL.set_ylabel('Signal')
        axL.grid(True, alpha=0.3)
        axL.legend()

        # right: residuals
        axR = axes[i, 1]
        axR.plot(t, residuals[:, v], alpha=0.9)
        axR.set_xlabel('Time (frames after tmask)')
        axR.set_ylabel('Residual')
        axR.grid(True, alpha=0.3)

        r2_val = r2_scores[v] if (0 <= v < len(r2_scores)) else np.nan
        if intercept_betas is not None:
            axL.set_title(f'Vertex {v} · R²={r2_val:.3f} · β₀={float(intercept_betas[v]):.3f}')
        else:
            axL.set_title(f'Vertex {v} · R²={r2_val:.3f}')
        axR.set_title(f'Vertex {v} · Residuals')

    plt.tight_layout()
    out = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_vertex_panels.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out



def plot_task_beta_diff_hist(diffs, output_dir, subj, task, ses, run):
    plt.figure(figsize=(6, 4))
    plt.hist(diffs.flatten(), bins=50, density=True, alpha=0.85)
    plt.xlabel('|Δ beta_task| (recomputed - saved)')
    plt.ylabel('Density')
    plt.title('Task beta absolute differences (sampled vertices)')
    out = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_task_beta_diff_hist.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


# -----------------------
# Report
# -----------------------
def create_evaluation_report(glm_data, r2_scores_sampled, residuals_sampled, Y_actual_sampled, Y_pred_sampled,
                             output_dir, subj, task, ses, run, sampled_vertices,
                             V_total, r2_scores_all=None, all_eval_count=None, beta_diff_stats=None):
    report = []
    report.append(f"# GLM Evaluation Report (Vertex-level)")
    report.append(f"**Subject:** {subj}  \n**Task:** {task}  \n**Session:** {ses}  \n**Run:** {run}")
    report.append("")
    report.append("## Model Specifications")
    report.append(f"- # regressors (K): {glm_data['design_matrix'].shape[1]}")
    report.append(f"- # task regressors: {len(glm_data['task_regressor_names'])}")
    report.append(f"- # time points after tmask (T_kept): {glm_data['design_matrix'].shape[0]}")
    report.append(f"- Degrees of freedom: {glm_data['dof']}")
    report.append(f"- AR(1) rho: {glm_data['rho_ar1']:.3f}")
    report.append("")
    report.append("## Vertex Sampling")
    report.append(f"- Total vertices available in run: {V_total}")
    report.append(f"- Evaluated (randomly sampled) vertices for plots: {len(sampled_vertices)}")
    report.append("")
    report.append("## Model Fit (Sampled Vertices)")
    report.append(f"- Mean R²: {np.nanmean(r2_scores_sampled):.3f}")
    report.append(f"- Median R²: {np.nanmedian(r2_scores_sampled):.3f}")
    report.append(f"- R² std: {np.nanstd(r2_scores_sampled):.3f}")
    report.append(f"- R² range: [{np.nanmin(r2_scores_sampled):.3f}, {np.nanmax(r2_scores_sampled):.3f}]")
    for thr in (0.0, 0.1, 0.2):
        frac = np.mean(r2_scores_sampled > thr)
        report.append(f"- Vertices with R² > {thr:.1f}: {int(np.sum(r2_scores_sampled > thr))}/{len(r2_scores_sampled)} ({100*frac:.1f}%)")
    report.append("")
    if r2_scores_all is not None:
        report.append("## Model Fit (Global R² distribution; reduced subset)")
        report.append(f"- Evaluated vertices for global R²: {all_eval_count}")
        report.append(f"- Mean R²: {np.nanmean(r2_scores_all):.3f}")
        report.append(f"- Median R²: {np.nanmedian(r2_scores_all):.3f}")
        report.append(f"- R² std: {np.nanstd(r2_scores_all):.3f}")
        report.append(f"- R² range: [{np.nanmin(r2_scores_all):.3f}, {np.nanmax(r2_scores_all):.3f}]")
        for thr in (0.0, 0.1, 0.2):
            frac = np.mean(r2_scores_all > thr)
            report.append(f"- Vertices with R² > {thr:.1f}: {int(np.sum(r2_scores_all > thr))}/{len(r2_scores_all)} ({100*frac:.1f}%)")
        report.append("")
    if beta_diff_stats is not None:
        mad, mx = beta_diff_stats
        report.append("## Task Beta Consistency (Saved vs Recomputed)")
        report.append(f"- Mean |Δ| over sampled vertices: {mad:.6g}")
        report.append(f"- Max  |Δ| over sampled vertices: {mx:.6g}")
        report.append("")
    report_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    return report_file


# -----------------------
# R² for a subset of all vertices (chunked)
# -----------------------
def compute_r2_subset_vertices(X, XtX_inv, Y_sub, chunk_size):
    """
    Computes R² for a subset of vertices Y_sub (T x V_sub) using OLS betas:
        betas = (X^T X)^{-1} X^T Y_chunk
        Yhat  = X betas
        R² per vertex
    Chunked over vertices to control memory.
    """
    T, V = Y_sub.shape
    r2_out = np.full(V, np.nan, dtype=np.float32)
    # Pre-filter constant columns inside subset (should be rare after global filtering)
    idx_nonzero = np.where(np.std(Y_sub, axis=0) > 0)[0]
    if idx_nonzero.size == 0:
        return r2_out
    for start in range(0, len(idx_nonzero), chunk_size):
        sel = idx_nonzero[start:start+chunk_size]
        Yc = Y_sub[:, sel]                 # (T x Vc)
        XtY = X.T @ Yc                     # (K x Vc)
        betas = XtX_inv @ XtY              # (K x Vc)
        Yhat = X @ betas                   # (T x Vc)
        for j, _ in enumerate(sel):
            try:
                r2_out[idx_nonzero[start + j]] = r2_score(Yc[:, j], Yhat[:, j])
            except Exception:
                r2_out[idx_nonzero[start + j]] = np.nan
    return r2_out


# NEW: R² for a subset using saved full betas (no recompute)
def compute_r2_subset_with_full_betas(X, betas_full, Y_full, idx_eval, chunk_size):
    """
    Computes R² for a subset of vertices using pre-saved full betas:
        Yhat = X @ betas_full[sel,:].T
    """
    r2_out = np.full(len(idx_eval), np.nan, dtype=np.float32)
    for start in range(0, len(idx_eval), chunk_size):
        sel = idx_eval[start:start+chunk_size]
        B = betas_full[sel, :].astype(np.float32)   # (Vc x K)
        Yhat = X @ B.T                               # (T x Vc)
        Yc = Y_full[:, sel]                          # (T x Vc)
        for j in range(len(sel)):
            try:
                r2_out[start + j] = r2_score(Yc[:, j], Yhat[:, j])
            except Exception:
                r2_out[start + j] = np.nan
    return r2_out


# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    np.random.seed(args.seed)

    glm_root = Path(args.glm_root) / args.subj
    fmri_root = Path(args.fmri_root) / args.subj / args.ses
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    glm_file = glm_root / args.ses / "func" / f"{args.subj}_{args.ses}_task-{args.task}_{args.run}_betas.h5"
    fmri_run_format = args.run.replace("run-0", "run-")
    fmri_file = fmri_root / f"{args.subj}_{args.ses}_task-{args.task}_{fmri_run_format}_space-Glasser64k_bold.dtseries.nii"

    if not glm_file.exists():
        raise FileNotFoundError(f"GLM file not found: {glm_file}")
    if not fmri_file.exists():
        raise FileNotFoundError(f"fMRI file not found: {fmri_file}")

    print(f"Loading GLM data from: {glm_file}")
    print(glm_file)
    glm_data = load_glm_data(glm_file)

    # NEW: hard-require betas_full to be present
    if not glm_data.get('has_betas_full', False):
        raise RuntimeError(
            f"`betas_full` dataset not found in H5: {glm_file}\n",
            f"This evaluator requires full betas.\n",
            f> "Re-run glm_analysis.py (saving betas_full) or augment existing files to add betas_full."
        )

    print(f"Loading fMRI data from: {fmri_file}")
    Y_full = load_fmri_data(fmri_file, args.tmask)  # (T_kept x V_total)
    T_kept, V_total = Y_full.shape
    print(f"Time points (after tmask): {T_kept}, total vertices: {V_total}")

    # Sample vertices with nonzero variance for detailed plots
    stds = np.std(Y_full, axis=0)
    nonzero = np.where(stds > 0)[0]
    if len(nonzero) == 0:
        raise RuntimeError("All vertices have zero variance after tmask.")
    V_sel = min(args.max_vertices, len(nonzero))
    sampled_vertices = np.sort(np.random.choice(nonzero, size=V_sel, replace=False))
    print(f"Randomly sampled {len(sampled_vertices)} / {len(nonzero)} non-zero-variance vertices for plots.")

    # Slice to sampled
    Y_actual = Y_full[:, sampled_vertices]  # (T_kept x V_sel)

    # Design
    X = glm_data['design_matrix'].astype(np.float32)  # (T_kept x K)
    K_total = X.shape[1]
    betas_selected = np.zeros((Y_actual.shape[1], K_total), dtype=np.float32)

    # Fill task betas in task columns from saved H5 (for sampled vertices)
    task_start, task_end = glm_data['task_col_start'], glm_data['task_col_end']
    betas_task_saved_sampled = glm_data['betas'][sampled_vertices, :]  # (V_sel x K_task)
    betas_selected[:, task_start:task_end] = betas_task_saved_sampled

    # CHANGED: Use saved FULL betas (error if absent above)
    print("Using betas_full from H5 (no recomputation).")
    betas_full_sampled = glm_data['betas_full'][sampled_vertices, :].astype(np.float32)  # (V_sel x K)
    # Consistency check on task betas (full vs saved task-only)
    abs_diff = np.abs(betas_full_sampled[:, task_start:task_end] - betas_task_saved_sampled)
    mean_abs_diff = float(np.nanmean(abs_diff))
    max_abs_diff = float(np.nanmax(abs_diff))
    print(f"[Beta consistency] mean |Δ| (task, full-vs-saved) = {mean_abs_diff:.6g}, max |Δ| = {max_abs_diff:.6g}")
    diff_fig = plot_task_beta_diff_hist(abs_diff, output_dir, args.subj, args.task, args.ses, args.run)

    betas_selected = betas_full_sampled
    recompute_ok = True  # retained for report formatting

    # --- NEW: grab intercept betas (β₀) for sampled vertices ---
    try:
        const_idx = glm_data['design_col_names'].index('constant')
        intercept_betas_local = betas_selected[:, const_idx]  # (V_sel,)
    except ValueError:
        const_idx = None
        intercept_betas_local = None


    # Predictions / residuals / R² for sampled vertices
    print("Computing predictions (sampled)...")
    Y_pred = compute_predictions(X, betas_selected)
    print("Computing residuals and R² (sampled)...")
    residuals = compute_residuals(Y_actual, Y_pred)
    r2_scores_sampled = compute_r2_scores(Y_actual, Y_pred)  # (V_sel,)

    # Interesting vertices: top-3 R² + 3 random others (keep BOTH)
    valid_indices = np.where(np.isfinite(r2_scores_sampled))[0]
    if len(valid_indices) == 0:
        interesting_local_idx = np.arange(min(6, Y_actual.shape[1]))
    else:
        top_n = min(3, len(valid_indices))
        top_local_idx = valid_indices[np.argsort(r2_scores_sampled[valid_indices])[-top_n:]]
        remaining = np.setdiff1d(np.arange(Y_actual.shape[1]), top_local_idx, assume_unique=False)
        rand_n = min(3, len(remaining))
        rand_local_idx = (np.random.choice(remaining, size=rand_n, replace=False)
                          if rand_n > 0 else np.array([], dtype=int))
        interesting_local_idx = np.unique(np.concatenate([top_local_idx, rand_local_idx]))[:6]

    interesting_global_vertices = sampled_vertices[interesting_local_idx]

    print("Creating visualizations...")
    pred_file = plot_predictions_vs_actual(
            Y_actual, Y_pred, interesting_local_idx, output_dir,
            args.subj, args.task, args.ses, args.run,
            intercept_betas=intercept_betas_local  # NEW
        )

    vertex_panels_file = plot_vertex_panels(
            Y_actual, Y_pred, residuals, r2_scores_sampled, interesting_local_idx,
            output_dir, args.subj, args.task, args.ses, args.run,
            intercept_betas=intercept_betas_local  # NEW
        )

    r2_sampled_fig = plot_r2_distribution(r2_scores_sampled, output_dir,
                                          args.subj, args.task, args.ses, args.run, tag="sampled")

    # Side-by-side panels with R² in title
    vertex_panels_file = plot_vertex_panels(Y_actual, Y_pred, residuals, r2_scores_sampled,
                                            interesting_local_idx, output_dir,
                                            args.subj, args.task, args.ses, args.run)

    # Optional: R² distribution over a reduced subset of ALL vertices (uses betas_full)
    r2_all = None
    all_eval_count = None
    if args.compute_all_r2 and (args.max_vertices_r2 is None or args.max_vertices_r2 > 0):
        print("Computing global R² distribution on a reduced subset...")
        # Choose eval indices among nonzero-variance vertices
        if args.max_vertices_r2 is None:
            idx_eval = nonzero
        else:
            idx_eval = (np.sort(np.random.choice(nonzero,
                                                 size=min(args.max_vertices_r2, len(nonzero)),
                                                 replace=False)))
        all_eval_count = len(idx_eval)
        r2_subset = compute_r2_subset_with_full_betas(X, glm_data['betas_full'],
                                                      Y_full, idx_eval, args.chunk_size)
        r2_all = r2_subset  # length = all_eval_count
        tag = f"all_subsetN{all_eval_count}"
        r2_all_fig = plot_r2_distribution(r2_all, output_dir, args.subj, args.task, args.ses, args.run, tag=tag)
    else:
        r2_all_fig = None

    # Report
    report_file = create_evaluation_report(
        glm_data, r2_scores_sampled, residuals, Y_actual, Y_pred,
        output_dir, args.subj, args.task, args.ses, args.run, sampled_vertices,
        V_total=V_total,
        r2_scores_all=r2_all,
        all_eval_count=all_eval_count,
        beta_diff_stats=(mean_abs_diff, max_abs_diff) if recompute_ok else None
    )

    # Console summary
    print("\n=== Evaluation Summary ===")
    print(f"Sampled vertices for plots: {len(sampled_vertices)} / non-zero-variance: {len(nonzero)}")
    print(f"Sampled R²  -> mean={np.nanmean(r2_scores_sampled):.3f}, median={np.nanmedian(r2_scores_sampled):.3f}, "
          f"range=[{np.nanmin(r2_scores_sampled):.3f}, {np.nanmax(r2_scores_sampled):.3f}]")
    if r2_all is not None:
        print(f"Global (subset N={all_eval_count}) R² -> mean={np.nanmean(r2_all):.3f}, "
              f"median={np.nanmedian(r2_all):.3f}, range=[{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]")
    print(f"Task beta consistency (sampled): mean |Δ|={mean_abs_diff:.6g}, max |Δ|={max_abs_diff:.6g}")
    print("Interesting vertices (global IDs):", interesting_global_vertices.tolist())
    print("Saved:")
    print(" -", pred_file)
    print(" -", vertex_panels_file)
  
    print(" -", r2_sampled_fig)
    if r2_all_fig:
        print(" -", r2_all_fig)
    if diff_fig:
        print(" -", diff_fig)
    print(" -", report_file)


if __name__ == "__main__":
    main()
