#!/usr/bin/env python3
"""
GLM Model Evaluation (vertex-level; Glasser dtseries) — FINAL + per-voxel R² + extreme/non-extreme analysis

Aligned with your final GLM saver. Relies on:
  Datasets:  betas_full (P x K), betas (P x K_task), design_matrix (T_kept x K), design_col_names (K,)
             Optional: yhat (T_kept x P), resid (T_kept x P)
  Attrs:     task_regressor_names, task_col_start, task_col_end,
             tmask_dropped, high_pass_sec, order_in_X, estimator, rho_ar1

Generates:
  1) Pred vs Actual traces for up to 6 interesting vertices (per-panel R² + suptitle R² summary)
  2) Residual panels (trace + histogram) for a few vertices
  3) R² hist + boxplot for the sampled vertices
  4) Global R² over all nonzero-variance vertices (used for extreme/non-extreme analyses)
  5) Histogram of |Δ| between betas_full (task slice) and betas (task-only)
  6) Markdown report summarizing key metrics
  7) R² CSVs: sampled, all nonzero-var, extreme-negative selected, non-extreme set


python sanity_check/glm_evaluation.py  \
        --subj sub-01 \
        --task ctxdm --ses ses-001 --run run-01 \
        --glm_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_decoding/trial_level_betas  \
        --fmri_root /project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled  \
        --output_dir /project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_decoding \
        --max_vertices 1000 --max_vertices_r2 2000 \
        --compute_all_r2 --seed 123
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import stats
from sklearn.metrics import r2_score

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from figure_style.figure_utils import (
    add_panel_labels,
    apply_clean_spines,
    get_figsize,
    get_palette,
    mm_to_inches,
    save_figure,
)

# Load shared publication style (figures4papers-compatible).
_STYLE_PATH = _PROJECT_ROOT / "figure_style" / "nature_style.mplstyle"
plt.style.use(str(_STYLE_PATH))

# -----------------------
# CLI
# -----------------------

def get_args():
    p = argparse.ArgumentParser(description="Evaluate GLM fit (matches final GLM outputs)")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--task", default="ctxdm")
    p.add_argument("--ses", default="ses-001")
    p.add_argument("--run", default="run-01")
    p.add_argument("--tmask", type=int, default=1)
    p.add_argument("--glm_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas")
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled")
    p.add_argument("--output_dir", default="/project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/glm_fitting_check_results/customized_glm")
    p.add_argument("--max_vertices", type=int, default=5000,
                   help="Randomly sample up to this many vertices for plots and per-vertex diagnostics.")
    p.add_argument("--seed", type=int, default=0)

    # Optional global R² subset plot (kept for continuity)
    p.add_argument("--compute_all_r2", action="store_true",
                   help="If set, also plot an R² distribution for a random subset of nonzero-variance vertices.")
    p.add_argument("--max_vertices_r2", type=int, default=20000,
                   help="Upper bound on vertices for the optional global R² subset plot; if <=0, skip. If None, use all nonzero-variance vertices.")
    p.add_argument("--chunk_size", type=int, default=5000,
                   help="Chunk size for global R² computations to limit memory.")

    # Extreme / non-extreme threshold
    p.add_argument("--r2_extreme_thr", type=float, default=-1.0,
                   help="Threshold below which R² counts as 'extremely negative'. Default: -1.0")
    p.add_argument("--journal", default="nature",
                   choices=["nature", "pnas", "cell", "science", "plos", "neuron", "jneurosci"],
                   help="Journal preset for publication figure sizing.")
    p.add_argument("--pred_span", default="double", choices=["single", "1.5", "double"],
                   help="Column span for the 3x2 prediction panel figure.")
    p.add_argument("--dist_span", default="single", choices=["single", "1.5", "double"],
                   help="Column span for the adjusted-R² distribution figure.")
    p.add_argument("--pred_full_width_mm", type=float, default=None,
                   help="Optional explicit total width (mm) for pred-vs-actual 3x2 figure (e.g., 190 for near full A4 width).")
    p.add_argument("--pred_panel_hw_ratio", type=float, default=0.5,
                   help="Per-panel height/width ratio for pred-vs-actual grid. 0.5 means 2:4 (height:width).")
    return p.parse_args()

# -----------------------
# Path helpers (mirror GLM script behavior)
# -----------------------

def run_aliases(run_label: str):
    """Return both unpadded and 2-digit-padded forms, regardless of input."""
    if isinstance(run_label, str) and run_label.startswith('run-') and run_label[4:].isdigit():
        n = int(run_label[4:])
        variants = [f'run-{n}', f'run-{n:02d}']
        seen = set(); out = []
        for v in variants:
            if v not in seen:
                seen.add(v); out.append(v)
        return out
    return [run_label]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

# -----------------------
# I/O
# -----------------------

def resolve_paths(subj, ses, task, run, glm_root, fmri_root):
    """Resolve dtseries and H5 by trying run aliases; fallback to glob; infer run_used."""
    aliases = run_aliases(run)
    fmri_dir = Path(fmri_root) / subj / ses
    glm_dir  = Path(glm_root)  / subj / ses / "func"

    ts_candidates = []
    for rl in aliases:
        ts_candidates.append(fmri_dir / f"{subj}_{ses}_task-{task}_{rl}_space-Glasser64k_bold_sm4.dtseries.nii")
        ts_candidates.append(fmri_dir / f"{subj}_{ses}_task-{task}_{rl}_space-Glasser64k_bold.dtseries.nii")
    ts_path = first_existing(ts_candidates)

    if ts_path is None:
        globbed_sm4 = sorted(fmri_dir.glob(f"{subj}_{ses}_task-{task}_*_space-Glasser64k_bold_sm4.dtseries.nii"))
        globbed_raw = sorted(fmri_dir.glob(f"{subj}_{ses}_task-{task}_*_space-Glasser64k_bold.dtseries.nii"))
        globbed = globbed_sm4 + globbed_raw
        if not globbed:
            tried = [p.as_posix() for p in ts_candidates]
            raise FileNotFoundError(
                f"Timeseries not found. Tried aliases: {tried} and no glob match in {fmri_dir}"
            )
        def extract_run_num_from_name(name: str):
            try:
                part = name.split(f"task-{task}_", 1)[1].split("_space-", 1)[0]
                return int(part.split('run-')[1]) if part.startswith('run-') and part[4:].isdigit() else None
            except Exception:
                return None
        target = extract_run_num_from_name(f"dummy_task-{task}_{run}_space-")
        chosen = None
        if target is not None:
            for g in globbed:
                n = extract_run_num_from_name(g.name)
                if n == target:
                    chosen = g; break
        ts_path = chosen or globbed[0]

    try:
        run_used = ts_path.name.split(f"task-{task}_", 1)[1].split("_space-", 1)[0]
    except Exception:
        run_used = run

    h5_path = glm_dir / f"{subj}_{ses}_task-{task}_{run_used}_betas.h5"
    if not h5_path.exists():
        h5_candidates = [glm_dir / f"{subj}_{ses}_task-{task}_{rl}_betas.h5" for rl in run_aliases(run_used)]
        h5_path = first_existing(h5_candidates)

    if (h5_path is None) or (not h5_path.exists()):
        raise FileNotFoundError(
            f"GLM H5 not found. Looked for {subj}_{ses}_task-{task}_{run_used}_betas.h5 and aliases under {glm_dir}"
        )

    return ts_path, h5_path, run_used

def load_glm_data(h5_path):
    data = {}
    with h5py.File(h5_path, 'r') as f:
        # Required
        data['betas_full'] = f['betas_full'][:]                 # (V x K)
        data['betas'] = f['betas'][:]                           # (V x K_task)
        data['design_matrix'] = f['design_matrix'][:]           # (T_kept x K)
        data['design_col_names'] = [n.decode('utf-8') if isinstance(n, bytes) else n
                                    for n in f['design_col_names'][:]]
        # GLM native R² metrics from glm_analysis.py
        if "r2_full" in f:
            data["r2_full"] = f["r2_full"][:]
        if "r2_noise" in f:
            data["r2_noise"] = f["r2_noise"][:]
        if "r2_task" in f:
            data["r2_task"] = f["r2_task"][:]
        if "r2_full_adj" in f:
            data["r2_full_adj"] = f["r2_full_adj"][:]
        if "r2_noise_adj" in f:
            data["r2_noise_adj"] = f["r2_noise_adj"][:]
        if "r2_task_adj" in f:
            data["r2_task_adj"] = f["r2_task_adj"][:]
        # Optional fast-path outputs
        data['has_yhat'] = 'yhat' in f
        if data['has_yhat']:
            data['yhat'] = f['yhat'][:]                         # (T_kept x V)
        data['has_resid'] = 'resid' in f
        if data['has_resid']:
            data['resid'] = f['resid'][:]                       # (T_kept x V)
        # Attributes
        attrs = f.attrs
        data['task_regressor_names'] = [n.decode('utf-8') if isinstance(n, bytes) else n for n in attrs['task_regressor_names']]
        data['task_col_start'] = int(attrs['task_col_start'])
        data['task_col_end'] = int(attrs['task_col_end'])
        data['rho_ar1'] = float(attrs.get('rho_ar1', 0.0))
        data['tmask_dropped'] = int(attrs.get('tmask_dropped', 0))
        data['high_pass_sec'] = float(attrs.get('high_pass_sec', np.nan))
        data['order_in_X'] = attrs.get('order_in_X', "")
        if isinstance(data['order_in_X'], bytes):
            data['order_in_X'] = data['order_in_X'].decode('utf-8')
        data['estimator'] = attrs.get('estimator', "")
        if isinstance(data['estimator'], bytes):
            data['estimator'] = data['estimator'].decode('utf-8')
    return data


def load_fmri_data(ts_path, tmask):
    _ = tmask  # kept for backward-compatible function signature
    return nib.load(str(ts_path)).get_fdata(dtype=np.float32)  # (T x V)


def preprocess_target_to_match_glm(Y_raw, glm, tmask_override=None):
    """
    Recreate the target space used by glm_analysis.py:
      1) drop tmask frames
      2) PSC against mean of full run
    """
    tmask = int(glm.get("tmask_dropped", tmask_override if tmask_override is not None else 0))
    psc_eps = float(glm.get("psc_eps", 1e-8) or 1e-8)

    Y_full = Y_raw.astype(np.float64, copy=False)
    keep = np.ones((Y_full.shape[0],), dtype=bool)
    keep[:tmask] = False
    Y = Y_full[keep, :]

    mu = np.mean(Y_full, axis=0, keepdims=True)
    safe = np.abs(mu) >= psc_eps
    Yproc = 100.0 * (Y - mu) / np.where(safe, mu, 1.0)
    Yproc[:, ~safe.squeeze()] = 0.0
    return Yproc.astype(np.float32)

# -----------------------
# Core
# -----------------------

def compute_predictions(X, betas_sel):
    return X @ betas_sel.T

def compute_r2(Y_true, Y_pred):
    V = Y_true.shape[1]
    out = np.empty(V, dtype=np.float32)
    for v in range(V):
        try:
            out[v] = r2_score(Y_true[:, v], Y_pred[:, v])
        except Exception:
            out[v] = np.nan
    return out

# -----------------------
# Plotting
# -----------------------

def plot_predictions_vs_actual(Y_true, Y_pred, vlist, outdir, subj, task, ses, run_used,
                               intercept=None, r2_scores=None, suptitle_extra=None, tag_suffix="",
                               labels=None):
    """
    vlist: indices into Y_true/Y_pred to plot (local indices)
    labels: optional list of labels (e.g., global vertex IDs) to show in titles, same length as vlist
    """
    n = min(len(vlist), 6)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n))
    if n == 1:
        axes = [axes]
    t = np.arange(Y_true.shape[0])
    for i in range(n):
        v = vlist[i]
        axes[i].plot(t, Y_true[:, v], label='Actual', alpha=0.85)
        axes[i].plot(t, Y_pred[:, v], label='Predicted', alpha=0.85)
        # Per-panel R² if provided
        r2_txt = None
        if r2_scores is not None and 0 <= v < len(r2_scores) and np.isfinite(r2_scores[v]):
            r2_txt = f"R²={r2_scores[v]:.3f}"
        label_id = labels[i] if (labels is not None and i < len(labels)) else v
        title = f"Vertex {label_id}"
        if r2_txt:
            title += f" · {r2_txt}"
        if intercept is not None and 0 <= v < len(intercept) and np.isfinite(intercept[v]):
            title += f" · β₀={float(intercept[v]):.3f}"
        axes[i].set_title(title)
        axes[i].set_xlabel('Time (frames after tmask)')
        axes[i].set_ylabel('Signal')
        axes[i].legend(); axes[i].grid(True, alpha=0.3)
    # Suptitle with R² summary
    if r2_scores is not None and len(vlist) > 0:
        sel = np.array([r2_scores[v] if 0 <= v < len(r2_scores) else np.nan for v in vlist])
        finite = sel[np.isfinite(sel)]
        if finite.size:
            sup = f"Pred vs Actual · R² mean={np.nanmean(finite):.3f}, median={np.nanmedian(finite):.3f}"
        else:
            sup = "Pred vs Actual · R² summary: n/a"
    else:
        sup = "Pred vs Actual"
    if suptitle_extra:
        sup += f" · {suptitle_extra}"
    plt.suptitle(sup)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    tag = f"_pred_vs_actual{tag_suffix}.png"
    out = Path(outdir) / f"{subj}_{ses}_task-{task}_{run_used}{tag}"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    return out


def plot_residual_panels(resid, vlist, outdir, subj, task, ses, run_used):
    n = min(3, len(vlist))
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)
    t = np.arange(resid.shape[0])
    for i in range(n):
        v = vlist[i]
        # trace
        axes[0, i].plot(t, resid[:, v], alpha=0.9)
        axes[0, i].set_title(f'Vertex {v}: Residuals')
        axes[0, i].set_xlabel('Time (frames after tmask)'); axes[0, i].set_ylabel('Residual')
        axes[0, i].grid(True, alpha=0.3)
        # histogram
        vals = resid[:, v]
        axes[1, i].hist(vals, bins=30, alpha=0.7, density=True)
        mu, sd = np.mean(vals), np.std(vals)
        x = np.linspace(vals.min(), vals.max(), 100)
        y = stats.norm.pdf(x, mu, sd) if sd > 0 else np.zeros_like(x)
        axes[1, i].plot(x, y, 'r--', label='Normal fit')
        axes[1, i].set_title(f'Vertex {v}: Residual Distribution')
        axes[1, i].set_xlabel('Residual'); axes[1, i].set_ylabel('Density')
        axes[1, i].legend(); axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(outdir) / f"{subj}_{ses}_task-{task}_{run_used}_residuals.png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    return out


def plot_r2_distribution(r2, outdir, subj, task, ses, run_used, tag="sampled"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    finite = r2[~np.isnan(r2)]
    axes[0].hist(finite, bins=50, density=True, alpha=0.8)
    if finite.size:
        axes[0].axvline(np.nanmean(finite), linestyle='--', label=f'Mean: {np.nanmean(finite):.3f}')
        axes[0].axvline(np.nanmedian(finite), linestyle='--', label=f'Median: {np.nanmedian(finite):.3f}')
    axes[0].set_xlabel('R²'); axes[0].set_ylabel('Density'); axes[0].set_title(f'R² Distribution ({tag})'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].boxplot(finite, vert=True); axes[1].set_ylabel('R²'); axes[1].set_title(f'R² Box Plot ({tag})'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(outdir) / f"{subj}_{ses}_task-{task}_{run_used}_r2_distribution_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(); return out


def plot_task_beta_diff_hist(diffs, outdir, subj, task, ses, run_used):
    plt.figure(figsize=(6, 4))
    plt.hist(diffs.flatten(), bins=50, density=True, alpha=0.85)
    plt.xlabel('|Δ beta_task| (full - saved_task)'); plt.ylabel('Density'); plt.title('Task beta absolute differences (sampled vertices)')
    out = Path(outdir) / f"{subj}_{ses}_task-{task}_{run_used}_task_beta_diff_hist.png"
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(); return out


def plot_beta_distribution(betas_task, outdir, subj, task, ses, run_used):
    vals = betas_task.flatten()
    vals = vals[np.isfinite(vals)]
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=80, density=True, alpha=0.85)
    if vals.size:
        plt.axvline(np.mean(vals), linestyle="--", linewidth=1.0, label=f"Mean: {np.mean(vals):.3g}")
        plt.axvline(np.median(vals), linestyle=":", linewidth=1.0, label=f"Median: {np.median(vals):.3g}")
    plt.xlabel('Task beta value')
    plt.ylabel('Density')
    plt.title('Task beta distribution (sampled vertices)')
    if vals.size:
        plt.legend()
    out = Path(outdir) / f"{subj}_{ses}_task-{task}_{run_used}_beta_distribution_task.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def plot_predictions_vs_actual_paper(
    Y_true,
    Y_pred,
    selected_local,
    selected_global,
    r2_full_adj_sel,
    r2_noise_adj_sel,
    r2_task_adj_sel,
    outdir,
    subj,
    task,
    ses,
    run_used,
    journal="nature",
    span="double",
    full_width_mm=None,
    panel_hw_ratio=0.5,
):
    """3x2 publication-style panel for six sampled vertices."""
    gap_w_mm = 1.5
    gap_h_mm = 4.5
    if full_width_mm is None:
        figsize, panel_w_mm, panel_h_mm = get_figsize(journal, span, n_cols=2, n_rows=3)
    else:
        panel_w_mm = (float(full_width_mm) - gap_w_mm) / 2.0
        panel_h_mm = panel_w_mm * float(panel_hw_ratio)
        total_h_mm = 3.0 * panel_h_mm + 2.0 * gap_h_mm
        figsize = (mm_to_inches(float(full_width_mm)), mm_to_inches(total_h_mm))
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    axes_flat = axes.ravel()

    palette = get_palette("vigil")
    c_actual = palette[2]
    c_pred = palette[0]
    t = np.arange(Y_true.shape[0])

    for i, ax in enumerate(axes_flat):
        if i >= len(selected_local):
            ax.axis("off")
            continue
        loc = int(selected_local[i])
        gid = int(selected_global[i])
        ax.plot(t, Y_true[:, loc], color=c_actual, linewidth=0.9, label="ground truth")
        ax.plot(t, Y_pred[:, loc], color=c_pred, linewidth=0.9, linestyle="--", label="predicted")
        ax.set_title(
            f"V{gid} | full={r2_full_adj_sel[i]:.3f}, task={r2_task_adj_sel[i]:.3f}, noise={r2_noise_adj_sel[i]:.3f}"
        )
        ax.set_ylabel("PSC (%)")
        if i >= 4:
            ax.set_xlabel("Time (TR)")
        apply_clean_spines(ax)
        if i == 0:
            ax.legend(loc="best")

    add_panel_labels(axes_flat[: min(6, len(selected_local))], x=-0.06, y=1.02)
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.985,
        bottom=0.07,
        wspace=max(0.08, gap_w_mm / panel_w_mm),
        hspace=max(0.08, gap_h_mm / panel_h_mm),
    )
    out_base = (
        Path(outdir)
        / f"{subj}_{ses}_task-{task}_{run_used}_pred_vs_actual_sampled6_adjR2_paper"
    )
    save_figure(fig, out_base, formats=("svg", "pdf"))
    plt.close(fig)
    return out_base


def plot_adjusted_r2_distribution_paper(
    r2_full_adj,
    r2_noise_adj,
    r2_task_adj,
    outdir,
    subj,
    task,
    ses,
    run_used,
    journal="nature",
    span="single",
):
    """Single-column publication-style adjusted R² distribution plot."""
    figsize, panel_w_mm, panel_h_mm = get_figsize(journal, span, n_cols=1, n_rows=1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    palette = get_palette("vigil")
    c_task = palette[0]
    c_noise = palette[1]
    c_full = palette[2]

    bins = 80
    ax.hist(r2_full_adj, bins=bins, density=True, alpha=0.45, color=c_full, label="R2 full adj")
    ax.hist(r2_noise_adj, bins=bins, density=True, alpha=0.45, color=c_noise, label="R2 noise adj")
    ax.hist(r2_task_adj, bins=bins, density=True, alpha=0.45, color=c_task, label="R2 task adj")
    ax.set_xlabel("Adjusted R2")
    ax.set_ylabel("Density")
    ax.set_title("Adjusted R2 distributions")
    ax.legend(loc="best")
    apply_clean_spines(ax)
    fig.subplots_adjust(
        wspace=max(0.05, 1.5 / panel_w_mm),
        hspace=max(0.05, 1.5 / panel_h_mm),
    )
    out_base = (
        Path(outdir)
        / f"{subj}_{ses}_task-{task}_{run_used}_adjusted_r2_distribution_paper"
    )
    save_figure(fig, out_base, formats=("svg", "pdf"))
    plt.close(fig)
    return out_base

# -----------------------
# Global R² (subset) using saved full betas
# -----------------------

def r2_subset_with_full_betas(X, betas_full, Y_full, idx_eval, chunk):
    r2_out = np.full(len(idx_eval), np.nan, dtype=np.float32)
    for start in range(0, len(idx_eval), chunk):
        sel = idx_eval[start:start+chunk]
        B = betas_full[sel, :].astype(np.float32)
        Yhat = X @ B.T
        Yc = Y_full[:, sel]
        for j in range(len(sel)):
            try:
                r2_out[start + j] = r2_score(Yc[:, j], Yhat[:, j])
            except Exception:
                r2_out[start + j] = np.nan
    return r2_out

# -----------------------
# Report
# -----------------------

def write_report(glm, r2_sampled, outdir, subj, task, ses, run_used, sampled_vertices, V_total,
                 r2_all=None, allN=None, beta_diff_stats=None, extreme_info=None):
    lines = []
    lines.append("# GLM Evaluation Report (Vertex-level)")
    lines.append(f"**Subject:** {subj}  ")
    lines.append(f"**Task:** {task}  ")
    lines.append(f"**Session:** {ses}  ")
    lines.append(f"**Run:** {run_used}")
    lines.append("")
    lines.append("## Model Specs")
    lines.append(f"- # regressors (K): {glm['design_matrix'].shape[1]}")
    lines.append(f"- # task regressors: {len(glm['task_regressor_names'])}")
    lines.append(f"- # time points after tmask (T_kept): {glm['design_matrix'].shape[0]}")
    if 'rho_ar1' in glm:
        lines.append(f"- AR(1) rho: {glm['rho_ar1']:.3f}")
    if 'estimator' in glm and glm['estimator']:
        lines.append(f"- Estimator: {glm['estimator']}")
    lines.append("")
    lines.append("## Vertex Sampling")
    lines.append(f"- Total vertices in run: {V_total}")
    lines.append(f"- Sampled vertices for plots: {len(sampled_vertices)}")
    lines.append("")
    lines.append("## Fit (Sampled Vertices)")
    lines.append(f"- Mean R²: {np.nanmean(r2_sampled):.3f}")
    lines.append(f"- Median R²: {np.nanmedian(r2_sampled):.3f}")
    lines.append(f"- R² std: {np.nanstd(r2_sampled):.3f}")
    lines.append(f"- R² range: [{np.nanmin(r2_sampled):.3f}, {np.nanmax(r2_sampled):.3f}]")
    for thr in (0.0, 0.1, 0.2):
        frac = np.mean(r2_sampled > thr)
        lines.append(f"- Vertices with R² > {thr:.1f}: {int(np.sum(r2_sampled > thr))}/{len(r2_sampled)} ({100*frac:.1f}%)")
    if r2_all is not None:
        lines.append("")
        lines.append("## Global R² (reduced subset)")
        lines.append(f"- Evaluated vertices: {allN}")
        lines.append(f"- Mean R²: {np.nanmean(r2_all):.3f}")
        lines.append(f"- Median R²: {np.nanmedian(r2_all):.3f}")
        lines.append(f"- R² std: {np.nanstd(r2_all):.3f}")
        lines.append(f"- R² range: [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]")
    if beta_diff_stats is not None:
        mad, mx = beta_diff_stats
        lines.append("")
        lines.append("## Task Beta Consistency (full vs saved task slice)")
        lines.append(f"- Mean |Δ| over sampled vertices: {mad:.6g}")
        lines.append(f"- Max  |Δ| over sampled vertices: {mx:.6g}")
    if extreme_info is not None:
        prop, count, thr, total = extreme_info
        lines.append("")
        lines.append("## Extremely Negative R²")
        lines.append(f"- Threshold: R² < {thr}")
        lines.append(f"- Count: {count} / {total} ({100.0*count/total:.2f}%)")

    rep = Path(outdir) / f"{subj}_{ses}_task-{task}_{run_used}_evaluation_report.md"
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return rep


# -----------------------
# Main
# -----------------------

def main():
    args = get_args()
    np.random.seed(args.seed)

    ts_path, h5_path, run_used = resolve_paths(args.subj, args.ses, args.task, args.run, args.glm_root, args.fmri_root)

    print(f"Loading GLM data from: {h5_path}")
    glm = load_glm_data(h5_path)

    print(f"Loading fMRI dtseries from: {ts_path}")
    Y_raw = load_fmri_data(ts_path, args.tmask)  # (T_full x V)
    Y_full = preprocess_target_to_match_glm(Y_raw, glm, tmask_override=args.tmask)  # (T_kept x V)
    if Y_full.shape[0] != glm["design_matrix"].shape[0]:
        raise ValueError(
            f"Target/design length mismatch: Y={Y_full.shape[0]}, X={glm['design_matrix'].shape[0]}. "
            f"Check tmask and input timeseries selection."
        )
    T, V_total = Y_full.shape

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # Sample vertices with nonzero variance
    stds = np.std(Y_full, axis=0)
    nonzero = np.where(stds > 0)[0]
    if len(nonzero) == 0:
        raise RuntimeError("All vertices have zero variance after tmask.")
    V_sel = min(args.max_vertices, len(nonzero))
    sampled_vertices = np.sort(np.random.choice(nonzero, size=V_sel, replace=False))

    # Slice to sampled
    Y = Y_full[:, sampled_vertices]
    X = glm['design_matrix'].astype(np.float32)

    # Betas
    B_full = glm['betas_full'][sampled_vertices, :].astype(np.float32)  # (V_sel x K)
    task_start, task_end = glm['task_col_start'], glm['task_col_end']
    B_task_saved = glm['betas'][sampled_vertices, :].astype(np.float32) # (V_sel x K_task)

    # Consistency check (task slice)
    diffs = np.abs(B_full[:, task_start:task_end] - B_task_saved)
    mean_abs_diff = float(np.nanmean(diffs)); max_abs_diff = float(np.nanmax(diffs))
    print(f"[Beta consistency] mean |Δ|={mean_abs_diff:.6g}, max |Δ|={max_abs_diff:.6g}")

    # Predictions and residuals (sampled)
    if glm.get('has_yhat', False):
        print("Using saved yhat slice for predictions...")
        Yhat = glm['yhat'][:, sampled_vertices]
    else:
        print("Computing predictions: X @ betas_full^T (sampled)...")
        Yhat = X @ B_full.T

    if glm.get('has_resid', False):
        print("Using saved residuals slice...")
        Resid = glm['resid'][:, sampled_vertices]
    else:
        Resid = Y - Yhat

    R2 = compute_r2(Y, Yhat)

    # Interesting vertices: top-3 by R² + 3 random
    valid = np.where(np.isfinite(R2))[0]
    if len(valid) == 0:
        interesting = np.arange(min(6, Y.shape[1]))
    else:
        top_n = min(3, len(valid))
        top_idx = valid[np.argsort(R2[valid])[-top_n:]]
        remaining = np.setdiff1d(np.arange(Y.shape[1]), top_idx)
        rand_n = min(3, len(remaining))
        rand_idx = np.random.choice(remaining, size=rand_n, replace=False) if rand_n > 0 else np.array([], dtype=int)
        interesting = np.unique(np.concatenate([top_idx, rand_idx]))[:6]

    # Intercept betas (for titles), if present
    try:
        const_idx = glm['design_col_names'].index('constant')
        intercept = B_full[:, const_idx]
    except ValueError:
        intercept = None

    # Plots — include per-panel R² and suptitle summary; label with global vertex IDs
    pred_png  = plot_predictions_vs_actual(
        Y, Yhat, interesting, outdir, args.subj, args.task, args.ses, run_used,
        intercept=intercept, r2_scores=R2, suptitle_extra=None, tag_suffix="",
        labels=list(sampled_vertices[interesting])
    )
    resid_png = plot_residual_panels(Resid, interesting, outdir, args.subj, args.task, args.ses, run_used)
    r2_png    = plot_r2_distribution(R2, outdir, args.subj, args.task, args.ses, run_used, tag="sampled")
    diff_png  = plot_task_beta_diff_hist(diffs, outdir, args.subj, args.task, args.ses, run_used)
    beta_dist_png = plot_beta_distribution(B_task_saved, outdir, args.subj, args.task, args.ses, run_used)

    # Save sampled R² CSV (global vertex ids + r2)
    r2_sampled_csv = Path(outdir) / f"{args.subj}_{args.ses}_task-{args.task}_{run_used}_r2_sampled.csv"
    np.savetxt(r2_sampled_csv,
               np.column_stack([sampled_vertices, R2]),
               delimiter=",", header="vertex,r2", comments="")

    # Global R² (all nonzero-variance vertices) for extreme/non-extreme analyses
    print("Computing R² for all nonzero-variance vertices...")
    r2_all_full = r2_subset_with_full_betas(X, glm['betas_full'], Y_full, nonzero, args.chunk_size)
    thr = args.r2_extreme_thr

    # EXTREME (R² < thr)
    extreme_mask = r2_all_full < thr
    extreme_count = int(np.nansum(extreme_mask))
    extreme_prop = extreme_count / len(nonzero)
    print(f"Extremely negative R² (< {thr}): {extreme_count} / {len(nonzero)} = {100*extreme_prop:.2f}%")

    # Save per-vertex R² CSV for all nonzero-variance vertices
    r2_all_csv = Path(outdir) / f"{args.subj}_{args.ses}_task-{args.task}_{run_used}_r2_all.csv"
    np.savetxt(r2_all_csv,
               np.column_stack([nonzero, r2_all_full]),
               delimiter=",", header="vertex,r2", comments="")

    # Plot pred_vs_actual for up to 6 truly extreme-negative vertices (most negative R²)
    pred_ext_png = None
    r2_ext_csv = None
    extreme_local = np.where(extreme_mask)[0]   # positions within 'nonzero'
    if extreme_local.size > 0:
        order = np.argsort(r2_all_full[extreme_local])  # ascending (most negative first)
        sel_local = extreme_local[order[:min(6, extreme_local.size)]]
        sel_global = nonzero[sel_local]  # map to GLOBAL vertex IDs

        B_ext = glm['betas_full'][sel_global, :].astype(np.float32)  # (n x K)
        Y_ext = Y_full[:, sel_global]                                # (T x n)
        Yhat_ext = X @ B_ext.T                                       # (T x n)

        # Use the R² that drove the selection (exactly matches)
        r2_ext_sel = r2_all_full[sel_local]

        # Intercept if present
        try:
            const_idx2 = glm['design_col_names'].index('constant')
            intercept_ext = B_ext[:, const_idx2]
        except ValueError:
            intercept_ext = None

        pred_ext_png = plot_predictions_vs_actual(
            Y_ext, Yhat_ext, list(range(Y_ext.shape[1])), outdir,
            args.subj, args.task, args.ses, run_used,
            intercept=intercept_ext, r2_scores=r2_ext_sel,
            suptitle_extra=f"extreme R² < {thr} (most negative)", tag_suffix="_extneg",
            labels=list(sel_global)  # show GLOBAL vertex IDs in titles
        )

        # CSV for selected extreme-negatives (GLOBAL ids + R²)
        r2_ext_csv = Path(outdir) / f"{args.subj}_{args.ses}_task-{args.task}_{run_used}_r2_extneg_selected.csv"
        np.savetxt(r2_ext_csv,
                   np.column_stack([sel_global, r2_ext_sel]),
                   delimiter=",", header="vertex,r2", comments="")

    # NON-EXTREME (R² >= thr)
    non_extreme_mask = (r2_all_full >= thr) & np.isfinite(r2_all_full)
    non_extreme_count = int(np.nansum(non_extreme_mask))
    non_extreme_prop_eval = non_extreme_count / len(nonzero)   # among evaluated (nonzero-variance)
    non_extreme_prop_all = non_extreme_count / V_total         # among ALL vertices (incl zero-variance)
    print(f"Non-extreme R² (>= {thr}): {non_extreme_count} / {len(nonzero)} = {100*non_extreme_prop_eval:.2f}% of evaluated")
    print(f"Non-extreme R² (>= {thr}) among ALL vertices: {non_extreme_count} / {V_total} = {100*non_extreme_prop_all:.2f}%")

    r2_nonextreme_png = None
    r2_nonextreme_csv = None
    if non_extreme_count > 0:
        r2_nonextreme = r2_all_full[non_extreme_mask]
        r2_nonextreme_png = plot_r2_distribution(
            r2_nonextreme, outdir, args.subj, args.task, args.ses, run_used,
            tag=f"nonextreme_all_thr{thr}"
        )
        # CSV for non-extreme vertices (GLOBAL ids + r2)
        nonextreme_vertices = nonzero[non_extreme_mask]
        r2_nonextreme_csv = Path(outdir) / f"{args.subj}_{args.ses}_task-{args.task}_{run_used}_r2_nonextreme_all.csv"
        np.savetxt(r2_nonextreme_csv,
                   np.column_stack([nonextreme_vertices, r2_nonextreme]),
                   delimiter=",", header="vertex,r2", comments="")

    # Optional global R² subset plot (unchanged behavior)
    r2_all = None; r2_all_png = None; allN = None
    if args.compute_all_r2 and (args.max_vertices_r2 is None or args.max_vertices_r2 > 0):
        if args.max_vertices_r2 is None:
            idx_eval = nonzero
        else:
            idx_eval = np.sort(np.random.choice(nonzero, size=min(args.max_vertices_r2, len(nonzero)), replace=False))
        allN = len(idx_eval)
        r2_all = r2_subset_with_full_betas(X, glm['betas_full'], Y_full, idx_eval, args.chunk_size)
        r2_all_png = plot_r2_distribution(r2_all, outdir, args.subj, args.task, args.ses, run_used, tag=f"all_subsetN{allN}")

    # Report
    report_md = write_report(glm, R2, outdir, args.subj, args.task, args.ses, run_used, sampled_vertices, V_total,
                             r2_all=r2_all, allN=allN, beta_diff_stats=(mean_abs_diff, max_abs_diff),
                             extreme_info=(extreme_prop, extreme_count, thr, len(nonzero)))

    # Console summary
    print("\n=== Evaluation Summary ===")
    print(f"Sampled vertices: {len(sampled_vertices)} / non-zero-variance: {len(nonzero)}")
    print(f"Sampled R² -> mean={np.nanmean(R2):.3f}, median={np.nanmedian(R2):.3f}, range=[{np.nanmin(R2):.3f}, {np.nanmax(R2):.3f}]")
    if r2_all is not None:
        print(f"Global (subset N={allN}) R² -> mean={np.nanmean(r2_all):.3f}, median={np.nanmedian(r2_all):.3f}, range=[{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]")
    print(f"Task beta consistency (sampled): mean |Δ|={mean_abs_diff:.6g}, max |Δ|={max_abs_diff:.6g}")
    print("Saved:")
    print(" -", pred_png)
    if pred_ext_png:
        print(" -", pred_ext_png)
    print(" -", resid_png)
    print(" -", r2_png)
    if r2_all_png:
        print(" -", r2_all_png)
    if r2_nonextreme_png:
        print(" -", r2_nonextreme_png)
    print(" -", diff_png)
    print(" -", beta_dist_png)
    print(" -", report_md)
    print(" -", r2_all_csv)
    print(" -", r2_sampled_csv)
    if r2_nonextreme_csv:
        print(" -", r2_nonextreme_csv)
    if r2_ext_csv:
        print(" -", r2_ext_csv)

    # Requested figures: publication-style outputs using adjusted R² from glm_analysis.py
    if all(k in glm for k in ("r2_full_adj", "r2_noise_adj", "r2_task_adj")):
        r2_full_adj_all = glm["r2_full_adj"].astype(np.float32)
        r2_noise_adj_all = glm["r2_noise_adj"].astype(np.float32)
        r2_task_adj_all = glm["r2_task_adj"].astype(np.float32)

        # Select six sampled vertices (highest adjusted task R² among sampled set).
        task_adj_sampled = r2_task_adj_all[sampled_vertices]
        finite_local = np.where(np.isfinite(task_adj_sampled))[0]
        if finite_local.size > 0:
            order = np.argsort(task_adj_sampled[finite_local])[::-1]
            top_local = finite_local[order[: min(6, finite_local.size)]]
        else:
            top_local = np.arange(min(6, len(sampled_vertices)))
        sel_local = np.array(top_local, dtype=int)
        sel_global = sampled_vertices[sel_local]

        pred_paper_base = plot_predictions_vs_actual_paper(
            Y,
            Yhat,
            selected_local=sel_local,
            selected_global=sel_global,
            r2_full_adj_sel=r2_full_adj_all[sel_global],
            r2_noise_adj_sel=r2_noise_adj_all[sel_global],
            r2_task_adj_sel=r2_task_adj_all[sel_global],
            outdir=outdir,
            subj=args.subj,
            task=args.task,
            ses=args.ses,
            run_used=run_used,
            journal=args.journal,
            span=args.pred_span,
            full_width_mm=args.pred_full_width_mm,
            panel_hw_ratio=args.pred_panel_hw_ratio,
        )

        nz = nonzero
        dist_paper_base = plot_adjusted_r2_distribution_paper(
            r2_full_adj=r2_full_adj_all[nz][np.isfinite(r2_full_adj_all[nz])],
            r2_noise_adj=r2_noise_adj_all[nz][np.isfinite(r2_noise_adj_all[nz])],
            r2_task_adj=r2_task_adj_all[nz][np.isfinite(r2_task_adj_all[nz])],
            outdir=outdir,
            subj=args.subj,
            task=args.task,
            ses=args.ses,
            run_used=run_used,
            journal=args.journal,
            span=args.dist_span,
        )

        print("Publication-style figures:")
        print(" -", pred_paper_base.with_suffix(".svg"))
        print(" -", pred_paper_base.with_suffix(".pdf"))
        print(" -", dist_paper_base.with_suffix(".svg"))
        print(" -", dist_paper_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
