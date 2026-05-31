#!/usr/bin/env python3
"""
Per-run trial-level GLM with fixed PSC preprocessing and run-wise variance partitioning.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from nilearn.glm.first_level import make_first_level_design_matrix, spm_hrf
from sklearn.linear_model import LinearRegression

from utils import glm_confounds_construction, load_valid_vertex_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser(description="Per-run trial-level GLM with fixed PSC + dual-model R2.")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames to drop at run start")
    p.add_argument("--correct_only", action="store_true", help="Use only correct trials")
    p.add_argument("--tasks", nargs="+", default=["ctxdm"])
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--include_types", nargs="+", default=["encoding", "delay"])
    p.add_argument(
        "--fmri_root",
        default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled",
    )
    p.add_argument("--conf_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--events_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data")
    p.add_argument("--high_pass_sec", type=float, default=128.0)
    p.add_argument("--psc_eps", type=float, default=1e-8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_yhat", action="store_true")
    p.add_argument("--save_resid", action="store_true")
    p.add_argument("--estimator", choices=["wls_ar1", "ols"], default="wls_ar1")
    p.add_argument("--ridge", type=float, default=0.0, help="Optional L2 ridge for OLS.")
    p.add_argument(
        "--compute_vif",
        action="store_true",
        help="Compute VIF for task regressors (slow). Default: off.",
    )
    p.add_argument(
        "--save_surface_plots",
        action="store_true",
        help="Generate brain surface R2 plots via nilearn (slow). Default: off.",
    )
    return p.parse_args()


def default_output_root(out_root_base: Path, correct_only: bool, subj: str) -> Path:
    return out_root_base / ("correct_trial_level_betas" if correct_only else "trial_level_betas") / subj


def discover_sessions(fmri_root_dir: Path):
    pattern = re.compile(r"^ses-\d+$")
    return sorted([p.name for p in fmri_root_dir.iterdir() if p.is_dir() and pattern.match(p.name)])


def discover_runs_from_timeseries(subj: str, ses: str, task: str, fmri_root_dir: Path):
    pat = f"{subj}_{ses}_task-{task}_*_space-Glasser64k_bold_sm4.dtseries.nii"
    files = sorted((fmri_root_dir / ses).glob(pat))
    run_pattern = re.compile(
        rf"{re.escape(subj)}_{re.escape(ses)}_task-{re.escape(task)}_(.+?)_space-Glasser64k_bold_sm4\.dtseries\.nii"
    )
    runs = []
    for f in files:
        m = run_pattern.match(f.name)
        if m:
            runs.append(m.group(1))
    return sorted(set(runs))


def run_aliases(run_label: str):
    m = re.match(r"^(run-)(\d+)$", run_label)
    if m:
        prefix, num = m.group(1), m.group(2)
        pad = f"{int(num):02d}"
        out = [f"{prefix}{num}"]
        if pad != num:
            out.append(f"{prefix}{pad}")
        return out
    return [run_label]


def run_number(run_label: str):
    m = re.match(r"^run-(\d+)$", str(run_label).strip())
    return int(m.group(1)) if m else None


def select_runs(runs: list[str], wanted: list[str] | None) -> list[str]:
    if not wanted:
        return runs
    wanted_set = set(wanted)
    wanted_num = {run_number(r) for r in wanted if run_number(r) is not None}
    selected = []
    for r in runs:
        rn = run_number(r)
        if r in wanted_set or (rn is not None and rn in wanted_num):
            selected.append(r)
    return selected


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def per_run_drift_and_intercept(num_trs: int, tr: float, high_pass_sec: float = 128.0):
    frame_times = np.arange(num_trs) * tr
    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=None,
        hrf_model=None,
        drift_model="cosine",
        high_pass=1.0 / high_pass_sec,
        add_regs=None,
        add_reg_names=None,
        oversampling=1,
    )
    cols = [c for c in dm.columns if c == "constant" or c.startswith("cosine")]
    return dm[cols].to_numpy(dtype=np.float32), cols


def _coerce_trial_number(v, fallback):
    try:
        return int(v)
    except Exception:
        return int(fallback)


def build_trialwise_regressors(
    df_events: pd.DataFrame,
    num_trs: int,
    tr: float,
    include_types: list[str],
    correct_only: bool = False,
):
    req_cols = {"onset_time", "offset_time", "type"}
    if not req_cols.issubset(df_events.columns):
        missing = ", ".join(sorted(req_cols - set(df_events.columns)))
        raise ValueError(f"Events missing required columns: {missing}")

    include_types = [t.lower() for t in include_types]
    types = df_events["type"].astype(str).str.strip().str.lower()
    mask = types.isin(include_types)

    if correct_only and "is_correct" in df_events.columns:
        def is_negative(v):
            if isinstance(v, (bool, np.bool_)):
                return bool(v) is False
            if pd.isna(v):
                return False
            return str(v).strip().lower() in {"false", "b"}

        mask = mask & (~df_events["is_correct"].apply(is_negative))

    selected = df_events.loc[mask].copy().reset_index(drop=True)
    if selected.empty:
        empty_cols = [
            "trial_index",
            "item_index",
            "trialNumber",
            "type",
            "onset_time",
            "offset_time",
            "is_correct",
            "stim_order",
            "location",
            "category",
            "object",
        ]
        return np.zeros((num_trs, 0), dtype=np.float32), [], pd.DataFrame(columns=empty_cols)

    selected["type"] = selected["type"].astype(str).str.strip().str.lower()
    selected["_type_order"] = selected["type"].map({t: i for i, t in enumerate(include_types)}).fillna(999).astype(int)
    if "trialNumber" in selected.columns:
        selected["_trial_num"] = [
            _coerce_trial_number(v, i + 1) for i, v in enumerate(selected["trialNumber"].tolist())
        ]
    else:
        selected["_trial_num"] = np.arange(1, selected.shape[0] + 1)
        selected["trialNumber"] = selected["_trial_num"]

    selected["_onset"] = pd.to_numeric(selected["onset_time"], errors="coerce")
    selected = selected.dropna(subset=["_onset"]).copy()
    selected = selected.sort_values(by=["_trial_num", "_type_order", "_onset"]).reset_index(drop=True)

    T = num_trs
    box = []
    names = []
    meta_rows = []
    item_counter = defaultdict(int)

    def get_field(row: pd.Series, key: str, default=np.nan):
        return row[key] if (key in row.index and pd.notna(row[key])) else default

    for i, row in selected.iterrows():
        try:
            on = float(row["onset_time"])
            off = float(row["offset_time"])
        except Exception:
            continue
        if not np.isfinite(on) or not np.isfinite(off):
            continue

        a = int(np.floor(on / tr))
        b = int(np.ceil(off / tr))
        a = max(0, min(a, T))
        b = max(0, min(b, T))
        if b <= a:
            continue

        vec = np.zeros((T,), dtype=np.float32)
        vec[a:b] = 1.0
        box.append(vec)

        trial_num = _coerce_trial_number(get_field(row, "trialNumber", i + 1), i + 1)
        typ = str(row["type"]).strip().lower()
        key = (trial_num, typ)
        item_counter[key] += 1
        item_idx = item_counter[key]
        try:
            names.append(f"trial{int(trial_num):03d}_item{item_idx:02d}_{typ}")
        except Exception:
            names.append(f"trial{i:03d}_item{item_idx:02d}_{typ}")

        meta_rows.append(
            {
                "trial_index": len(names) - 1,
                "item_index": item_idx,
                "trialNumber": trial_num,
                "type": typ,
                "onset_time": on,
                "offset_time": off,
                "is_correct": get_field(row, "is_correct"),
                "stim_order": get_field(row, "stim_order"),
                "location": get_field(row, "locmod"),
                "category": get_field(row, "ctgmod"),
                "object": get_field(row, "objmod"),
            }
        )

    if not box:
        empty_cols = [
            "trial_index",
            "item_index",
            "trialNumber",
            "type",
            "onset_time",
            "offset_time",
            "is_correct",
            "stim_order",
            "location",
            "category",
            "object",
        ]
        return np.zeros((T, 0), dtype=np.float32), [], pd.DataFrame(columns=empty_cols)

    R_box = np.stack(box, axis=1)
    h = spm_hrf(tr, oversampling=1)
    R_hrf = np.zeros_like(R_box, dtype=np.float32)
    for k in range(R_box.shape[1]):
        conv = np.convolve(R_box[:, k], h)
        R_hrf[:, k] = conv[:T]

    # Reorder columns to interleave epochs by (trial, item, type):
    # trial001_item01_encoding, trial001_item01_delay, trial001_item02_encoding, ...
    type_order = {"encoding": 0, "delay": 1}
    sort_keys = []
    for name in names:
        parts = name.split("_")
        trial_num = int(parts[0].replace("trial", ""))
        item_num = int(parts[1].replace("item", ""))
        typ = parts[2]
        sort_keys.append((trial_num, item_num, type_order.get(typ, 99)))

    sorted_indices = sorted(range(len(names)), key=lambda idx: sort_keys[idx])
    R_hrf = R_hrf[:, sorted_indices]
    names = [names[idx] for idx in sorted_indices]
    meta_rows_reordered = [meta_rows[idx] for idx in sorted_indices]
    for new_idx, row in enumerate(meta_rows_reordered):
        row["trial_index"] = new_idx

    trial_info = pd.DataFrame(meta_rows_reordered).reset_index(drop=True)
    return R_hrf, names, trial_info


def preprocess_y_fixed_psc(Y_full: np.ndarray, keep: np.ndarray, psc_eps: float):
    mu = np.mean(Y_full.astype(np.float64), axis=0, keepdims=True)
    Y = Y_full[keep, :].astype(np.float64)
    safe = np.abs(mu) >= float(psc_eps)
    Yproc = 100.0 * (Y - mu) / np.where(safe, mu, 1.0)
    Yproc[:, ~safe.squeeze()] = 0.0
    return Yproc.astype(np.float32), mu.astype(np.float32), int(np.sum(~safe))


def estimate_rho_from_resid(resid_run: np.ndarray) -> float:
    e = resid_run - resid_run.mean(axis=0, keepdims=True)
    num = float(np.sum(e[1:] * e[:-1]))
    den = float(np.sum(e[:-1] * e[:-1]) + 1e-12)
    return float(np.clip(num / den, -0.99, 0.99))


def whiten_ar1(X: np.ndarray, Y: np.ndarray, rho: float):
    Xw = X.astype(np.float64, copy=True)
    Yw = Y.astype(np.float64, copy=True)
    Xw[1:] -= rho * Xw[:-1]
    Yw[1:] -= rho * Yw[:-1]
    return Xw, Yw


def fit_glm_with_ar1(X: np.ndarray, Y: np.ndarray):
    reg0 = LinearRegression(fit_intercept=False).fit(X, Y)
    resid0 = Y - reg0.predict(X)
    rho = estimate_rho_from_resid(resid0)
    Xw, Yw = whiten_ar1(X, Y, rho)
    B, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    return B.T.astype(np.float32), float(rho)


def fit_glm_with_ols(X: np.ndarray, Y: np.ndarray, ridge: float = 0.0):
    X64 = X.astype(np.float64, copy=False)
    Y64 = Y.astype(np.float64, copy=False)
    if ridge > 0.0:
        K = X64.shape[1]
        B = np.linalg.solve(X64.T @ X64 + ridge * np.eye(K), X64.T @ Y64)
    else:
        B, _, _, _ = np.linalg.lstsq(X64, Y64, rcond=None)
    return B.T.astype(np.float32)


def fit_model(X: np.ndarray, Y: np.ndarray, estimator: str, ridge: float):
    if estimator == "wls_ar1":
        betas, rho = fit_glm_with_ar1(X, Y)
        return betas, rho, "WLS_AR1"
    betas = fit_glm_with_ols(X, Y, ridge=ridge)
    return betas, 0.0, "OLS"


def compute_vif(X_task: np.ndarray, task_names: list[str]):
    vifs = {}
    k = X_task.shape[1]
    for i in range(k):
        y_i = X_task[:, i]
        X_others = np.delete(X_task, i, axis=1)
        if X_others.shape[1] == 0:
            vifs[task_names[i]] = 1.0
            continue
        coef, _, _, _ = np.linalg.lstsq(X_others, y_i, rcond=None)
        y_hat = X_others @ coef
        ss_res = np.sum((y_i - y_hat) ** 2)
        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        vifs[task_names[i]] = float(1.0 / (1.0 - r2 + 1e-12))
    return vifs


def adjusted_r2(r2: np.ndarray, T: int, K: int) -> np.ndarray:
    """R2_adj = 1 - (1 - R2) * (T - 1) / (T - K - 1)."""
    denom = T - K - 1
    if denom <= 0:
        return np.full_like(r2, np.nan, dtype=np.float32)
    out = 1.0 - (1.0 - r2.astype(np.float64)) * (T - 1.0) / float(denom)
    return out.astype(np.float32)


def save_r2_histogram(
    path: Path,
    subj: str,
    ses: str,
    run: str,
    task: str,
    r2_full,
    r2_noise,
    r2_task,
    r2_full_adj,
    r2_noise_adj,
    r2_task_adj,
    valid_mask,
    T_kept: int,
    K_full: int,
    K_noise: int,
    K_task: int,
):
    vals_full = r2_full[valid_mask]
    vals_noise = r2_noise[valid_mask]
    vals_task = r2_task[valid_mask]
    vals_full_adj = r2_full_adj[valid_mask]
    vals_noise_adj = r2_noise_adj[valid_mask]
    vals_task_adj = r2_task_adj[valid_mask]

    plt.figure(figsize=(10, 6))
    plt.hist(vals_full_adj, bins=50, range=(-0.5, 1.0), alpha=0.6, color="C0", label="r2_full_adj")
    plt.hist(vals_noise_adj, bins=50, range=(-0.5, 1.0), alpha=0.6, color="C1", label="r2_noise_adj")
    plt.hist(vals_task_adj, bins=50, range=(-0.5, 1.0), alpha=0.6, color="C2", label="r2_task_adj")
    plt.hist(
        vals_full,
        bins=50,
        range=(-0.5, 1.0),
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color="C0",
        label="r2_full",
    )
    plt.hist(
        vals_noise,
        bins=50,
        range=(-0.5, 1.0),
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color="C1",
        label="r2_noise",
    )
    plt.hist(
        vals_task,
        bins=50,
        range=(-0.5, 1.0),
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color="C2",
        label="r2_task",
    )
    for vals, color in [(vals_full_adj, "C0"), (vals_noise_adj, "C1"), (vals_task_adj, "C2")]:
        if vals.size > 0:
            plt.axvline(float(np.nanmedian(vals)), linestyle="--", color=color, linewidth=1.5)
    plt.xlim(-0.5, 1.0)
    plt.xlabel("R2")
    plt.ylabel("Vertex count")
    plt.title(
        f"{subj} {ses} {run} {task} — R2 distributions "
        f"(solid=adjusted, dashed=unadjusted)"
    )
    plt.text(
        0.01,
        0.99,
        f"T={T_kept}, K_full={K_full}, K_noise={K_noise}, K_task={K_task}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_example_vertices(path: Path, Yproc: np.ndarray, yhat_full: np.ndarray, r2_task: np.ndarray, valid_mask: np.ndarray):
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid vertices available", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return

    rank_idx = valid_idx[np.argsort(r2_task[valid_idx])[::-1]]
    top_idx = rank_idx[: min(6, rank_idx.size)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()
    trs = np.arange(Yproc.shape[0])

    for i, ax in enumerate(axes):
        if i >= top_idx.size:
            ax.axis("off")
            continue
        vidx = int(top_idx[i])
        ax.plot(trs, Yproc[:, vidx], color="C0", linewidth=1.0, label="actual")
        ax.plot(trs, yhat_full[:, vidx], color="C1", linewidth=1.0, label="predicted")
        ax.set_title(f"vertex {vidx}, r2_task={float(r2_task[vidx]):.3f}")
        ax.set_xlabel("time (TRs)")
        ax.set_ylabel("PSC (%)")
        if i == 0:
            ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_design_corr_heatmap(path: Path, X_full: np.ndarray, col_names: list[str]):
    X = X_full.astype(np.float64)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    fig_size = max(8, min(24, 0.25 * len(col_names)))
    plt.figure(figsize=(fig_size, fig_size))
    im = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(col_names))
    plt.xticks(ticks, col_names, rotation=90, fontsize=6)
    plt.yticks(ticks, col_names, fontsize=6)
    plt.title("Design matrix column correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_vif_bar(path: Path, vif_dict: dict[str, float]):
    names = list(vif_dict.keys())
    vals = np.array([vif_dict[n] for n in names], dtype=np.float64)
    colors = ["red" if v > 10.0 else "green" for v in vals]

    width = max(10, min(40, 0.25 * len(names)))
    plt.figure(figsize=(width, 5))
    plt.bar(np.arange(len(names)), vals, color=colors)
    plt.axhline(10.0, linestyle="--", color="black", linewidth=1.5)
    plt.xticks(np.arange(len(names)), names, rotation=90, fontsize=6)
    plt.ylabel("VIF")
    plt.title("VIF per task regressor")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def fetch_nilearn_fslr32k_inflated_surfaces():
    try:
        fsLR = datasets.fetch_surf_fsaverage(mesh="fsLR32k")
        lh_inflated = Path(fsLR["infl_left"])
        rh_inflated = Path(fsLR["infl_right"])
        return lh_inflated, rh_inflated, "fsLR32k", None
    except Exception as e:
        # Backward-compatible fallback for nilearn versions without fsLR32k.
        fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage6")
        lh_inflated = Path(fsavg["infl_left"])
        rh_inflated = Path(fsavg["infl_right"])
        return lh_inflated, rh_inflated, "fsaverage6_fallback", str(e)


def _split_hemispheres(arr: np.ndarray):
    return arr[:32492], arr[32492:64984]


def _mesh_vertex_count(mesh_path: Path) -> int:
    gii = nib.load(str(mesh_path))
    return int(gii.darrays[0].data.shape[0])


def _resample_metric_to_mesh_vertices(metric: np.ndarray, n_vertices: int) -> np.ndarray:
    if metric.shape[0] == n_vertices:
        return metric
    x_old = np.linspace(0.0, 1.0, metric.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, n_vertices, dtype=np.float64)
    finite = np.isfinite(metric)
    if np.sum(finite) < 2:
        return np.full((n_vertices,), np.nan, dtype=np.float32)
    y_new = np.interp(x_new, x_old[finite], metric[finite].astype(np.float64))
    return y_new.astype(np.float32)


def mask_invalid_vertices(metric: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.full((64984,), np.nan, dtype=np.float32)
    if metric.shape[0] == 64984 and valid_mask.shape[0] == 64984:
        out[valid_mask] = metric[valid_mask]
    else:
        n = min(metric.shape[0], valid_mask.shape[0], 64984)
        m = valid_mask[:n]
        tmp = out[:n]
        tmp[m] = metric[:n][m]
        out[:n] = tmp
    return out


def save_r2_surface_grid(
    path: Path,
    lh_surf: Path,
    rh_surf: Path,
    r2_full_adj: np.ndarray,
    r2_noise_adj: np.ndarray,
    r2_task_adj: np.ndarray,
    valid_mask: np.ndarray,
):
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), subplot_kw={"projection": "3d"})
    rows = [("r2_full_adj", r2_full_adj), ("r2_noise_adj", r2_noise_adj), ("r2_task_adj", r2_task_adj)]
    n_lh = _mesh_vertex_count(lh_surf)
    n_rh = _mesh_vertex_count(rh_surf)
    for i, (label, arr) in enumerate(rows):
        mean_val = float(np.nanmean(arr[valid_mask])) if np.any(valid_mask) else np.nan
        arr_64k = mask_invalid_vertices(arr, valid_mask)
        lh_map, rh_map = _split_hemispheres(arr_64k)
        lh_map = _resample_metric_to_mesh_vertices(lh_map, n_lh)
        rh_map = _resample_metric_to_mesh_vertices(rh_map, n_rh)
        plotting.plot_surf_stat_map(
            surf_mesh=str(lh_surf),
            stat_map=lh_map,
            hemi="left",
            view="lateral",
            cmap="hot",
            vmin=0,
            vmax=1,
            colorbar=True,
            axes=axes[i, 0],
            title=f"{label} (mean={mean_val:.3f}) — LH",
        )
        plotting.plot_surf_stat_map(
            surf_mesh=str(rh_surf),
            stat_map=rh_map,
            hemi="right",
            view="lateral",
            cmap="hot",
            vmin=0,
            vmax=1,
            colorbar=True,
            axes=axes[i, 1],
            title=f"{label} (mean={mean_val:.3f}) — RH",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_r2_task_adj_surface_both_views(
    path: Path,
    lh_surf: Path,
    rh_surf: Path,
    metric: np.ndarray,
    valid_mask: np.ndarray,
    subj: str,
    ses: str,
    run: str,
    task: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"})
    metric_64k = mask_invalid_vertices(metric, valid_mask)
    lh_map, rh_map = _split_hemispheres(metric_64k)
    n_lh = _mesh_vertex_count(lh_surf)
    n_rh = _mesh_vertex_count(rh_surf)
    lh_map = _resample_metric_to_mesh_vertices(lh_map, n_lh)
    rh_map = _resample_metric_to_mesh_vertices(rh_map, n_rh)
    plotting.plot_surf_stat_map(str(lh_surf), lh_map, hemi="left", view="lateral", cmap="hot", vmin=0, vmax=1, colorbar=True, axes=axes[0, 0], title="r2_task_adj — LH lateral")
    plotting.plot_surf_stat_map(str(rh_surf), rh_map, hemi="right", view="lateral", cmap="hot", vmin=0, vmax=1, colorbar=True, axes=axes[0, 1], title="r2_task_adj — RH lateral")
    plotting.plot_surf_stat_map(str(lh_surf), lh_map, hemi="left", view="medial", cmap="hot", vmin=0, vmax=1, colorbar=True, axes=axes[1, 0], title="r2_task_adj — LH medial")
    plotting.plot_surf_stat_map(str(rh_surf), rh_map, hemi="right", view="medial", cmap="hot", vmin=0, vmax=1, colorbar=True, axes=axes[1, 1], title="r2_task_adj — RH medial")
    fig.suptitle(f"r2_task_adj — {subj} {ses} {run} {task}", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = get_args()
    subj = args.subj
    tr = args.tr
    tmask = args.tmask
    include_types = [t.lower() for t in args.include_types]
    correct_only = args.correct_only

    fmri_root_dir = Path(args.fmri_root) / subj
    confounds_root_dir = Path(args.conf_root) / subj
    events_root_dir = Path(args.events_root) / subj
    output_dir = default_output_root(Path(args.out_root), correct_only, subj)
    sanity_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/glm_sanity") / subj

    global_valid_mask = load_valid_vertex_mask().astype(bool)
    nilearn_surface_ready = False
    lh_inflated = None
    rh_inflated = None
    try:
        lh_inflated, rh_inflated, surf_source, surf_err = fetch_nilearn_fslr32k_inflated_surfaces()
        nilearn_surface_ready = True
        if surf_source == "fsLR32k":
            print(f"[nilearn] fetched fsLR32k inflated surfaces: LH={lh_inflated}, RH={rh_inflated}")
        else:
            print(
                "[nilearn] fsLR32k unavailable; using fsaverage5 fallback: "
                f"LH={lh_inflated}, RH={rh_inflated}; fsLR32k error={surf_err}"
            )
    except Exception as e:
        print(f"[nilearn] failed to fetch fsLR32k inflated surfaces: {e}")

    sessions = discover_sessions(fmri_root_dir)
    if args.sessions:
        sessions = [s for s in sessions if s in set(args.sessions)]
    print(f"Sessions: {sessions}")

    for task_name in args.tasks:
        for ses in sessions:
            runs = discover_runs_from_timeseries(subj, ses, task_name, fmri_root_dir)
            runs = select_runs(runs, args.runs)
            print(f"[{task_name} | {ses}] runs discovered from timeseries: {runs}")
            if not runs:
                continue

            for run_label in runs:
                aliases = run_aliases(run_label)
                ts_candidates = [
                    fmri_root_dir / ses / f"{subj}_{ses}_task-{task_name}_{rl}_space-Glasser64k_bold_sm4.dtseries.nii"
                    for rl in aliases
                ]
                ev_candidates = [
                    events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{rl}_events.tsv"
                    for rl in aliases
                ]
                cf_candidates = [
                    confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{rl}_desc-confounds_timeseries.tsv"
                    for rl in aliases
                ]

                ts_path = first_existing(ts_candidates)
                ev_path = first_existing(ev_candidates)
                cf_path = first_existing(cf_candidates)
                if ts_path is None or ev_path is None or cf_path is None:
                    print(f"[skip] Missing one or more files for {ses} {run_label} {task_name}")
                    continue

                matched_label = None
                for rl, p in zip(aliases, ts_candidates):
                    if p == ts_path:
                        matched_label = rl
                        break
                run_used = matched_label or run_label
                base_name = f"{subj}_{ses}_task-{task_name}_{run_used}"
                print(f"[run] {base_name}")

                target_subdir = output_dir / ses / "func"
                h5_file = target_subdir / f"{base_name}_betas.h5"
                design_csv = target_subdir / f"{base_name}_design.csv"
                design_matrix_csv = target_subdir / f"{base_name}_design_matrix.csv"

                if h5_file.exists() and not args.overwrite:
                    print(f"[skip] exists: {h5_file}")
                    continue

                Y_full = nib.load(str(ts_path)).get_fdata(dtype=np.float32)
                if Y_full.ndim != 2:
                    raise ValueError(f"Timeseries must be 2D (T x P), got {Y_full.shape}")
                T, P = Y_full.shape

                df_confounds = pd.read_csv(cf_path, sep="\t")
                C = np.nan_to_num(glm_confounds_construction(df_confounds), nan=0.0).astype(np.float32)
                conf_names = [f"conf_{i}" for i in range(C.shape[1])]
                if C.shape[0] != T:
                    raise ValueError(f"Confounds rows ({C.shape[0]}) != BOLD rows ({T})")
                conf_keep = [j for j in range(C.shape[1]) if np.std(C[:, j]) >= 1e-8]
                C = C[:, conf_keep]
                conf_names = [conf_names[j] for j in conf_keep]

                df_events = clean_events(pd.read_csv(ev_path, sep="\t"))
                R_task, trial_names, trial_info = build_trialwise_regressors(
                    df_events=df_events,
                    num_trs=T,
                    tr=tr,
                    include_types=include_types,
                    correct_only=correct_only,
                )
                if R_task.shape[1] == 0:
                    print(f"[skip] no selected events in {ev_path}")
                    continue

                D, drift_names = per_run_drift_and_intercept(T, tr, high_pass_sec=args.high_pass_sec)

                keep = np.ones((T,), dtype=bool)
                keep[:tmask] = False
                Yproc, psc_baseline, n_low_mean = preprocess_y_fixed_psc(Y_full, keep=keep, psc_eps=args.psc_eps)
                Ck = C[keep, :]
                Dk = D[keep, :]
                Rk = R_task[keep, :]

                X_full = np.hstack([Ck, Dk, Rk]).astype(np.float32)
                X_noise = np.hstack([Ck, Dk]).astype(np.float32)
                design_col_names = conf_names + drift_names + list(trial_names)

                K_task = Rk.shape[1]
                task_col_start = X_full.shape[1] - K_task
                task_col_end = X_full.shape[1]

                betas_full, rho_full, estimator_name = fit_model(X_full, Yproc, args.estimator, args.ridge)
                betas_noise, rho_noise, _ = fit_model(X_noise, Yproc, args.estimator, args.ridge)
                betas_task = betas_full[:, task_col_start:task_col_end]

                yhat_full = (X_full @ betas_full.T).astype(np.float32)
                yhat_noise = (X_noise @ betas_noise.T).astype(np.float32)
                SS_total = np.sum((Yproc - Yproc.mean(axis=0, keepdims=True)) ** 2, axis=0)
                SS_total = np.where(SS_total < 1e-12, np.nan, SS_total)
                SS_res_full = np.sum((Yproc - yhat_full) ** 2, axis=0)
                SS_res_noise = np.sum((Yproc - yhat_noise) ** 2, axis=0)
                r2_full = (1.0 - SS_res_full / SS_total).astype(np.float32)
                r2_noise = (1.0 - SS_res_noise / SS_total).astype(np.float32)
                r2_task = (r2_full - r2_noise).astype(np.float32)

                T_kept = Yproc.shape[0]
                K_full = X_full.shape[1]
                K_noise = X_noise.shape[1]
                K_task = K_full - K_noise
                r2_full_adj = adjusted_r2(r2_full, T_kept, K_full)
                r2_noise_adj = adjusted_r2(r2_noise, T_kept, K_noise)
                r2_task_adj = (r2_full_adj - r2_noise_adj).astype(np.float32)

                if args.compute_vif:
                    vif_dict = compute_vif(Rk, trial_names)
                    vif_vals = np.array(list(vif_dict.values()), dtype=np.float32)
                    vif_max = float(np.nanmax(vif_vals)) if vif_vals.size else 0.0
                    vif_mean = float(np.nanmean(vif_vals)) if vif_vals.size else 0.0
                else:
                    vif_dict = {}
                    vif_max = 0.0
                    vif_mean = 0.0

                valid_mask = np.ones((P,), dtype=bool)
                if global_valid_mask.shape[0] == P:
                    valid_mask = global_valid_mask.copy()
                else:
                    print(f"[warn] valid vertex mask length {global_valid_mask.shape[0]} != P {P}; using all vertices.")

                sanity_dir = sanity_root / ses / run_used
                sanity_dir.mkdir(parents=True, exist_ok=True)
                fig_r2_hist = sanity_dir / "r2_histogram.png"
                fig_examples = sanity_dir / "example_vertices.png"
                fig_corr = sanity_dir / "design_matrix_correlation.png"
                fig_vif = sanity_dir / "vif_task_regressors.png"
                fig_r2_surface = sanity_dir / "r2_surface.png"
                fig_r2_task_adj_surface = sanity_dir / "r2_task_adj_surface_both_views.png"

                save_r2_histogram(
                    fig_r2_hist,
                    subj,
                    ses,
                    run_used,
                    task_name,
                    r2_full,
                    r2_noise,
                    r2_task,
                    r2_full_adj,
                    r2_noise_adj,
                    r2_task_adj,
                    valid_mask,
                    T_kept=T_kept,
                    K_full=K_full,
                    K_noise=K_noise,
                    K_task=K_task,
                )
                save_example_vertices(fig_examples, Yproc, yhat_full, r2_task, valid_mask)
                save_design_corr_heatmap(fig_corr, X_full, design_col_names)
                if args.compute_vif and vif_dict:
                    save_vif_bar(fig_vif, vif_dict)
                else:
                    print("[skip] VIF not computed (use --compute_vif to enable)")
                if args.save_surface_plots and nilearn_surface_ready and lh_inflated is not None and rh_inflated is not None:
                    save_r2_surface_grid(
                        fig_r2_surface,
                        lh_inflated,
                        rh_inflated,
                        r2_full_adj,
                        r2_noise_adj,
                        r2_task_adj,
                        valid_mask,
                    )
                    save_r2_task_adj_surface_both_views(
                        fig_r2_task_adj_surface,
                        lh_inflated,
                        rh_inflated,
                        r2_task_adj,
                        valid_mask,
                        subj,
                        ses,
                        run_used,
                        task_name,
                    )
                    print(f"[save] sanity_r2_surface: {fig_r2_surface}")
                    print(f"[save] sanity_r2_task_adj_surface: {fig_r2_task_adj_surface}")
                else:
                    print("[skip] surface plots not requested (use --save_surface_plots to enable)")

                yhat_to_save = yhat_full if args.save_yhat else None
                resid_to_save = (Yproc - yhat_full).astype(np.float32) if args.save_resid else None

                target_subdir.mkdir(parents=True, exist_ok=True)
                trial_info.sort_values(by="trial_index").to_csv(design_csv, index=False)
                pd.DataFrame(X_full, columns=design_col_names).to_csv(design_matrix_csv, index=False)

                with h5py.File(h5_file, "w") as h5f:
                    h5f.create_dataset("betas_full", data=betas_full.astype(np.float32))
                    h5f.create_dataset("betas", data=betas_task.astype(np.float32))
                    h5f.create_dataset("valid_mask", data=valid_mask.astype(np.uint8))
                    h5f.create_dataset("design_matrix", data=X_full.astype(np.float32))

                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    h5f.create_dataset(
                        "design_col_names",
                        data=np.array(design_col_names, dtype=object),
                        dtype=str_dtype,
                    )
                    h5f.create_dataset("r2_full", data=r2_full)
                    h5f.create_dataset("r2_noise", data=r2_noise)
                    h5f.create_dataset("r2_task", data=r2_task)
                    h5f.create_dataset("r2_full_adj", data=r2_full_adj)
                    h5f.create_dataset("r2_noise_adj", data=r2_noise_adj)
                    h5f.create_dataset("r2_task_adj", data=r2_task_adj)
                    h5f.create_dataset("psc_baseline_mean", data=psc_baseline.astype(np.float32))
                    if yhat_to_save is not None:
                        h5f.create_dataset("yhat", data=yhat_to_save, compression="gzip")
                    if resid_to_save is not None:
                        h5f.create_dataset("resid", data=resid_to_save, compression="gzip")

                    h5f.attrs.create("task_regressor_names", np.array(trial_names, dtype=object), dtype=str_dtype)
                    h5f.attrs["regressor_level"] = "trial"
                    h5f.attrs["task_col_start"] = int(task_col_start)
                    h5f.attrs["task_col_end"] = int(task_col_end)
                    h5f.attrs["tmask_dropped"] = int(tmask)
                    h5f.attrs["high_pass_sec"] = float(args.high_pass_sec)
                    h5f.attrs["order_in_X"] = "[confounds, drifts+intercept, task(trials)]"
                    h5f.attrs["estimator"] = estimator_name
                    h5f.attrs["rho_ar1"] = float(rho_full)
                    h5f.attrs["rho_ar1_noise_model"] = float(rho_noise)
                    h5f.attrs["vif_max"] = float(vif_max)
                    h5f.attrs["vif_mean"] = float(vif_mean)
                    h5f.attrs["psc_eps"] = float(args.psc_eps)
                    h5f.attrs["psc_equation"] = "100 * (Y - mean(Y_full)) / mean(Y_full)"
                    h5f.attrs["n_low_mean_psc_units"] = int(n_low_mean)
                    h5f.attrs["n_valid_grayordinates"] = int(np.sum(valid_mask))
                    h5f.attrs["n_total_grayordinates"] = int(P)
                    h5f.attrs["smoothing"] = "external_wb_command_cifti_smoothing_4mm_FWHM"
                    h5f.attrs["T_kept"] = int(T_kept)
                    h5f.attrs["K_full"] = int(K_full)
                    h5f.attrs["K_noise"] = int(K_noise)
                    h5f.attrs["K_task"] = int(K_task)
                    h5f.attrs["adj_r2_note"] = (
                        "R2_adj = 1 - (1-R2)*(T-1)/(T-K-1); "
                        "r2_task_adj = r2_full_adj - r2_noise_adj"
                    )

                print(f"[save] h5: {h5_file}")
                print(f"[save] design: {design_csv}")
                print(f"[save] design_matrix: {design_matrix_csv}")
                print(f"[save] sanity_r2_hist: {fig_r2_hist}")
                print(f"[save] sanity_examples: {fig_examples}")
                print(f"[save] sanity_corr: {fig_corr}")
                if args.compute_vif and vif_dict:
                    print(f"[save] sanity_vif: {fig_vif}")
                print(
                    "[stats] "
                    f"mean_r2_full={float(np.nanmean(r2_full[valid_mask])):.6f} "
                    f"mean_r2_noise={float(np.nanmean(r2_noise[valid_mask])):.6f} "
                    f"mean_r2_task={float(np.nanmean(r2_task[valid_mask])):.6f} "
                    f"mean_r2_task_adj={float(np.nanmean(r2_task_adj[valid_mask])):.6f} "
                    f"vif_max={vif_max:.6f} vif_mean={vif_mean:.6f}"
                )
                print(
                    f"[R2 adjusted] full_adj: {float(np.nanmean(r2_full_adj[valid_mask])):.3f}  "
                    f"noise_adj: {float(np.nanmean(r2_noise_adj[valid_mask])):.3f}  "
                    f"task_adj: {float(np.nanmean(r2_task_adj[valid_mask])):.3f}"
                )
                print(
                    f"[Degrees of freedom] T={T_kept}, K_full={K_full}, K_noise={K_noise}, K_task={K_task}"
                )

    print("[done]")


if __name__ == "__main__":
    main()
