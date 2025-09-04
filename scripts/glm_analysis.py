#!/usr/bin/env python3
"""
Per-run **trial-level** GLM with cosine drifts + AR(1), HRF-convolved *one regressor per event row*.
Saves the **FULL design matrix** + names so you can do contrasts and design visualization later.

Outputs per run (under out_root/.../sub-XX/ses-YY/func/):
  - <base>_betas.h5
      datasets:
        betas                 (P x K_task)     # trial-wise betas, in the same order as task_regressor_names
        XtX_inv               (K x K)          # for full X
        sigma2                (P,)             # residual variance per parcel
        residuals_shape       (2,)             # (T_kept, P)
        all_regressors_shape  (2,)             # (T_kept, K)
        design_matrix         (T_kept x K)     # FULL X = [confounds, drifts+intercept, task(trials)]
        design_col_names      (K,)             # HDF5 string dtype
      attrs:
        task_regressor_names  [list of trial column names]
        regressor_level       "trial"
        task_col_start, task_col_end (int)     # inclusive-exclusive slice in X for task columns
        dof, rho_ar1, tmask_dropped, high_pass_sec
        order_in_X            "[confounds, drifts+intercept, task(trials)]"
  - <base>_design.csv            # per-trial metadata (trialNumber, type, etc.) in the same column order as betas
  - <base>_design_matrix.csv     # human-friendly CSV of FULL design matrix

Example
-------
python glm_analysis.py \
  --subj sub-01 \
  --tasks 1back  \
  --include_types encoding delay \
  --out_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay \
  --overwrite
"""
from __future__ import annotations
import re
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import nibabel as nib

from nilearn.glm.first_level import spm_hrf, make_first_level_design_matrix

from utils import glm_confounds_construction, standardize_run_label

# -----------------------
# CLI / hyperparams
# -----------------------

def get_args():
    p = argparse.ArgumentParser(description="Per-run trial-level GLM with cosine drifts + AR(1), HRF-convolved.")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames to drop at run start")
    p.add_argument("--correct_only", action="store_true", help="Use only correct trials")
    p.add_argument("--tasks", nargs="+", default=["ctxdm"], help="Task names to process")
    p.add_argument("--include_types", nargs="+", default=["encoding","delay"], help="Event types to include as trial regressors (lowercased)")
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled")
    p.add_argument("--conf_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--events_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data")
    p.add_argument("--high_pass_sec", type=float, default=128.0, help="Cosine high-pass cutoff in seconds")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

# -----------------------
# Utilities
# -----------------------

def default_output_root(out_root_base: Path, correct_only: bool, subj: str) -> Path:
    return (out_root_base / ("correct_trial_level_betas" if correct_only else "trial_level_betas") / subj)


def discover_sessions(fmri_root_dir: Path):
    pattern = re.compile(r".*\d$")
    return sorted([p.name for p in fmri_root_dir.iterdir() if p.is_dir() and pattern.match(p.name)])


def discover_runs_for_task_session(subj, ses, task, fmri_root_dir: Path):
    pattern = f"{subj}_{ses}_task-{task}_*_space-Glasser64k_bold.dtseries.nii"
    matching_files = list((fmri_root_dir / ses).glob(pattern))
    run_pattern = re.compile(
        rf"{re.escape(subj)}_{re.escape(ses)}_task-{re.escape(task)}_(.+?)_space-Glasser64k_bold\.dtseries\.nii"
    )
    runs = []
    for f in matching_files:
        m = run_pattern.match(f.name)
        if m:
            runs.append(m.group(1))
    return sorted(runs)


def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    return df


# -----------------------
# Regressor builders (trial-wise)
# -----------------------

def build_trialwise_regressors(
    df_events: pd.DataFrame,
    num_trs: int,
    tr: float,
    include_types: list[str],
    correct_only: bool = False,
):
    """
    Build one HRF-convolved regressor per *event row* whose `type` is in include_types.

    If `correct_only` is True and `is_correct` exists, EXCLUDE rows where is_correct is:
      - False (boolean)
      - 'false' (any case)
      - 'b' (any case)
    All other values are accepted (including NaN).

    TR-binned boxcar:
      onset_tr = floor(onset_time / TR)
      offset_tr = floor(offset_time / TR)
      add 1.0 on [onset_tr, offset_tr) clipped to [0, T)
    """
    req_cols = {"onset_time", "offset_time", "type"}
    if not req_cols.issubset(df_events.columns):
        missing = ", ".join(sorted(req_cols - set(df_events.columns)))
        raise ValueError(f"Events missing required columns: {missing}")

    # Type filtering
    types = df_events["type"].astype(str).str.strip().str.lower()
    mask_type = types.isin([t.lower() for t in include_types])

    # Correct-only filtering per your rule
    if correct_only and "is_correct" in df_events.columns:
        def is_negative(v):
            # exclude only False / 'false' / 'b'
            if isinstance(v, (bool, np.bool_)):
                return (v is False) or (v == False)
            if pd.isna(v):
                return False  # NaN is acceptable
            sv = str(v).strip().lower()
            return sv in {"false", "b"}

        mask_correct = ~df_events["is_correct"].apply(is_negative)
        mask = mask_type & mask_correct
    else:
        mask = mask_type

    df_sel = df_events.loc[mask].reset_index(drop=True)

    T = num_trs
    box = []
    names = []
    meta_rows = []

    # Safe getter
    def g(row: pd.Series, key, default=np.nan):
        return row[key] if (key in row.index and pd.notna(row[key])) else default

    for i, row in df_sel.iterrows():
        on = float(row["onset_time"]) if pd.notna(row["onset_time"]) else np.nan
        off = float(row["offset_time"]) if pd.notna(row["offset_time"]) else np.nan
        if not np.isfinite(on) or not np.isfinite(off):
            continue

        a = int(np.floor(on / tr))
        b = int(np.floor(off / tr))
        a = max(0, min(a, T))
        b = max(0, min(b, T))

        vec = np.zeros((T,), dtype=np.float32)
        if b > a:
            vec[a:b] = 1.0
            box.append(vec)
        else:
            continue  # skip zero-length after binning

        tn = g(row, "trialNumber", i)
        typ = str(row["type"]).strip().lower()
        try:
            names.append(f"trial{int(tn):03d}_{typ}")
        except Exception:
            names.append(f"trial{i:03d}_{typ}")

        # Keep original is_correct value (no coercion) for bookkeeping
        meta_rows.append({
            "trial_index": len(names) - 1,
            "trialNumber": tn,
            "type": typ,
            "onset_time": on,
            "offset_time": off,
            "is_correct": g(row, "is_correct"),
            "stim_order": g(row, "stim_order"),
            "location": g(row, "locmod"),
            "category": g(row, "ctgmod"),
            "object": g(row, "objmod"),
        })

    if len(box) == 0:
        empty_cols = ["trial_index","trialNumber","type","onset_time","offset_time","is_correct","stim_order","location","category","object"]
        return np.zeros((T, 0), dtype=np.float32), [], pd.DataFrame(columns=empty_cols)

    R_box = np.stack(box, axis=1)  # (T x K)

    # Convolve with canonical SPM HRF
    h = spm_hrf(tr, oversampling=1)
    R_hrf = np.zeros_like(R_box, dtype=np.float32)
    for k in range(R_box.shape[1]):
        tmp = np.convolve(R_box[:, k], h)
        R_hrf[:, k] = tmp[:T]

    trial_info = pd.DataFrame(meta_rows).sort_values(by="trial_index").reset_index(drop=True)
    return R_hrf, names, trial_info


# -----------------------
# Drifts + intercept (nilearn-style cosine)
# -----------------------

def per_run_drift_and_intercept(num_trs: int, tr: float, high_pass_sec: float = 128.0):
    frame_times = np.arange(num_trs) * tr
    dm = make_first_level_design_matrix(
        frame_times,
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


# -----------------------
# AR(1) estimation + whitening
# -----------------------

def estimate_rho_from_resid(resid_run: np.ndarray) -> float:
    e = resid_run - resid_run.mean(axis=0, keepdims=True)
    num = float(np.sum(e[1:] * e[:-1]))
    den = float(np.sum(e[:-1] * e[:-1]) + 1e-12)
    rho = num / den
    return float(np.clip(rho, -0.99, 0.99))


def whiten_ar1(X: np.ndarray, Y: np.ndarray, rho: float):
    Xw = X.copy(); Yw = Y.copy()
    Xw[1:] -= rho * Xw[:-1]
    Yw[1:] -= rho * Yw[:-1]
    return Xw, Yw


def fit_glm_with_ar1(X: np.ndarray, Y: np.ndarray):
    # Pass 1: quick OLS
    reg0 = LinearRegression().fit(X, Y)
    resid0 = Y - reg0.predict(X)
    rho = estimate_rho_from_resid(resid0)

    # Pass 2: whiten and refit
    Xw, Yw = whiten_ar1(X, Y, rho)
    XtX = Xw.T @ Xw
    ridge = 1e-6 * np.eye(XtX.shape[0], dtype=np.float32)
    XtX_inv = np.linalg.inv(XtX + ridge)
    B = XtX_inv @ (Xw.T @ Yw)     # (K x P)
    Yhat = Xw @ B                 # (T x P)
    resid = Yw - Yhat             # (T x P)

    dof = Xw.shape[0] - np.linalg.matrix_rank(Xw)
    dof = int(max(dof, 1))
    sigma2 = (resid**2).sum(axis=0) / dof

    betas_full = B.T              # (P x K)
    return betas_full, XtX_inv, sigma2, dof, resid


# -----------------------
# Main
# -----------------------

def main():
    args = get_args()

    subj = args.subj
    tr = args.tr
    tmask = args.tmask
    correct_only = args.correct_only
    include_types = [t.lower() for t in args.include_types]

    fmri_root_dir = Path(args.fmri_root) / subj
    confounds_root_dir = Path(args.conf_root) / subj
    events_root_dir = Path(args.events_root) / subj
    output_dir = default_output_root(Path(args.out_root), correct_only, subj)

    sessions = discover_sessions(fmri_root_dir)
    print("Sessions:", sessions)

    for task_name in args.tasks:
        for ses in sessions:
            runs = discover_runs_for_task_session(subj, ses, task_name, fmri_root_dir)
            print(f"[{task_name} | {ses}] runs: {runs}")

            for run in runs:
                # Paths
                behavioral_file = events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{standardize_run_label(run)}_events.tsv"
                
                if not behavioral_file.exists():
                    print(f"⚠️ Behavioral file {behavioral_file} does not exist, skipping.")
                    continue

                timeseries_file = fmri_root_dir / ses / f"{subj}_{ses}_task-{task_name}_{run}_space-Glasser64k_bold.dtseries.nii"
                confounds_file = confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{run}_desc-confounds_timeseries.tsv"

                if not timeseries_file.exists() or not confounds_file.exists():
                    print(f"⚠️ Missing data for {ses} {run} {task_name}, skipping.")
                    continue

                # Output paths
                rel_behavioral_path = behavioral_file.relative_to(events_root_dir)
                target_subdir = output_dir / rel_behavioral_path.parent
                base_name = behavioral_file.stem.replace("_events", "")
                h5_file = target_subdir / f"{base_name}_betas.h5"
                design_csv = target_subdir / f"{base_name}_design.csv"
                design_matrix_csv = target_subdir / f"{base_name}_design_matrix.csv"

                if h5_file.exists() and not args.overwrite:
                    print(f"[⏩] Skipping {h5_file} — already exists.")
                    continue

                # ---------------- Load data ----------------
                Y = nib.load(str(timeseries_file)).get_fdata(dtype=np.float32)  # (T x P)
                assert not np.isnan(Y).any(), "NaNs in timeseries_data"

                df_confounds = pd.read_csv(confounds_file, sep="\t")
                C_mat = glm_confounds_construction(df_confounds)
                if isinstance(C_mat, pd.DataFrame):
                    conf_names = list(C_mat.columns)
                    C = np.nan_to_num(C_mat.values, nan=0.0).astype(np.float32)
                else:
                    C = np.nan_to_num(C_mat, nan=0.0).astype(np.float32)
                    conf_names = [f"conf_{i}" for i in range(C.shape[1])]

                df_events = pd.read_csv(behavioral_file, sep="\t")
                # print(f"inspection of the length of behavioral_file: {len(df_events)}")
                df_events = clean_events(df_events)
                if correct_only and "is_correct" in df_events.columns:
                    df_events = df_events[~df_events["is_correct"].apply(
                        lambda v: (isinstance(v, (bool, np.bool_)) and v is False) or
                                (isinstance(v, str) and v.strip().lower() in {"false", "b"})
                    )]

                # print(f"inspection of the length of df_events after filtering: {len(df_events)}")

                T = Y.shape[0]
                # print(f"shape of timeseries data: {Y.shape} (T x P)")
                

                # ------------- Build task regressors (trial-wise) -------------
                R_task, trial_names, trial_info = build_trialwise_regressors(df_events, T, tr, include_types, correct_only=correct_only)
                # print(f"shape of trial_info: {len(trial_info)} (K x ?)")
                if R_task.shape[1] == 0:
                    print(f"⚠️ No trials of types {include_types} in {behavioral_file}, skipping.")
                    continue

                # ------------- Drifts + intercept -------------
                D, drift_names = per_run_drift_and_intercept(T, tr, high_pass_sec=args.high_pass_sec)

                # tmask skip (apply to all time-dependent arrays)
                keep = np.ones((T,), dtype=bool)
                keep[:tmask] = False
                # print(f"shape of Y before tmask: {Y.shape}")
                Y = Y[keep, :]
                # print(f"shape of Y after tmask: {Y.shape}")
                C = C[keep, :]
                # print(f"shape of R_task before tmask: {R_task.shape}")
                R_task = R_task[keep, :]
                # print(f"shape of R_task after tmask: {R_task.shape}")
                D = D[keep, :]
                # print(f"shape of D after tmask: {D.shape}")

                # ------------- Assemble X -------------
                X = np.hstack([C, D, R_task]).astype(np.float32)
                Yf = Y.astype(np.float32)
                design_col_names = conf_names + drift_names + list(trial_names)



                # ------------- Fit GLM with AR(1) -------------
                betas_full, XtX_inv, sigma2, dof, resid = fit_glm_with_ar1(X, Yf)
                print(f"shape of design matrix X: {X.shape} (T_kept x K)")
                print(f"shape of betas_full: {betas_full.shape} (P x K)")
                print(f"shape of timeseries data Yf: {Yf.shape} (T_kept x P)")

                # Indices for task columns (last K_task)
                K_task = R_task.shape[1]
                betas_task = betas_full[:, -K_task:]
                print(f"shape of betas_task : {betas_task.shape} (P x K_task)")
                task_col_start = X.shape[1] - K_task
                task_col_end = X.shape[1]

                # ------------- Save outputs -------------
                target_subdir.mkdir(parents=True, exist_ok=True)

                # Per-trial metadata CSV in task order
                # Ensure trial_info matches trial_names order
                trial_info_sorted = trial_info.sort_values(by='trial_index')
                trial_info_sorted.to_csv(design_csv, index=False)

                # Full design matrix CSV
                pd.DataFrame(X, columns=design_col_names).to_csv(design_matrix_csv, index=False)

                # HDF5  (ONLY THE BLOCK BELOW IS MODIFIED TO ALSO SAVE FULL BETAS)
                with h5py.File(h5_file, "w") as h5f:
                    # --- NEW: save full betas for ALL regressors (P x K) ---
                    h5f.create_dataset("betas_full", data=betas_full.astype(np.float32))  # (P x K)

                    # Core outputs (kept as in your original code)
                    h5f.create_dataset("betas", data=betas_task.astype(np.float32))        # (P x K_task)
                    h5f.create_dataset("all_regressors_shape", data=np.array(X.shape, dtype=np.int32))
                    # Stats
                    h5f.create_dataset("XtX_inv", data=XtX_inv.astype(np.float32))          # (K x K)
                    h5f.create_dataset("sigma2", data=sigma2.astype(np.float32))            # (P,)
                    h5f.create_dataset("residuals_shape", data=np.array(resid.shape, dtype=np.int32))
                    # Full design matrix + names
                    h5f.create_dataset("design_matrix", data=X.astype(np.float32))          # (T_kept x K)
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    h5f.create_dataset("design_col_names", data=np.array(design_col_names, dtype=object), dtype=str_dtype)

                    # Metadata (unchanged, + one helpful provenance attribute)
                    h5f.attrs.create("task_regressor_names", np.array(trial_names, dtype=str_dtype), dtype=str_dtype)
                    h5f.attrs["regressor_level"] = "trial"
                    h5f.attrs["task_col_start"] = task_col_start
                    h5f.attrs["task_col_end"] = task_col_end
                    h5f.attrs["dof"] = dof
                    h5f.attrs["rho_ar1"] = float(estimate_rho_from_resid(resid))
                    h5f.attrs["tmask_dropped"] = tmask
                    h5f.attrs["high_pass_sec"] = args.high_pass_sec
                    h5f.attrs["order_in_X"] = "[confounds, drifts+intercept, task(trials)]"
                    # --- NEW: estimator provenance (so downstream knows WLS/AR1 vs OLS) ---
                    h5f.attrs["estimator"] = "WLS_AR1"

                print(f"[🎯] Saved betas + full design to {h5_file}")
                print(f"[🧾] Trial design CSV: {design_csv}")
                print(f"[📐] Full design matrix CSV: {design_matrix_csv}")

    print("[✅] Done.")


if __name__ == "__main__":
    main()
