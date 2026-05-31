#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Visualize GLM fit quality for selected grayordinates.")
    p.add_argument("--h5", required=True, help="Path to *_betas.h5")
    p.add_argument("--bold", required=True, help="Path to original dtseries.nii")
    p.add_argument("--n", type=int, default=6, help="Number of grayordinates to plot")
    p.add_argument("--mode", choices=["best", "worst", "random", "manual"], default="best")
    p.add_argument("--grayord_idx", nargs="+", type=int, default=None, help="Indices for manual mode")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--tr", type=float, default=1.49, help="TR in seconds")
    return p.parse_args()


def contiguous_blocks(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return list(zip(starts, ends))


def main():
    args = parse_args()
    np.random.seed(0)

    h5_path = Path(args.h5)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        betas = f["betas"][:]  # (P x n_trials)
        X = f["design_matrix"][:]  # (T x K)
        names_raw = f["design_col_names"][:]
        col_names = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in names_raw]
        valid_mask = f["valid_mask"][:].astype(bool) if "valid_mask" in f else np.ones(betas.shape[0], dtype=bool)
        psc_baseline = f["psc_baseline_mean"][:]  # (1 x P)
        tmask = int(f.attrs.get("tmask_dropped", 1))
        task_reg_names_raw = f.attrs.get("task_regressor_names", [])
        task_regressor_names = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in list(task_reg_names_raw)
        ]
        betas_full = f["betas_full"][:] if "betas_full" in f else None
        yhat_stored = f["yhat"][:] if "yhat" in f else None

        if "r2_task" in f:
            r2 = f["r2_task"][:].astype(np.float32)
            if np.all(~np.isfinite(r2)) and "r2_full" in f:
                r2 = f["r2_full"][:].astype(np.float32)
        elif "r2_full" in f:
            r2 = f["r2_full"][:].astype(np.float32)
        else:
            r2 = np.full((betas.shape[0],), np.nan, dtype=np.float32)

    Y_raw = nib.load(args.bold).get_fdata(dtype=np.float32)
    Y_raw = Y_raw[tmask:, :]
    mu = psc_baseline.astype(np.float64)
    safe_mu = np.where(np.abs(mu) > 1e-8, mu, np.nan)
    Y_psc = 100.0 * (Y_raw.astype(np.float64) - mu) / safe_mu
    Y_psc = np.nan_to_num(Y_psc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Reconstruct plotting yhat with priority:
    # 1) full-model design @ betas_full, 2) stored yhat, 3) condition-level OLS
    if betas_full is not None:
        yhat_plot = (X.astype(np.float64) @ betas_full[:, :X.shape[1]].T.astype(np.float64)).astype(np.float32)
        print("yhat reconstructed from betas_full @ design_matrix")
    elif yhat_stored is not None:
        yhat_plot = yhat_stored.astype(np.float32)
        print("WARNING: using stored yhat from last LSS iteration")
    else:
        enc_cols_idx = [i for i, n in enumerate(col_names) if "encoding" in n]
        del_cols_idx = [i for i, n in enumerate(col_names) if "delay" in n]
        noise_idx = [i for i, n in enumerate(col_names) if "encoding" not in n and "delay" not in n]
        r_enc = X[:, enc_cols_idx].mean(axis=1, keepdims=True)
        r_del = X[:, del_cols_idx].mean(axis=1, keepdims=True)
        X_cond = np.hstack([X[:, noise_idx], r_enc, r_del])
        B, _, _, _ = np.linalg.lstsq(X_cond.astype(np.float64), Y_psc.astype(np.float64), rcond=None)
        yhat_plot = (X_cond.astype(np.float64) @ B).astype(np.float32)
        print("yhat computed from condition-level OLS fit for plotting")

    valid_indices = np.where(valid_mask)[0]
    n = min(args.n, valid_indices.size)
    if n == 0:
        raise ValueError("No valid grayordinates available.")

    r2_valid = r2[valid_indices]
    finite_valid = np.isfinite(r2_valid)
    if np.any(finite_valid):
        order = np.argsort(r2_valid[finite_valid])
        finite_indices = valid_indices[finite_valid]
    else:
        order = np.arange(valid_indices.size)
        finite_indices = valid_indices

    if args.mode == "best":
        sel = finite_indices[order][-n:][::-1]
    elif args.mode == "worst":
        sel = finite_indices[order][:n]
    elif args.mode == "random":
        sel = np.random.choice(valid_indices, size=n, replace=False)
    else:
        if not args.grayord_idx:
            raise ValueError("--grayord_idx is required for manual mode.")
        sel = np.array(args.grayord_idx, dtype=int)

    enc_cols = [i for i, n in enumerate(col_names) if "encoding" in n]
    del_cols = [i for i, n in enumerate(col_names) if "delay" in n]
    T = X.shape[0]
    time_ax = np.arange(T) * args.tr
    fig, axes = plt.subplots(n, 2, figsize=(16, 3 * n), gridspec_kw={"width_ratios": [3, 1]})
    if n == 1:
        axes = np.array([axes])

    # Trial name list used for beta histogram splitting
    trial_names = task_regressor_names if task_regressor_names else [n for n in col_names if ("encoding" in n or "delay" in n)]
    enc_idx = [i for i, nm in enumerate(trial_names) if "encoding" in nm]
    del_idx = [i for i, nm in enumerate(trial_names) if "delay" in nm]

    for row, idx in enumerate(sel):
        ax_ts, ax_hist = axes[row, 0], axes[row, 1]
        # Plot shading first (behind traces), per-trial contiguous blocks
        for col_idx in enc_cols:
            reg = X[:, col_idx]
            threshold = 0.3 * reg.max()
            if threshold <= 0:
                continue
            on_mask = reg > threshold
            changes = np.diff(on_mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            if on_mask[0]:
                starts = np.concatenate([[0], starts])
            if on_mask[-1]:
                ends = np.concatenate([ends, [T]])
            for s, e in zip(starts, ends):
                ax_ts.axvspan(time_ax[s], time_ax[min(e, T - 1)],
                              color="red", alpha=0.25, linewidth=0, zorder=1)

        for col_idx in del_cols:
            reg = X[:, col_idx]
            threshold = 0.3 * reg.max()
            if threshold <= 0:
                continue
            on_mask = reg > threshold
            changes = np.diff(on_mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            if on_mask[0]:
                starts = np.concatenate([[0], starts])
            if on_mask[-1]:
                ends = np.concatenate([ends, [T]])
            for s, e in zip(starts, ends):
                ax_ts.axvspan(time_ax[s], time_ax[min(e, T - 1)],
                              color="blue", alpha=0.15, linewidth=0, zorder=1)

        # Actual and predicted traces
        ax_ts.plot(time_ax, Y_psc[:, idx],
                   color="gray", linewidth=0.8, alpha=0.6,
                   label="actual PSC", zorder=2)
        ax_ts.plot(time_ax, yhat_plot[:, idx],
                   color="#1f77b4", linewidth=1.5,
                   label="predicted", zorder=3)

        sig = Y_psc[:, idx]
        ymin = np.nanpercentile(sig, 1) - 0.5
        ymax = np.nanpercentile(sig, 99) + 0.5
        ax_ts.set_ylim(ymin, ymax)

        ax_ts.set_ylabel("% BOLD change")
        if row == n - 1:
            ax_ts.set_xlabel("time (s)")
        r2_val = float(r2[idx]) if np.isfinite(r2[idx]) else float("nan")
        ax_ts.set_title(f"grayordinate {idx} | R²_task={r2_val:.3f}")
        if row == 0:
            ax_ts.legend(loc="upper right", fontsize=8)

        b_enc = betas[idx, enc_idx] if enc_idx else np.array([], dtype=np.float32)
        b_del = betas[idx, del_idx] if del_idx else np.array([], dtype=np.float32)
        if b_enc.size:
            ax_hist.hist(b_enc, bins=20, color="red", alpha=0.6, label="encoding")
        if b_del.size:
            ax_hist.hist(b_del, bins=20, color="blue", alpha=0.6, label="delay")
        ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax_hist.set_xlabel("β (% BOLD)")
        enc_mu = float(np.nanmean(b_enc)) if b_enc.size else float("nan")
        del_mu = float(np.nanmean(b_del)) if b_del.size else float("nan")
        ax_hist.set_title(f"enc μ={enc_mu:.2f} | del μ={del_mu:.2f}")
        ax_hist.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
