#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


def run_number(run_label: str):
    m = re.match(r"^run-(\d+)$", str(run_label).strip())
    return int(m.group(1)) if m else None


def attr_value(attrs, key, default=np.nan):
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def format_num(v, prec=3):
    if isinstance(v, str):
        return v
    try:
        fv = float(v)
    except Exception:
        return "nan"
    if not np.isfinite(fv):
        return "nan"
    return f"{fv:.{prec}f}"


def main():
    ap = argparse.ArgumentParser(description="Compare ablation H5 outputs side-by-side.")
    ap.add_argument("--dir", required=True, help="Directory containing *_betas.h5 files")
    ap.add_argument("--run", required=True, help="Run identifier, e.g. run-1")
    ap.add_argument("--task", required=True, help="Task name, e.g. ctxdm")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        raise FileNotFoundError(f"Directory does not exist: {d}")

    target_run_num = run_number(args.run)
    rows = []
    for h5_path in sorted(d.rglob("*_betas.h5")):
        name = h5_path.name
        if f"task-{args.task}_" not in name:
            continue

        m = re.search(r"_run-(\d+)_betas\.h5$", name)
        if not m:
            continue
        run_num = int(m.group(1))
        if target_run_num is not None and run_num != target_run_num:
            continue

        with h5py.File(h5_path, "r") as f:
            mode = str(attr_value(f.attrs, "preproc_mode", "unknown"))
            estimator = str(attr_value(f.attrs, "estimator", "unknown"))
            rho = attr_value(f.attrs, "rho_ar1", np.nan)
            vif_max = attr_value(f.attrs, "vif_max", np.nan)
            low_mu = attr_value(f.attrs, "n_low_mean_psc_units", np.nan)

            betas = f["betas"][:] if "betas" in f else np.array([])
            beta_mean = float(np.nanmean(betas)) if betas.size else np.nan
            beta_std = float(np.nanstd(betas)) if betas.size else np.nan

            if "r2_task" in f:
                r2_task = float(np.nanmean(f["r2_task"][:]))
            elif "r2_full" in f:
                r2_task = float(np.nanmean(f["r2_full"][:]))
            else:
                r2_task = np.nan

            if "psc_baseline_mean" in f:
                psc_mu = float(np.nanmean(f["psc_baseline_mean"][:]))
            else:
                psc_mu = np.nan

        flags = []
        if np.isfinite(beta_std) and beta_std < 0.01:
            flags.append("⚠ flat betas")
        if np.isfinite(r2_task) and r2_task < 0:
            flags.append("⚠ negative R²")
        if np.isfinite(psc_mu) and psc_mu < 10:
            flags.append("⚠ near-zero PSC mean")
        try:
            vif_val = float(vif_max)
        except Exception:
            vif_val = np.nan
        if np.isfinite(vif_val) and vif_val > 10:
            flags.append("⚠ high collinearity")

        rows.append({
            "mode": mode,
            "estimator": estimator,
            "rho": float(rho) if np.isfinite(float(rho)) else np.nan,
            "beta_mean": beta_mean,
            "beta_std": beta_std,
            "r2_task": r2_task,
            "psc_mu": psc_mu,
            "vif_max": vif_val if np.isfinite(vif_val) else np.nan,
            "n_low_mean_psc_units": low_mu,
            "flags": ", ".join(flags),
            "file": str(h5_path),
        })

    if not rows:
        print("No matching files found.")
        return

    print("mode          | estimator | rho   | beta_mean | beta_std | r2_task | psc_mu  | vif_max | flags")
    for r in rows:
        mode = f"{r['mode']:<13}"
        est = f"{r['estimator']:<9}"
        line = (
            f"{mode}| {est}| {format_num(r['rho'],3):>5} | "
            f"{format_num(r['beta_mean'],3):>9} | {format_num(r['beta_std'],3):>8} | "
            f"{format_num(r['r2_task'],3):>7} | {format_num(r['psc_mu'],1):>7} | "
            f"{format_num(r['vif_max'],2):>7} | {r['flags']}"
        )
        print(line)


if __name__ == "__main__":
    main()
