#!/usr/bin/env python3
"""Corrected split-half noise-ceiling estimator (region + network) on the clean delay-beta bundle.

Estimand: for every subject, task variant (12) and split iteration, the unique retained delay betas of the
variant are divided into two disjoint, size-balanced halves; halves are averaged within variant; at every
spatial unit the two 12-element response vectors are correlated (Pearson, across task variants). Raw r are
kept for every subject x resolution x unit x iteration. Aggregation across the 50 iterations is done in
Fisher-z space; the bounded Spearman-Brown estimate is NC = 2 rbar / (1 + rbar) if rbar > 0 else 0.

Input bundle (from build_clean_delay_betas_nc.py or legacy_pickle_to_bundle.py):
  <input_dir>/<subj>_delay_provenance.csv   columns used: record_id, <label_col>, <retain_col>, label_resolved, stim_order
  <input_dir>/<subj>_delay_region.npy       n_records x 360
  <input_dir>/<subj>_delay_network.npy      n_records x 12

Controls / sensitivity (flags): --permute_labels one_half|both, --equalize_counts, --inject_duplicates,
--exclude_records_csv, --require_resolved, --loco (always computed; cheap).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

TASK_VARIANTS = [
    "dms_loc",
    "nback_loc", "nback_ctg", "nback_obj",
    "ctxdm_col", "ctxdm_lco",
    "interdms_loc_ABAB", "interdms_loc_ABBA",
    "interdms_obj_ABAB", "interdms_obj_ABBA",
    "interdms_ctg_ABAB", "interdms_ctg_ABBA",
]
NETWORK_NAMES = {
    1: "primary visual", 2: "secondary visual", 3: "somatomotor", 4: "cingulo-opercular",
    5: "dorsal attention", 6: "language", 7: "frontoparietal", 8: "auditory",
    9: "default mode", 10: "posterior multimodal", 11: "ventral multimodal", 12: "orbito-affective",
}
EPS = 1e-6


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def glasser_names(atlas_path: str) -> list[str]:
    try:
        img = nib.load(atlas_path)
        table = img.header.get_axis(0).label[0]
        return [table[i][0] for i in range(1, 361)]
    except Exception:  # noqa: BLE001
        return [f"region_{i}" for i in range(1, 361)]


def colwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between columns of A and B (both n_conditions x n_units), computed across rows."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A ** 2).sum(axis=0) * (B ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~np.isfinite(r)] = np.nan
    return r


def fisher_mean(r: np.ndarray, axis=0) -> np.ndarray:
    z = np.arctanh(np.clip(r, -1 + EPS, 1 - EPS))
    return np.tanh(np.nanmean(z, axis=axis))


def bounded_sb(rbar: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rbar, dtype=float)
    pos = rbar > 0
    out[pos] = 2 * rbar[pos] / (1 + rbar[pos])
    out[~np.isfinite(rbar)] = np.nan
    return out


def legacy_periter_sb(r: np.ndarray) -> np.ndarray:
    """Diagnostic only: legacy per-iteration Spearman-Brown with clipping to +-0.9999, then mean."""
    rc = np.clip(r, -0.9999, 0.9999)
    return np.nanmean(2 * rc / (1 + rc), axis=0)


def make_splits(n: int, b: int, seed: int):
    """Disjoint balanced split of range(n) for iteration b. Odd n: the extra record alternates (A on even b)."""
    rng = np.random.default_rng([seed, b])
    perm = rng.permutation(n)
    mid = n // 2
    if n % 2 == 1 and b % 2 == 0:
        mid += 1
    return perm[:mid], perm[mid:]


def summarize_units(nc: np.ndarray, raw_r: np.ndarray, rbar: np.ndarray) -> dict:
    v = nc[np.isfinite(nc)]
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n_units": int(nc.size), "n_finite": int(v.size),
        "mean_nc": float(v.mean()), "median_nc": float(np.median(v)), "sd_nc": float(v.std(ddof=1)),
        "iqr_nc": float(q3 - q1), "q1_nc": float(q1), "q3_nc": float(q3), "min_nc": float(v.min()), "max_nc": float(v.max()),
        "prop_zero": float((v == 0).mean()), "n_zeroed_rbar_le_0": int((rbar <= 0).sum()),
        "prop_gt_0.2": float((v > 0.2).mean()), "prop_gt_0.5": float((v > 0.5).mean()), "prop_gt_0.7": float((v > 0.7).mean()),
        "frac_raw_r_negative": float(np.nanmean(raw_r < 0)),
        "mean_rbar_fisher": float(np.nanmean(rbar)), "median_rbar_fisher": float(np.nanmedian(rbar)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_col", default="task_variant_lookup")
    ap.add_argument("--retain_col", default="retain_trial_level")
    ap.add_argument("--require_resolved", action="store_true", help="drop records whose label_resolved is False")
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--permute_labels", choices=["none", "one_half", "both"], default="none")
    ap.add_argument("--equalize_counts", action="store_true")
    ap.add_argument("--inject_duplicates", action="store_true",
                    help="DIAGNOSTIC: re-introduce the legacy duplicate multiplicity (delay after item>1 stored twice)")
    ap.add_argument("--exclude_records_csv", default="", help="CSV with a record_id column to exclude")
    ap.add_argument("--min_per_variant", type=int, default=2)
    ap.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--variant_name", default="")
    args = ap.parse_args()

    t0 = time.time()
    subj = args.subj
    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prov = pd.read_csv(inp / f"{subj}_delay_provenance.csv")
    mats = {"region": np.load(inp / f"{subj}_delay_region.npy"), "network": np.load(inp / f"{subj}_delay_network.npy")}
    for k, m in mats.items():
        if m.shape[0] != len(prov):
            raise SystemExit(f"{k} matrix rows {m.shape[0]} != provenance rows {len(prov)}")
        if not np.all(np.isfinite(m)):
            raise SystemExit(f"non-finite values in {k} matrix")
    if mats["region"].shape[1] != 360 or mats["network"].shape[1] != 12:
        raise SystemExit("unexpected spatial axes")

    # ---------------- record selection ----------------
    keep = prov[args.retain_col].astype(bool).to_numpy()
    if "phase" in prov.columns and not (prov.phase == "delay").all():
        raise SystemExit("non-delay records present in bundle")
    if args.require_resolved and "label_resolved" in prov.columns:
        keep &= prov.label_resolved.astype(bool).to_numpy()
    n_excluded_extra = 0
    if args.exclude_records_csv:
        ex = set(pd.read_csv(args.exclude_records_csv).record_id.astype(str))
        m = prov.record_id.astype(str).isin(ex).to_numpy()
        n_excluded_extra = int((m & keep).sum())
        keep &= ~m
    labels = prov[args.label_col].astype(str).to_numpy()
    keep &= np.isin(labels, TASK_VARIANTS)
    sel = prov[keep].copy()
    sel["label"] = labels[keep]
    sel = sel.sort_values("record_id").reset_index(drop=True)  # deterministic order
    if sel.record_id.duplicated().any():
        raise SystemExit("duplicate record_id in selected records")
    idx_of = {rid: i for i, rid in enumerate(prov.record_id)}
    sel["row"] = sel.record_id.map(idx_of).astype(int)

    # equalized trial counts (deterministic subsample, one draw per subject)
    subsample_note = None
    if args.equalize_counts:
        n_min = int(sel.groupby("label").size().min())
        rng = np.random.default_rng([args.seed, 999_983])
        parts = []
        for tv in TASK_VARIANTS:
            g = sel[sel.label == tv]
            take = np.sort(rng.choice(len(g), size=n_min, replace=False))
            parts.append(g.iloc[take])
        sel = pd.concat(parts).reset_index(drop=True)
        subsample_note = f"equalized to n={n_min} unique records per task variant (seed [{args.seed}, 999983])"

    # duplicate injection (diagnostic): expand records by legacy multiplicity
    multiplicity = np.ones(len(sel), dtype=int)
    if args.inject_duplicates:
        # Re-introduce the legacy pair+self storage: provenance.legacy_multiplicity is 2 when the same
        # physical delay beta was written under both the pair key and the single-frame key.
        if "legacy_multiplicity" in sel.columns:
            multiplicity = sel.legacy_multiplicity.fillna(1).to_numpy().astype(int)
            multiplicity = np.clip(multiplicity, 1, None)
        else:
            multiplicity = np.where(sel.stim_order.to_numpy() > 1, 2, 1)

    groups = {}
    for tv in TASK_VARIANTS:
        g = sel[sel.label == tv]
        rows = np.repeat(g.row.to_numpy(), multiplicity[g.index.to_numpy()])
        rids = np.repeat(g.record_id.to_numpy(), multiplicity[g.index.to_numpy()])
        if len(rows) < args.min_per_variant:
            raise SystemExit(f"task variant {tv} has only {len(rows)} retained records (< {args.min_per_variant})")
        groups[tv] = (rows, rids)

    # ---------------- splits + correlations ----------------
    n_iter = args.n_iter
    half_assign = {tv: np.zeros((len(groups[tv][0]), n_iter), dtype=np.int8) for tv in TASK_VARIANTS}
    raw_r = {res: np.full((n_iter, mats[res].shape[1]), np.nan) for res in mats}
    loco_r = {res: np.full((12, n_iter, mats[res].shape[1]), np.nan) for res in mats}
    perm_check_identical = True
    half_sizes = {tv: [] for tv in TASK_VARIANTS}
    shared_trial_violations = 0
    for b in range(n_iter):
        H1 = {res: np.zeros((12, mats[res].shape[1])) for res in mats}
        H2 = {res: np.zeros((12, mats[res].shape[1])) for res in mats}
        for ci, tv in enumerate(TASK_VARIANTS):
            rows, rids = groups[tv]
            a, bidx = make_splits(len(rows), b, args.seed + 1000 * ci)
            half_assign[tv][a, b] = 1
            half_assign[tv][bidx, b] = 2
            if len(set(rids[a]) & set(rids[bidx])):
                shared_trial_violations += 1  # only possible with --inject_duplicates
            half_sizes[tv].append((len(a), len(bidx)))
            for res in mats:
                H1[res][ci] = mats[res][rows[a]].mean(axis=0)
                H2[res][ci] = mats[res][rows[bidx]].mean(axis=0)
        if args.permute_labels != "none":
            prng = np.random.default_rng([args.seed, 777, b])
            p = prng.permutation(12)
            for res in mats:
                if args.permute_labels == "one_half":
                    H2[res] = H2[res][p]
                else:
                    r_before = colwise_pearson(H1[res], H2[res])
                    H1[res], H2[res] = H1[res][p], H2[res][p]
                    if not np.allclose(r_before, colwise_pearson(H1[res], H2[res]), atol=1e-10, equal_nan=True):
                        perm_check_identical = False
        for res in mats:
            raw_r[res][b] = colwise_pearson(H1[res], H2[res])
            for k in range(12):
                m = np.ones(12, bool)
                m[k] = False
                loco_r[res][k, b] = colwise_pearson(H1[res][m], H2[res][m])

    if shared_trial_violations and not args.inject_duplicates:
        raise SystemExit("split halves share physical trials")

    # ---------------- aggregation ----------------
    region_names = glasser_names(args.atlas_path)
    unit_tables = {}
    summaries = {}
    for res in mats:
        R = raw_r[res]
        rbar = fisher_mean(R)
        r_arith = np.nanmean(R, axis=0)
        r_med = np.nanmedian(R, axis=0)
        nc = bounded_sb(rbar)
        nc_arith = bounded_sb(r_arith)
        nc_med = bounded_sb(r_med)
        legacy = legacy_periter_sb(R)
        names = region_names if res == "region" else [NETWORK_NAMES[i] for i in range(1, 13)]
        df = pd.DataFrame({
            "subject": subj, "resolution": res, "unit_index": np.arange(1, R.shape[1] + 1), "unit_name": names,
            "r_fisher_mean": rbar, "r_arith_mean": r_arith, "r_median": r_med,
            "frac_iter_r_negative": np.nanmean(R < 0, axis=0), "n_iter_nan": np.isnan(R).sum(axis=0),
            "nc_bounded_sb": nc, "zeroed_rbar_le_0": rbar <= 0,
            "nc_from_arith_mean": nc_arith, "nc_from_median": nc_med,
            "nc_legacy_periter_sb_DIAGNOSTIC": legacy, "legacy_periter_abs_gt_1": np.abs(legacy) > 1,
        })
        for k, tv in enumerate(TASK_VARIANTS):
            df[f"loco_nc_without_{tv}"] = bounded_sb(fisher_mean(loco_r[res][k]))
        loco_nc = np.stack([df[f"loco_nc_without_{tv}"].to_numpy() for tv in TASK_VARIANTS], axis=1)
        df["loco_min_nc"] = loco_nc.min(axis=1)
        df["loco_max_nc"] = loco_nc.max(axis=1)
        df["loco_max_drop"] = nc - loco_nc.min(axis=1)  # largest decrease in NC when one variant is left out
        df["loco_dominant_condition"] = [TASK_VARIANTS[i] for i in loco_nc.argmin(axis=1)]
        df["loco_dominated_flag"] = (df.loco_max_drop > 0.2) & (df.nc_bounded_sb > 0.2)
        unit_tables[res] = df
        df.to_csv(out / f"unit_estimates_{res}.csv", index=False)
        np.save(out / f"raw_r_{res}.npy", R.astype(np.float32))
        np.save(out / f"loco_raw_r_{res}.npy", loco_r[res].astype(np.float32))
        long = pd.DataFrame(R).reset_index().melt(id_vars="index", var_name="unit_index", value_name="r")
        long = long.rename(columns={"index": "iteration"})
        long["unit_index"] += 1
        long.insert(0, "resolution", res)
        long.insert(0, "subject", subj)
        long.to_csv(out / f"raw_split_r_long_{res}.csv", index=False)
        summaries[res] = summarize_units(nc, R, rbar)
        summaries[res].update({
            "mean_nc_from_arith_mean": float(np.nanmean(nc_arith)), "mean_nc_from_median": float(np.nanmean(nc_med)),
            "mean_legacy_periter_sb_DIAGNOSTIC": float(np.nanmean(legacy)),
            "n_legacy_periter_abs_gt_1": int((np.abs(legacy) > 1).sum()),
            "n_units_loco_dominated": int(df.loco_dominated_flag.sum()),
        })

    # ---------------- splits + counts ----------------
    np.savez_compressed(out / "splits.npz",
                        **{f"half_{tv}": half_assign[tv] for tv in TASK_VARIANTS},
                        **{f"record_ids_{tv}": np.array(groups[tv][1], dtype="U64") for tv in TASK_VARIANTS},
                        seed=args.seed, n_iter=n_iter)
    counts = []
    for ci, tv in enumerate(TASK_VARIANTS):
        hs = np.array(half_sizes[tv])
        counts.append({"subject": subj, "task_variant": tv, "n_records_used": len(groups[tv][0]),
                       "n_unique_physical": len(set(groups[tv][1])),
                       "half_A_size_min": int(hs[:, 0].min()), "half_A_size_max": int(hs[:, 0].max()),
                       "half_B_size_min": int(hs[:, 1].min()), "half_B_size_max": int(hs[:, 1].max()),
                       "odd_count": len(groups[tv][0]) % 2 == 1})
    counts = pd.DataFrame(counts)
    counts.to_csv(out / "trial_counts_per_variant.csv", index=False)

    config = {
        "subject": subj, "variant_name": args.variant_name, "args": vars(args),
        "n_records_in_bundle": int(len(prov)), "n_records_retained": int(len(sel)),
        "n_records_excluded_by_exclude_csv": n_excluded_extra, "subsample_note": subsample_note,
        "duplicate_injection_multiplicity_gt1": int((multiplicity > 1).sum()) if args.inject_duplicates else 0,
        "split_policy": "disjoint permutation split per task variant and iteration; rng = default_rng([seed + 1000*variant_index, iteration]); "
                        "odd counts: extra record to half A on even iterations, half B on odd iterations",
        "aggregation": "Fisher-z mean of raw r across iterations -> rbar; NC = 2 rbar/(1+rbar) if rbar>0 else 0",
        "eps_clip": EPS, "shared_trial_violations": shared_trial_violations,
        "both_halves_permutation_identical_to_unpermuted": perm_check_identical if args.permute_labels == "both" else None,
        "git_commit_worktree": git_commit(Path(__file__).resolve().parents[2]),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "elapsed_sec": round(time.time() - t0, 1), "command": " ".join(sys.argv),
        "summaries": summaries,
    }
    with open(out / "config_and_summary.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(json.dumps({"subject": subj, "variant": args.variant_name, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
