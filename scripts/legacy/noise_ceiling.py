#!/usr/bin/env python3
"""
Noise ceiling at voxel / region / network levels.
Two methods:
  - traditional: split-half correlation across conditions (+ optional SB)
  - variance_based: participant-variance decomposition (manuscript style)

Examples
--------
# Variance-based, correlation units (sqrt), per-network
python noise_ceiling.py \
  --method variance_based \
  --subject sub-01 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-01_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --unit region --sqrt_to_r 1 --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/variance_based 

# Traditional split-half (your original)
python noise_ceiling.py \
  --method traditional \
  --subject sub-01 \
  --input_pickle /path/sub-01_task_condition_betas.pkl \
  --unit voxel --n_bootstrap 50 --min_reps 4 --apply_spearman_brown 1 --standardize_halves 1 \
  --output_dir /tmp/nc_trad
"""

from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nibabel as nib


# ---------- helpers ----------
def colwise_correlation(X, Y):
    assert X.shape == Y.shape, f"Shape mismatch: {X.shape} vs {Y.shape}"
    n = X.shape[1]
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        x, y = X[:, i], Y[:, i]
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        r, _ = pearsonr(x, y)
        out[i] = r
    return out

def spearman_brown(r, k=2.0):
    return (k * r) / (1.0 + (k - 1.0) * r)

def load_glasser_labels(atlas_path: str):
    lab_img = nib.load(atlas_path)
    lab = np.asarray(lab_img.get_fdata()).squeeze().reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids

def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]
    def avg_by_region(A):
        out = np.full((A.shape[0], len(region_ids)), np.nan, dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx.size: out[:, j] = np.nanmean(A[:, idx], axis=1)
        return out
    return avg_by_region

def load_region_to_network_map(assign_path: str, max_region_id: int):
    txt = Path(assign_path).read_text().strip().splitlines()
    arr = np.zeros(max_region_id + 1, dtype=int)
    for i, line in enumerate(txt, start=1):
        if i <= max_region_id:
            arr[i] = int(line.strip())
    return arr

def make_network_averager(region_ids: np.ndarray, region_to_net: np.ndarray, n_networks: int = 12):
    nets = [[] for _ in range(n_networks)]
    for j, rid in enumerate(region_ids):
        nid = region_to_net[rid]
        if 1 <= nid <= n_networks:
            nets[nid - 1].append(j)
    def avg_by_network(R):
        out = np.full((R.shape[0], n_networks), np.nan, dtype=np.float32)
        for k, cols in enumerate(nets):
            if cols: out[:, k] = np.nanmean(R[:, cols], axis=1)
        return out
    return avg_by_network

def zscore_columns(X, eps=1e-8):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    return (X - mu) / (sd + eps)

# ---------- variance-based NC (manuscript style) ----------
def nc_variance_based(condition_betas: dict[str, list[np.ndarray]],
                      unit_transform=None,
                      n_participants: int | None = None,
                      rectify_signal: bool = True,
                      sqrt_to_r: bool = False):
    """
    Treat each condition's repetitions as 'participants'.
    Returns per-unit NC in variance fraction (0..1), or sqrt space if sqrt_to_r.
    """
    # Build a list of conditions with >=2 reps (need variance across reps)
    valid = [(k, v) for k, v in condition_betas.items() if len(v) >= 2]
    if not valid:
        raise SystemExit("Variance-based NC: need at least 2 repetitions per condition.")

    # Stack into ragged form: we will compute per-condition stats on the fly.
    # Also infer feature dim from first vec.
    first_vec = np.asarray(valid[0][1][0]).reshape(-1)
    n_features = first_vec.size

    # A small helper to apply optional unit transforms (region/network)
    def apply_unit(A_reps):  # A_reps: (n_reps, n_features)
        if unit_transform is None:
            return A_reps  # (n_reps, n_units=n_features)
        return unit_transform(A_reps)  # (n_reps, n_units)

    # First pass: determine n_units and collect per-condition means/variances
    # We will accumulate:
    #  - noise variances per condition and unit: var_across_reps(c, u)
    #  - participant-mean per condition and unit: mean_over_reps(c, u)
    noise_var_list = []   # list of arrays (n_units,)
    cond_mean_list = []   # list of arrays (n_units,)
    nrep_list = []

    for cond, reps_list in valid:
        reps = np.stack([np.asarray(x).reshape(-1) for x in reps_list], axis=0)  # (n_reps, n_features)
        reps = apply_unit(reps)  # (n_reps, n_units)
        # unbiased var across reps (ddof=1) per unit
        if reps.shape[0] < 2:
            continue
        noise_var = np.nanvar(reps, axis=0, ddof=1)
        cond_mean = np.nanmean(reps, axis=0)
        noise_var_list.append(noise_var)
        cond_mean_list.append(cond_mean)
        nrep_list.append(reps.shape[0])

    if not noise_var_list:
        raise SystemExit("Variance-based NC: no conditions with >=2 reps after transform.")

    noise_var_mean = np.nanmean(np.stack(noise_var_list, axis=0), axis=0)  # (n_units,)
    cond_means = np.stack(cond_mean_list, axis=0)  # (n_conditions, n_units)
    n_units = cond_means.shape[1]
    Np = 1

    
    var_across_conditions = np.nanvar(cond_means, axis=0, ddof=1)  # (n_units,)

    # Signal variance estimate per unit:
    # sigma_signal^2 = var(group-avg) - noise_var/Np   (half-wave rectified)
    sigma_signal2 = var_across_conditions - (noise_var_mean / max(Np, 1))
    if rectify_signal:
        sigma_signal2 = np.maximum(sigma_signal2, 0.0)

    # Noise ceiling in variance fraction
    denom = sigma_signal2 + (noise_var_mean / max(Np, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        nc = np.where(denom > 0, sigma_signal2 / denom, np.nan)  # (n_units,)

    if sqrt_to_r:
        # convert to correlation units to compare to Pearson r
        nc = np.sqrt(np.clip(nc, 0.0, 1.0))

    return nc  # per-unit NC

# ---------- traditional NC (your original) ----------
def nc_traditional(condition_betas: dict[str, list[np.ndarray]],
                   unit_transform=None,
                   n_bootstrap: int = 50,
                   min_reps: int = 4,
                   rng: np.random.Generator = np.random.default_rng(0),
                   standardize_halves: bool = False,
                   apply_spearman_brown_flag: bool = True):
    boot_nc = []
    for _ in range(n_bootstrap):
        half1_rows, half2_rows = [], []
        for cond, reps_list in condition_betas.items():
            if len(reps_list) < min_reps:
                continue
            reps = np.stack([np.asarray(v).reshape(-1) for v in reps_list], axis=0)  # (n_reps, n_features)
            if unit_transform is not None:
                reps = unit_transform(reps)  # (n_reps, n_units)
            perm = rng.permutation(reps.shape[0])
            mid = reps.shape[0] // 2
            A = reps[perm[:mid]].mean(axis=0)
            B = reps[perm[mid:]].mean(axis=0)
            if A.size == 0 or B.size == 0:
                continue
            half1_rows.append(A)
            half2_rows.append(B)
        if not half1_rows:
            continue
        H1 = np.stack(half1_rows, axis=0)
        H2 = np.stack(half2_rows, axis=0)
        if standardize_halves:
            H1 = zscore_columns(H1); H2 = zscore_columns(H2)
        r_cols = colwise_correlation(H1, H2)
        if apply_spearman_brown_flag:
            r_cols = np.clip(r_cols, -0.9999, 0.9999)
            r_cols = spearman_brown(r_cols, k=2.0)
        boot_nc.append(r_cols)
    if not boot_nc:
        raise SystemExit("Traditional NC: no valid splits (not enough data per condition).")
    return np.nanmean(np.stack(boot_nc, axis=0), axis=0)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["variance_based", "traditional"], default="variance_based")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True, help="Path to {condition: [beta_vec,...]}")
    ap.add_argument("--output_dir", required=True)

    # Shared spatial granularity
    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="voxel")
    ap.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--network_assignments_path", help="Region index (line) -> network id (1..12)")
    ap.add_argument("--network_mapping_json", default="")

    # Traditional method knobs
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument("--min_reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply_spearman_brown", type=int, default=1)
    ap.add_argument("--standardize_halves", type=int, default=0)

    # Variance-based method knobs
    ap.add_argument("--n_participants", type=int, default=None,
                    help="Effective participant count Np; if omitted, infer from median repetitions.")
    ap.add_argument("--sqrt_to_r", type=int, default=0,
                    help="If 1, return sqrt(NC) to express in correlation units (as in manuscript).")
    ap.add_argument("--rectify_signal", type=int, default=1,
                    help="Half-wave rectify sigma_signal^2 at 0 (recommended).")

    # Optional: sentence-level split-half SE for variance-based (like SI 5B)
    ap.add_argument("--nc_se_bootstrap", type=int, default=0,
                    help="If >0, compute SE by random sentence splits; saves per-iteration NC too.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Load grouped betas
    with open(Path(args.input_pickle), "rb") as f:
        condition_betas = pickle.load(f)
    if not isinstance(condition_betas, dict):
        raise SystemExit("input_pickle must be a dict: {condition_key: [beta_vec, ...]}")

    # Infer feature length
    any_key = next(iter(condition_betas))
    first_vec = np.asarray(condition_betas[any_key][0]).reshape(-1)
    n_features = first_vec.size

    # Prepare spatial transforms
    avg_by_region = None
    avg_by_network = None
    region_ids = None
    unit_name = args.unit
    n_units = n_features

    if args.unit in ("region", "network"):
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        if labels_vec.size != n_features:
            raise SystemExit(f"Atlas feature count ({labels_vec.size}) != beta vector length ({n_features}). "
                             "Ensure same space and indexing.")
        avg_by_region = make_region_averager(labels_vec, region_ids)
        n_units = len(region_ids)
        unit_name = "region"

    if args.unit == "network":
        if not args.network_assignments_path:
            raise SystemExit("--network_assignments_path required for unit=network")
        max_region_id = int(region_ids.max())
        region_to_net = load_region_to_network_map(args.network_assignments_path, max_region_id)
        avg_by_network = make_network_averager(region_ids, region_to_net, n_networks=12)
        unit_name = "network"
        n_units = 12
        # optional names
        if args.network_mapping_json and Path(args.network_mapping_json).exists():
            with open(args.network_mapping_json, "r") as f:
                network_name_map = {int(k): v for k, v in json.load(f).items()}
        else:
            network_name_map = {
                1: "primary visual", 2: "secondary visual", 3: "somatomotor", 4: "cingulo-opercular",
                5: "dorsal attention", 6: "language", 7: "frontparietal", 8: "auditory",
                9: "default mode", 10: "posterior multimodal", 11: "ventral multimodal", 12: "orbito-affective",
            }

    # Compose a single transform callable for convenience
    def unit_transform(A_reps):
        # A_reps: (n_reps, n_features)
        X = A_reps
        if avg_by_region is not None:
            X = avg_by_region(X)  # (n_reps, n_regions)
        if avg_by_network is not None:
            X = avg_by_network(X)  # (n_reps, 12)
        return X

    if args.unit == "voxel":
        unit_transform_callable = None
        unit_plot_name = "voxel"
    else:
        unit_transform_callable = unit_transform
        unit_plot_name = unit_name

    # ----- compute NC -----
    if args.method == "variance_based":
        nc = nc_variance_based(
            condition_betas,
            unit_transform=unit_transform_callable,
            n_participants=args.n_participants,
            rectify_signal=bool(args.rectify_signal),
            sqrt_to_r=bool(args.sqrt_to_r),
        )

        # Optional SE via sentence split-halves (like SI 5B)
        se_vals = None
        if args.nc_se_bootstrap and args.nc_se_bootstrap > 0:
            # Build condition means & noise variances once, then split conditions
            # For simplicity, recompute nc over random halves of conditions.
            boots = []
            cond_keys = [k for k, v in condition_betas.items() if len(v) >= 2]
            for _ in range(args.nc_se_bootstrap):
                perm = rng.permutation(len(cond_keys))
                halfA = {cond_keys[i]: condition_betas[cond_keys[i]] for i in perm[:len(perm)//2]}
                halfB = {cond_keys[i]: condition_betas[cond_keys[i]] for i in perm[len(perm)//2:]}
                try:
                    ncA = nc_variance_based(halfA, unit_transform_callable, args.n_participants,
                                            bool(args.rectify_signal), bool(args.sqrt_to_r))
                    ncB = nc_variance_based(halfB, unit_transform_callable, args.n_participants,
                                            bool(args.rectify_signal), bool(args.sqrt_to_r))
                    # SD across two independent estimates per iteration
                    boots.append(np.sqrt(((ncA - ncB) ** 2) / (2 - 1)))
                except SystemExit:
                    # not enough data in one half — skip
                    continue
            if boots:
                se_vals = np.nanmean(np.stack(boots, axis=0), axis=0)

    else:
        nc = nc_traditional(
            condition_betas,
            unit_transform=unit_transform_callable,
            n_bootstrap=args.n_bootstrap,
            min_reps=args.min_reps,
            rng=rng,
            standardize_halves=bool(args.standardize_halves),
            apply_spearman_brown_flag=bool(args.apply_spearman_brown),
        )

        se_vals = None  # not computed here

    # ----- save & plot -----
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.subject}_noise_ceiling_per_{unit_plot_name}_{args.method}"

    # Plot
    plt.figure(figsize=(10, 6))
    vals = nc[np.isfinite(nc)]
    plt.hist(vals, bins=60, edgecolor='black')
    xlabel = ("Noise ceiling (sqrt variance fraction)" if (args.method=="variance_based" and args.sqrt_to_r)
              else "Noise ceiling (variance fraction)" if args.method=="variance_based"
              else "Noise ceiling (Pearson r)")
    plt.title(f"Noise Ceiling Distribution ({args.subject}, per-{unit_plot_name}, {args.method})")
    plt.xlabel(xlabel)
    plt.ylabel(f"Number of {unit_plot_name}s"); plt.grid(True, alpha=0.3); plt.tight_layout()
    fig_path = outdir / f"{suffix}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()

    # Save NC
    nc_path = outdir / f"{suffix}.pkl"
    with open(nc_path, "wb") as f:
        pickle.dump({"nc": nc, "se": se_vals}, f)

    print(f"📊 Plot saved: {fig_path}")
    print(f"💾 Noise ceiling saved: {nc_path}")
    if se_vals is not None:
        print("Included per-unit SE (sentence split-halves).")

if __name__ == "__main__":
    main()
