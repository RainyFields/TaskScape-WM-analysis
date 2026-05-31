#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser(
        description="Compute subject-level average R2 map across all runs/sessions at Glasser region level."
    )
    p.add_argument("--subj", required=True, help="Subject ID, e.g., sub-03")
    p.add_argument(
        "--h5_root",
        default="/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas",
        help="Root that contains <subj>/<ses>/func/*_betas.h5",
    )
    p.add_argument(
        "--atlas_path",
        default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii",
        help="Glasser dense label file",
    )
    p.add_argument(
        "--metric",
        default="r2_full",
        choices=["r2_full", "r2_noise", "r2_task", "r2_full_adj", "r2_noise_adj", "r2_task_adj"],
        help="R2 metric to average across runs. If r2_full is missing, it is reconstructed from yhat+resid.",
    )
    p.add_argument(
        "--journal",
        default="nature",
        choices=["nature", "pnas", "cell", "science", "plos", "neuron", "jneurosci"],
    )
    p.add_argument("--span", default="double", choices=["single", "1.5", "double"])
    p.add_argument(
        "--out_dir",
        default="/project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc",
        help="Output directory for figure + summaries",
    )
    return p.parse_args()


def load_figure_style_utils():
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from figure_style.figure_utils import get_figsize, save_figure

    return project_root, get_figsize, save_figure


def fetch_surfaces():
    try:
        fs = datasets.fetch_surf_fsaverage(mesh="fsLR32k")
        return Path(fs["infl_left"]), Path(fs["infl_right"]), "fsLR32k"
    except Exception:
        fs = datasets.fetch_surf_fsaverage(mesh="fsaverage6")
        return Path(fs["infl_left"]), Path(fs["infl_right"]), "fsaverage6"


def parse_task_from_name(name: str) -> str:
    m = re.search(r"task-([^_]+)", name)
    return m.group(1) if m else "unknown"


def load_region_labels(atlas_path: Path) -> np.ndarray:
    atlas = nib.load(str(atlas_path)).get_fdata()
    if atlas.ndim == 2:
        labels = atlas[0]
    else:
        labels = np.squeeze(atlas)
    return labels.astype(np.int32)


def compute_region_averages(vertex_map: np.ndarray, labels: np.ndarray, n_parcels: int = 360) -> np.ndarray:
    region_vals = np.full((n_parcels,), np.nan, dtype=np.float32)
    for rid in range(1, n_parcels + 1):
        idx = labels == rid
        if np.any(idx):
            region_vals[rid - 1] = np.nanmean(vertex_map[idx])
    return region_vals


def project_region_to_surface(region_vals: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.full(labels.shape, np.nan, dtype=np.float32)
    n_parcels = region_vals.shape[0]
    for rid in range(1, n_parcels + 1):
        out[labels == rid] = region_vals[rid - 1]
    return out


def split_hemispheres(values: np.ndarray):
    return values[:32492], values[32492:64984]


def maybe_resample(metric: np.ndarray, target_n: int) -> np.ndarray:
    if metric.shape[0] == target_n:
        return metric.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, metric.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_n, dtype=np.float64)
    finite = np.isfinite(metric)
    if np.sum(finite) < 2:
        return np.full((target_n,), np.nan, dtype=np.float32)
    y = np.interp(x_new, x_old[finite], metric[finite].astype(np.float64))
    return y.astype(np.float32)


def reconstruct_r2_full(yhat: np.ndarray, resid: np.ndarray) -> np.ndarray:
    y = yhat.astype(np.float64) + resid.astype(np.float64)
    ss_res = np.sum(resid.astype(np.float64) ** 2, axis=0)
    y_center = y - np.mean(y, axis=0, keepdims=True)
    ss_tot = np.sum(y_center**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)
    r2[~np.isfinite(r2)] = np.nan
    return r2.astype(np.float32)


def main():
    args = get_args()
    project_root, get_figsize, save_figure = load_figure_style_utils()
    plt.style.use(str(project_root / "figure_style" / "nature_style.mplstyle"))

    subj_root = Path(args.h5_root) / args.subj
    h5_files = sorted(subj_root.rglob("*_betas.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found under: {subj_root}")

    atlas_labels = load_region_labels(Path(args.atlas_path))
    if atlas_labels.shape[0] != 64984:
        raise RuntimeError(f"Unexpected atlas size: {atlas_labels.shape[0]} (expected 64984)")

    run_metric_maps = []
    task_counts: dict[str, int] = {}
    n_direct = 0
    n_reconstructed = 0
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            if args.metric in f:
                metric = f[args.metric][:].astype(np.float32)
                n_direct += 1
            elif args.metric == "r2_full" and "yhat" in f and "resid" in f:
                metric = reconstruct_r2_full(f["yhat"][:], f["resid"][:])
                n_reconstructed += 1
            else:
                continue
            valid = f["valid_mask"][:].astype(bool) if "valid_mask" in f else np.ones_like(metric, dtype=bool)
        if metric.shape[0] != 64984:
            continue
        metric = metric.copy()
        metric[~valid] = np.nan
        run_metric_maps.append(metric)
        task = parse_task_from_name(h5_path.name)
        task_counts[task] = task_counts.get(task, 0) + 1

    if not run_metric_maps:
        raise RuntimeError(f"No valid run maps found with metric '{args.metric}'")

    stack = np.stack(run_metric_maps, axis=0)
    vertex_avg = np.nanmean(stack, axis=0).astype(np.float32)
    region_avg = compute_region_averages(vertex_avg, atlas_labels, n_parcels=360)
    region_projected = project_region_to_surface(region_avg, atlas_labels)

    lh, rh = split_hemispheres(region_projected)
    lh_surf, rh_surf, surf_source = fetch_surfaces()
    n_lh = int(nib.load(str(lh_surf)).darrays[0].data.shape[0])
    n_rh = int(nib.load(str(rh_surf)).darrays[0].data.shape[0])
    lh_map = maybe_resample(lh, n_lh)
    rh_map = maybe_resample(rh, n_rh)

    finite_region = region_avg[np.isfinite(region_avg)]
    vmin = float(np.nanpercentile(finite_region, 1)) if finite_region.size else 0.0
    vmax = float(np.nanpercentile(finite_region, 99)) if finite_region.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0.0, 1.0

    figsize, _, _ = get_figsize(args.journal, args.span, n_cols=2, n_rows=2)
    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw={"projection": "3d"})
    plotting.plot_surf_stat_map(
        str(lh_surf),
        lh_map,
        hemi="left",
        view="lateral",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        axes=axes[0, 0],
        title=f"{args.metric} region-avg — LH lateral",
    )
    plotting.plot_surf_stat_map(
        str(rh_surf),
        rh_map,
        hemi="right",
        view="lateral",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        axes=axes[0, 1],
        title=f"{args.metric} region-avg — RH lateral",
    )
    plotting.plot_surf_stat_map(
        str(lh_surf),
        lh_map,
        hemi="left",
        view="medial",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        axes=axes[1, 0],
        title=f"{args.metric} region-avg — LH medial",
    )
    plotting.plot_surf_stat_map(
        str(rh_surf),
        rh_map,
        hemi="right",
        view="medial",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        axes=axes[1, 1],
        title=f"{args.metric} region-avg — RH medial",
    )
    fig.suptitle(
        f"{args.subj} | region-level average of {args.metric}\n"
        f"all sessions/runs (n_runs={stack.shape[0]}), surface={surf_source}",
        y=0.995,
    )
    fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.02, wspace=0.05, hspace=0.08)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{args.subj}_all_sessions_runs_{args.metric}_region_avg_surface"
    save_figure(fig, stem, formats=("svg", "pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    np.save(out_dir / f"{args.subj}_all_sessions_runs_{args.metric}_vertex_avg.npy", vertex_avg)
    np.save(out_dir / f"{args.subj}_all_sessions_runs_{args.metric}_region_avg.npy", region_avg)

    summary_txt = out_dir / f"{args.subj}_all_sessions_runs_{args.metric}_region_avg_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"subject: {args.subj}\n")
        f.write(f"metric: {args.metric}\n")
        f.write(f"n_runs_used: {stack.shape[0]}\n")
        f.write(f"n_metric_direct: {n_direct}\n")
        f.write(f"n_metric_reconstructed: {n_reconstructed}\n")
        f.write(f"h5_root: {Path(args.h5_root)}\n")
        f.write(f"atlas: {Path(args.atlas_path)}\n")
        f.write(f"surface_source: {surf_source}\n")
        f.write("task_counts:\n")
        for task, count in sorted(task_counts.items()):
            f.write(f"  {task}: {count}\n")
        f.write(f"region_mean: {float(np.nanmean(region_avg)):.6f}\n")
        f.write(f"region_std: {float(np.nanstd(region_avg)):.6f}\n")
        f.write(f"region_min: {float(np.nanmin(region_avg)):.6f}\n")
        f.write(f"region_max: {float(np.nanmax(region_avg)):.6f}\n")

    print(f"[save] figure_svg: {stem.with_suffix('.svg')}")
    print(f"[save] figure_pdf: {stem.with_suffix('.pdf')}")
    print(f"[save] figure_png: {stem.with_suffix('.png')}")
    print(f"[save] region_avg_npy: {out_dir / f'{args.subj}_all_sessions_runs_{args.metric}_region_avg.npy'}")
    print(f"[save] summary: {summary_txt}")
    print(
        f"[stats] n_runs={stack.shape[0]} (direct={n_direct}, reconstructed={n_reconstructed}) "
        f"region_mean={float(np.nanmean(region_avg)):.6f} region_std={float(np.nanstd(region_avg)):.6f}"
    )


if __name__ == "__main__":
    main()
