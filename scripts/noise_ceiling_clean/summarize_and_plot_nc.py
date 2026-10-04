#!/usr/bin/env python3
"""Group summaries and the manuscript noise-ceiling figure from regenerated estimates.

Reads results/noise_ceiling/regenerated_clean/estimates/primary/<subj>/ and writes tables, source data,
the main figure (PDF/SVG/PNG) and supplementary diagnostics under figures/noise_ceiling/regenerated_clean/.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
WT = PROJECT / ".worktrees/nc_regen"
RES = WT / "results/noise_ceiling/regenerated_clean"
FIG = WT / "figures/noise_ceiling/regenerated_clean"
CANON_RES = PROJECT / "results/noise_ceiling/regenerated_clean"
CANON_FIG = PROJECT / "figures/noise_ceiling/regenerated_clean"
OLD = PROJECT / "results/noise_ceiling/final_version"
sys.path.insert(0, "/project/def-pbellec/xuan")
from figure_style.figure_utils import add_panel_labels, apply_clean_spines, get_figsize, save_figure

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
MESH = {
    "left": PROJECT / "data/fsLR_32k_midthickness/tpl-fsLR_hemi-L_den-32k_midthickness.surf.gii",
    "right": PROJECT / "data/fsLR_32k_midthickness/tpl-fsLR_hemi-R_den-32k_midthickness.surf.gii",
}
CMAP = LinearSegmentedColormap.from_list("nc", ["#F7F7F7", "#9BC8FA", "#3775BA", "#0F4D92", "#0c2458"])
BINS = np.linspace(0, 1, 21)
BLUE = "#3775BA"
GRAY = "#CFCECE"


def unit_stats(nc: np.ndarray, raw_r: np.ndarray | None = None) -> dict:
    v = nc[np.isfinite(nc)]
    q1, q3 = np.percentile(v, [25, 75])
    out = {
        "n_units": int(v.size),
        "mean_nc": float(v.mean()), "median_nc": float(np.median(v)), "sd_nc": float(v.std(ddof=1)),
        "iqr_nc": float(q3 - q1), "min_nc": float(v.min()), "max_nc": float(v.max()),
        "prop_zero": float((v == 0).mean()),
        "prop_gt_0.2": float((v > 0.2).mean()), "prop_gt_0.5": float((v > 0.5).mean()), "prop_gt_0.7": float((v > 0.7).mean()),
    }
    if raw_r is not None:
        out["frac_raw_r_negative"] = float(np.nanmean(raw_r < 0))
    return out


def load_primary(subj, res):
    d = RES / "estimates/primary" / subj
    df = pd.read_csv(d / f"unit_estimates_{res}.csv")
    raw = np.load(d / f"raw_r_{res}.npy")
    return df, raw


def paint(values_360, lab):
    v = np.full(lab.size, np.nan, np.float32)
    for i, val in enumerate(values_360):
        v[lab == (i + 1)] = val
    n_L = lab.size // 2
    return v[:n_L], v[n_L:]


def render_hemi(mesh, data, vmin, vmax, cmap):
    """Render one lateral view to an RGBA image. Zeros stay on the color scale; NaNs are medial wall."""
    fig = plt.figure(figsize=(2.4, 1.7))
    ax = fig.add_subplot(111, projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=str(mesh), stat_map=np.asarray(data, np.float32),
        hemi="left" if "hemi-L" in str(mesh) else "right", view="lateral",
        cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=ax, figure=fig,
        threshold=None, symmetric_cbar=False,
    )
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def render_both(values_360, lab, vmin, vmax, cmap):
    left, right = paint(values_360, lab)
    return np.concatenate([render_hemi(MESH["left"], left, vmin, vmax, cmap),
                           render_hemi(MESH["right"], right, vmin, vmax, cmap)], axis=1)


def mirror_csv(path: Path):
    """Keep a copy of small tables on the main project path as well as the worktree."""
    dest = CANON_RES / path.relative_to(RES)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (RES / "summaries").mkdir(parents=True, exist_ok=True)
    lab = np.asarray(nib.load(str(PROJECT / "data/Glasser_LR_Dense64k.dlabel.nii")).get_fdata()).reshape(-1).astype(int)
    net = np.loadtxt(PROJECT / "data/cortex_parcel_network_assignments.txt").astype(int)

    # ---------- tables ----------
    rows = []
    source_frames = []
    primary = {}
    for subj in SUBJECTS:
        primary[subj] = {}
        for res in ("region", "network"):
            df, raw = load_primary(subj, res)
            nc = df.nc_bounded_sb.to_numpy()
            if not np.all(np.isfinite(nc)):
                raise SystemExit(f"non-finite NC {subj} {res}")
            st = unit_stats(nc, raw)
            st.update(subject=subj, resolution=res, summary="subject")
            # sensitivity columns already on the unit table
            st["mean_r_fisher"] = float(df.r_fisher_mean.mean())
            st["mean_nc_arith"] = float(df.nc_from_arith_mean.mean())
            st["mean_nc_median"] = float(df.nc_from_median.mean())
            st["mean_nc_legacy_diag"] = float(df.nc_legacy_periter_sb_DIAGNOSTIC.mean())
            st["n_legacy_abs_gt_1"] = int(df.legacy_periter_abs_gt_1.sum())
            st["n_loco_dominated"] = int(df.loco_dominated_flag.sum())
            st["n_zeroed"] = int(df.zeroed_rbar_le_0.sum())
            rows.append(st)
            primary[subj][res] = (df, raw, nc)
            part = df[["subject", "resolution", "unit_index", "unit_name", "nc_bounded_sb", "r_fisher_mean",
                       "frac_iter_r_negative", "zeroed_rbar_le_0", "nc_from_arith_mean", "nc_from_median",
                       "loco_max_drop", "loco_dominant_condition", "loco_dominated_flag"]].copy()
            source_frames.append(part)
    subj_tbl = pd.DataFrame(rows)

    pooled_rows = []
    balanced_rows = []
    for res in ("region", "network"):
        blocks = [primary[s][res][2] for s in SUBJECTS]
        raws = [primary[s][res][1] for s in SUBJECTS]
        pooled = unit_stats(np.concatenate(blocks), np.concatenate(raws, axis=1))
        pooled.update(resolution=res, summary="pooled_spatial_units")
        pooled_rows.append(pooled)
        means = subj_tbl.loc[subj_tbl.resolution == res, "mean_nc"].to_numpy()
        medians = subj_tbl.loc[subj_tbl.resolution == res, "median_nc"].to_numpy()
        balanced_rows.append({
            "resolution": res, "summary": "subject_balanced",
            "mean_of_subject_means": float(means.mean()), "sd_of_subject_means": float(means.std(ddof=1)),
            "min_subject_mean": float(means.min()), "max_subject_mean": float(means.max()),
            "mean_of_subject_medians": float(medians.mean()),
            "mean_of_subject_prop_zero": float(subj_tbl.loc[subj_tbl.resolution == res, "prop_zero"].mean()),
            "mean_of_subject_frac_raw_r_negative": float(subj_tbl.loc[subj_tbl.resolution == res, "frac_raw_r_negative"].mean()),
        })
    subj_tbl.to_csv(RES / "summaries/subject_summaries.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(RES / "summaries/pooled_summaries.csv", index=False)
    pd.DataFrame(balanced_rows).to_csv(RES / "summaries/subject_balanced_summaries.csv", index=False)
    source = pd.concat(source_frames, ignore_index=True)
    source.to_csv(FIG / "source_data_noise_ceiling.csv", index=False)

    # variant comparison (subject-balanced mean of means)
    var_rows = []
    for name in ["primary", "labels_lookup", "trial_level_correct", "equalized_counts", "perm_one_half", "perm_both",
                 "exclude_extreme_psc", "inject_duplicates_DIAGNOSTIC", "legacy_misindexed", "seed_check_rerun"]:
        for res in ("region", "network"):
            means = []
            for subj in SUBJECTS:
                f = RES / "estimates" / name / subj / "config_and_summary.json"
                if not f.exists():
                    continue
                means.append(json.loads(f.read_text())["summaries"][res]["mean_nc"])
            if not means:
                continue
            var_rows.append({"variant": name, "resolution": res, "n_subjects": len(means),
                             "subject_balanced_mean": float(np.mean(means)), "sd_across_subjects": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                             "subject_means": ";".join(f"{m:.4f}" for m in means)})
    pd.DataFrame(var_rows).to_csv(RES / "summaries/variant_comparison.csv", index=False)

    # seed reproduction
    seed_rows = []
    for subj in SUBJECTS:
        for res in ("region", "network"):
            a = pd.read_csv(RES / "estimates/primary" / subj / f"unit_estimates_{res}.csv").nc_bounded_sb
            b = pd.read_csv(RES / "estimates/seed_check_rerun" / subj / f"unit_estimates_{res}.csv").nc_bounded_sb
            seed_rows.append({"subject": subj, "resolution": res, "max_abs_diff": float(np.max(np.abs(a - b)))})
    pd.DataFrame(seed_rows).to_csv(RES / "summaries/seed_reproduction.csv", index=False)

    # old manuscript numbers
    old_rows = []
    for subj in SUBJECTS:
        for res, unit in (("region", "region"), ("network", "network")):
            st = json.loads((OLD / f"{subj}_agg-task_unit-{unit}_stats.json").read_text())
            signed = np.load(OLD / f"{subj}_agg-task_unit-{unit}_signed.npy")
            old_rows.append({"subject": subj, "resolution": res, "old_mean_nc": st["mean_nc"], "old_sd": st["std_nc"],
                             "old_n_negative": st["n_negative"], "old_min": float(np.min(signed)), "old_max": float(np.max(signed))})
    old = pd.DataFrame(old_rows)
    old.to_csv(RES / "summaries/old_final_version_stats.csv", index=False)

    # ---------- main figure ----------
    plt.style.use("/project/def-pbellec/xuan/figure_style/nature_style.mplstyle")
    figsize, panel_w, panel_h = get_figsize("nature", "double", 5, 3, height_per_row_mm=42)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 5, hspace=0.55, wspace=0.35)
    hist_axes = []
    for i, subj in enumerate(SUBJECTS):
        ax = fig.add_subplot(gs[0, i])
        hist_axes.append(ax)
        nc = primary[subj]["region"][2]
        ax.hist(nc, bins=BINS, color=BLUE, edgecolor="white", linewidth=0.3)
        n0 = int((nc == 0).sum())
        if n0:
            ax.plot([0], [n0], "o", color="#850c0a", ms=3, zorder=5)
        med = float(np.median(nc))
        mu = float(nc.mean())
        ax.axvline(med, color="#0c2458", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(subj.replace("sub-", ""), fontsize=7, pad=2)
        ax.text(0.97, 0.95, f"median {med:.2f}\nmean {mu:.2f}\nzero {n0}", transform=ax.transAxes,
                ha="right", va="top", fontsize=5)
        apply_clean_spines(ax)
        if i == 0:
            ax.set_ylabel("Regions")
        ax.set_xlabel("Noise ceiling")
    map_axes = []
    for row, res in ((1, "region"), (2, "network")):
        for i, subj in enumerate(SUBJECTS):
            ax = fig.add_subplot(gs[row, i])
            map_axes.append(ax)
            if res == "region":
                vals = primary[subj]["region"][2]
            else:
                vals = primary[subj]["network"][2][net - 1]
            ax.imshow(render_both(vals, lab, 0, 1, CMAP))
            ax.set_axis_off()
            if i == 0:
                ax.set_title("Region" if res == "region" else "Network", fontsize=6, loc="left", pad=1)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.22])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("NC", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    add_panel_labels([hist_axes[0], map_axes[0], map_axes[5]], labels=["a", "b", "c"], x=-0.28, y=1.02, fontsize=8)
    fig.savefig(FIG / "noise_ceiling_main.png", dpi=600)
    save_figure(fig, FIG / "noise_ceiling_main", formats=("svg", "pdf"))
    plt.close(fig)

    # ---------- supplementary: both hemispheres, raw r, fraction negative, old vs new, equalized, loco ----------
    # old vs new distributions
    figsize2, _, _ = get_figsize("nature", "double", 2, 1, height_per_row_mm=55)
    fig, axes = plt.subplots(1, 2, figsize=figsize2)
    for ax, res in zip(axes, ("region", "network")):
        new = np.concatenate([primary[s][res][2] for s in SUBJECTS])
        oldv = np.concatenate([np.load(OLD / f"{s}_agg-task_unit-{res}_signed.npy") for s in SUBJECTS])
        ax.hist(oldv, bins=np.linspace(-1, 1, 41), color="#E9A6A1", alpha=0.85, label="previous", density=True)
        ax.hist(new, bins=BINS, color=BLUE, alpha=0.75, label="regenerated", density=True)
        ax.set_xlabel("Estimate")
        ax.set_title(res)
        apply_clean_spines(ax)
        ax.legend(frameon=False, fontsize=5)
    axes[0].set_ylabel("Density")
    save_figure(fig, FIG / "supp_old_vs_new", formats=("svg", "pdf"))
    fig.savefig(FIG / "supp_old_vs_new.png", dpi=600)
    plt.close(fig)

    # equalized comparison
    fig, ax = plt.subplots(figsize=get_figsize("nature", "single", 1, 1, height_per_row_mm=70)[0])
    xs, ys, labels = [], [], []
    for res, marker in (("region", "o"), ("network", "s")):
        for subj in SUBJECTS:
            a = json.loads((RES / "estimates/primary" / subj / "config_and_summary.json").read_text())["summaries"][res]["mean_nc"]
            b = json.loads((RES / "estimates/equalized_counts" / subj / "config_and_summary.json").read_text())["summaries"][res]["mean_nc"]
            ax.plot(a, b, marker, color=BLUE if res == "region" else "#0F4D92", ms=4)
        ax.plot([], [], marker, color=BLUE if res == "region" else "#0F4D92", label=res)
    lims = [0, 1]
    ax.plot(lims, lims, color="#CFCECE", lw=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("All unique trials (mean NC)")
    ax.set_ylabel("Equalized trial counts (mean NC)")
    ax.legend(frameon=False, fontsize=6)
    apply_clean_spines(ax)
    save_figure(fig, FIG / "supp_equalized_counts", formats=("svg", "pdf"))
    fig.savefig(FIG / "supp_equalized_counts.png", dpi=600)
    plt.close(fig)

    # LOCO
    loco = []
    for subj in SUBJECTS:
        for res in ("region", "network"):
            df = primary[subj][res][0]
            loco.append({"subject": subj, "resolution": res, "n_dominated": int(df.loco_dominated_flag.sum()),
                         "median_max_drop": float(df.loco_max_drop.median()),
                         "max_drop": float(df.loco_max_drop.max())})
    pd.DataFrame(loco).to_csv(RES / "summaries/loco_stability.csv", index=False)
    fig, ax = plt.subplots(figsize=get_figsize("nature", "single", 1, 1, height_per_row_mm=70)[0])
    locodf = pd.DataFrame(loco)
    width = 0.35
    x = np.arange(len(SUBJECTS))
    for j, res in enumerate(("region", "network")):
        y = [int(locodf[(locodf.subject == s) & (locodf.resolution == res)].n_dominated.iloc[0]) for s in SUBJECTS]
        ax.bar(x + (j - 0.5) * width, y, width=width, color=BLUE if res == "region" else "#9BC8FA", label=res)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sub-", "") for s in SUBJECTS])
    ax.set_ylabel("Units whose NC drops > 0.2\nwhen one variant is left out")
    ax.legend(frameon=False, fontsize=6)
    apply_clean_spines(ax)
    save_figure(fig, FIG / "supp_loco", formats=("svg", "pdf"))
    fig.savefig(FIG / "supp_loco.png", dpi=600)
    plt.close(fig)

    # raw r and fraction-negative, both hemispheres, region level
    figsize3, _, _ = get_figsize("nature", "double", 5, 2, height_per_row_mm=36)
    for kind, getter, vmin, vmax, cmap, label in (
        ("raw_r", lambda df: df.r_fisher_mean.to_numpy(), -0.2, 1.0, "coolwarm", "mean split-half r"),
        ("frac_negative", lambda df: df.frac_iter_r_negative.to_numpy(), 0, 1, "magma", "fraction of splits with r < 0"),
    ):
        fig = plt.figure(figsize=figsize3)
        gs = fig.add_gridspec(2, 5, hspace=0.15, wspace=0.05)
        for i, subj in enumerate(SUBJECTS):
            df = primary[subj]["region"][0]
            vals = getter(df)
            L, R = paint(vals, lab)
            for row, (hemi, data) in enumerate((("left", L), ("right", R))):
                ax = fig.add_subplot(gs[row, i])
                ax.imshow(render_hemi(MESH[hemi], data, vmin, vmax, cmap))
                ax.set_axis_off()
                if row == 0:
                    ax.set_title(subj.replace("sub-", ""), fontsize=6)
        fig.savefig(FIG / f"supp_region_{kind}.png", dpi=600)
        save_figure(fig, FIG / f"supp_region_{kind}", formats=("svg", "pdf"))
        plt.close(fig)

    config = {
        "journal": "nature", "span": "double", "figsize_in": list(figsize),
        "panel_width_mm": panel_w, "panel_height_mm": panel_h,
        "histogram_bins": BINS.tolist(), "color_scale_nc": [0, 1],
        "cmap": ["#F7F7F7", "#9BC8FA", "#3775BA", "#0F4D92", "#0c2458"],
        "histogram_annotation": "median, mean, and count of exact zeros",
        "brain_view": "left and right lateral views in each map cell; zeros remain on the [0, 1] scale",
        "primary_variant": "estimates/primary",
        "seed": 20260926, "n_iter": 50,
        "png_dpi": 600,
    }
    (FIG / "plot_config.json").write_text(json.dumps(config, indent=2))
    for p in (RES / "summaries").glob("*.csv"):
        mirror_csv(p)
    CANON_FIG.mkdir(parents=True, exist_ok=True)
    for p in FIG.iterdir():
        if p.is_file():
            (CANON_FIG / p.name).write_bytes(p.read_bytes())
    print(subj_tbl.to_string(index=False))
    print(pd.DataFrame(balanced_rows).to_string(index=False))
    print(pd.DataFrame(pooled_rows).to_string(index=False))
    print("seed", pd.DataFrame(seed_rows).to_string(index=False))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
