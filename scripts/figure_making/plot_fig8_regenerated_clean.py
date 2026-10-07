#!/usr/bin/env python3
"""Figure 8 (regenerated from the clean similarity analysis) + supplementary figures.

Reads ONLY outputs of scripts/similarity_analysis_clean.py (never the old production CSVs, except for the
'as published' reference markers in the audit figure, which are labelled as such).

Figures (PDF + SVG + PNG 600 dpi, each with a *_source_data.csv):
  fig8_within_across_similarity      3 panels (a vertex, b region, c network); x = DMS, 1-Back, InterDMS, CTXDM
  figS_network_matrices_family       5x5 network-level split-half matrices, one per task family (mean over tasks and splits)
  figS_network_matrices_tasks        5x5 network-level matrices for the 12 task variants
  figS_audit_variants                family-level within/across for: legacy pickles, fixed indexing (+duplicates), fixed+dedup (lookup labels),
                                     final (block-file-verified labels), plus the published production CSV values as hollow markers
  figS_sensitivity                   primary vs exclude-inconsistent-runs / behaviour-relabel / robust-PSC / session-scale variants
  figS_task_level                    12 task variants x 3 resolutions, within (mean over 5 subjects) and across (mean over 10 pairs)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/project/def-pbellec/xuan/figure_style")
from figure_utils import get_figsize, save_figure, apply_clean_spines, add_panel_labels
plt.style.use("/project/def-pbellec/xuan/figure_style/nature_style.mplstyle")
plt.rcParams["savefig.dpi"] = 600

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import paths  # outputs under OUTPUT_ROOT; never reads the legacy tree
# EXTERNAL style dependency: /project/def-pbellec/xuan/figure_style (separate repo; not legacy, not dataset)
REPO = paths.REPO_ROOT    # tracked published reference tables (results/similarity_analysis/with_dms) live in this repo
FAMILIES = ["DMS", "1-Back", "InterDMS", "CTXDM"]
RES = [("vertex", "Vertices (58,396)"), ("region", "Glasser regions (360)"), ("network", "Cole–Anticevic networks (12)")]
SUBJ = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
C_WITHIN, C_ACROSS = "#0C2458", "#B64342"
FORMATS = ("pdf", "svg", "png")
PROD = {"vertex": "voxel", "region": "region", "network": "network"}
PROD_TASK_FAMILY = {"dms_loc": "DMS", "nback_loc": "1-Back", "nback_ctg": "1-Back", "nback_obj": "1-Back", "ctxdm_lco": "CTXDM", "ctxdm_col": "CTXDM"}


def fam_of(t):
    return PROD_TASK_FAMILY.get(t, "InterDMS" if t.startswith("interdms") else t)


def within_across_panel(ax, task, fam, res, ylim, show_legend=False, title=None):
    tk = task[(task.spatial_resolution == res)]; fm = fam[(fam.spatial_resolution == res)]
    x = np.arange(len(FAMILIES)); off = 0.16
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    for i, f in enumerate(FAMILIES):
        t = tk[tk.task_family == f]; m = fm[fm.task_family == f].iloc[0]
        jit = np.linspace(-0.06, 0.06, len(t)) if len(t) > 1 else np.zeros(1)
        ax.scatter(x[i] - off + jit, t.within_mean, s=6, color=C_WITHIN, alpha=0.55, lw=0, zorder=2)
        ax.scatter(x[i] + off + jit, t.across_mean, s=6, color=C_ACROSS, alpha=0.55, lw=0, zorder=2)
        ax.scatter(x[i] - off, m.within_mean, s=28, color=C_WITHIN, edgecolor="white", lw=0.5, zorder=3)
        ax.scatter(x[i] + off, m.across_mean, s=28, color=C_ACROSS, edgecolor="white", lw=0.5, marker="s", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES); ax.set_ylim(*ylim); ax.set_xlim(-0.5, len(FAMILIES) - 0.5)
    ax.set_ylabel("Split-half similarity (Pearson r)")
    if title:
        ax.set_title(title, fontsize=7, pad=3)
    apply_clean_spines(ax)
    if show_legend:
        h = [Line2D([], [], marker="o", color=C_WITHIN, ls="", ms=5, label="Within-subject (family mean)"),
             Line2D([], [], marker="s", color=C_ACROSS, ls="", ms=5, label="Across-subject (family mean)"),
             Line2D([], [], marker="o", color="0.4", ls="", ms=2.5, alpha=0.7, label="Individual task variant")]
        ax.legend(handles=h, loc="upper right", frameon=False, fontsize=5.5, handletextpad=0.3, borderaxespad=0.2)


def fig_main(task, fam, out):
    task = task[task.variant == "clean_verified"]; fam = fam[fam.variant == "clean_verified"]
    lo = min(task.within_mean.min(), task.across_mean.min()); hi = max(task.within_mean.max(), task.across_mean.max())
    ylim = (min(-0.1, np.floor(lo * 10) / 10 - 0.05), min(1.0, np.ceil(hi * 10) / 10 + 0.05))
    figsize, _, _ = get_figsize("nature", "double", 3, 1, height_per_row_mm=52)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for k, (ax, (res, lab)) in enumerate(zip(axes, RES)):
        within_across_panel(ax, task, fam, res, ylim, show_legend=(k == 2), title=lab)
        if k:
            ax.set_ylabel("")
    add_panel_labels(axes, labels=["a", "b", "c"], x=-0.12, y=1.04, fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, out / "fig8_within_across_similarity", formats=FORMATS); plt.close(fig)
    src = task[["spatial_resolution", "task_family", "task", "task_label", "within_mean", "within_sd_subjects", "within_mc_sd_splits",
                "across_mean", "across_sd_pairs", "across_mc_sd_splits", "n_subjects", "n_pairs"]].copy(); src["level"] = "task"
    fsrc = fam[["spatial_resolution", "task_family", "within_mean", "within_sd_tasks", "across_mean", "across_sd_tasks", "n_tasks"]].copy(); fsrc["level"] = "family"
    pd.concat([src, fsrc], ignore_index=True).to_csv(out / "fig8_within_across_similarity_source_data.csv", index=False)


def load_mats(results, variant="clean_verified"):
    z = np.load(results / "matrices" / f"similarity_matrices_{variant}.npz", allow_pickle=True)
    return {res: z[res] for res, _ in RES}, list(z["tasks"])


def sym(M):
    """Symmetrise off-diagonal (0.5*(r_ij+r_ji)); keep diagonal."""
    S = 0.5 * (M + M.T); np.fill_diagonal(S, np.diag(M)); return S


def draw_matrix(ax, M, title, vmin=-1, vmax=1, annotate=True):
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([s[-2:] for s in SUBJ], fontsize=5); ax.set_yticklabels([s[-2:] for s in SUBJ], fontsize=5)
    ax.set_title(title, fontsize=6, pad=2)
    if annotate:
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4, color="white" if abs(M[i, j]) > 0.6 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_matrices(results, task_tab, out):
    mats, tasks = load_mats(results)
    Mn = np.nanmean(mats["network"], axis=1)     # (12 tasks, 5, 5) mean over splits
    rows = []
    figsize, _, _ = get_figsize("nature", "double", 4, 1, height_per_row_mm=48)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for ax, f in zip(axes, FAMILIES):
        idx = [i for i, t in enumerate(tasks) if fam_of(t) == f]
        M = sym(Mn[idx].mean(0)); im = draw_matrix(ax, M, f"{f} ({len(idx)} task{'s' if len(idx) > 1 else ''})")
        for i in range(5):
            for j in range(5):
                rows.append(dict(figure="family", task_family=f, task="", subject_row=SUBJ[i], subject_col=SUBJ[j], r=M[i, j]))
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); cb.set_label("Pearson r (network level)", fontsize=6); cb.ax.tick_params(labelsize=5)
    save_figure(fig, out / "figS_network_matrices_family", formats=FORMATS); plt.close(fig)

    figsize, _, _ = get_figsize("nature", "double", 4, 3, height_per_row_mm=46)
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    lab = task_tab[(task_tab.variant == "clean_verified") & (task_tab.spatial_resolution == "network")].set_index("task").task_label
    for ax, (i, t) in zip(axes.ravel(), enumerate(tasks)):
        M = sym(Mn[i]); im = draw_matrix(ax, M, lab.get(t, t))
        for a in range(5):
            for b in range(5):
                rows.append(dict(figure="task", task_family=fam_of(t), task=t, subject_row=SUBJ[a], subject_col=SUBJ[b], r=M[a, b]))
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label("Pearson r (network level)", fontsize=6); cb.ax.tick_params(labelsize=5)
    save_figure(fig, out / "figS_network_matrices_tasks", formats=FORMATS); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "figS_network_matrices_source_data.csv", index=False)


def fig_variants(fam, out, variants, names, fname, prod_ref=None, title=None):
    figsize, _, _ = get_figsize("nature", "double", 3, 1, height_per_row_mm=58)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    x = np.arange(len(FAMILIES)); nv = len(variants); wbar = 0.8 / nv
    alphas = np.linspace(0.35, 1.0, nv)
    rows = []
    for k, (ax, (res, lab)) in enumerate(zip(axes, RES)):
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        for vi, (v, name) in enumerate(zip(variants, names)):
            d = fam[(fam.variant == v) & (fam.spatial_resolution == res)].set_index("task_family").reindex(FAMILIES)
            xx = x - 0.4 + wbar * (vi + 0.5)
            ax.bar(xx, d.within_mean, wbar * 0.9, color=C_WITHIN, alpha=alphas[vi], lw=0, label=f"{name}" if k == 0 else None)
            ax.scatter(xx, d.across_mean, s=9, color=C_ACROSS, alpha=alphas[vi], marker="s", lw=0, zorder=3)
            for f in FAMILIES:
                rows.append(dict(spatial_resolution=res, variant=v, variant_label=name, task_family=f, within_mean=d.loc[f, "within_mean"], across_mean=d.loc[f, "across_mean"]))
        if prod_ref is not None:
            p = prod_ref[prod_ref.spatial_resolution == res]
            for f in FAMILIES:
                q = p[p.task_family == f]
                if len(q):
                    ax.scatter([x[FAMILIES.index(f)]], q.within_mean, s=16, facecolor="none", edgecolor=C_WITHIN, lw=0.7, zorder=4)
                    ax.scatter([x[FAMILIES.index(f)]], q.across_mean, s=16, facecolor="none", edgecolor=C_ACROSS, lw=0.7, marker="s", zorder=4)
                    rows.append(dict(spatial_resolution=res, variant="production_csv", variant_label="published production CSV (task-level mean; vertex capped at 300 trials)",
                                     task_family=f, within_mean=float(q.within_mean), across_mean=float(q.across_mean)))
        ax.set_xticks(x); ax.set_xticklabels(FAMILIES); ax.set_title(lab, fontsize=7, pad=3); apply_clean_spines(ax)
        if k == 0:
            ax.set_ylabel("Split-half similarity (Pearson r)")
    h = [plt.Rectangle((0, 0), 1, 1, color=C_WITHIN, alpha=a, label=n) for a, n in zip(alphas, names)]
    h += [Line2D([], [], marker="s", color=C_ACROSS, ls="", ms=3, label="Across-subject (squares)")]
    if prod_ref is not None:
        h += [Line2D([], [], marker="o", mfc="none", mec="0.2", ls="", ms=4, label="Published production values")]
    axes[-1].legend(handles=h, loc="upper right", frameon=False, fontsize=5, handlelength=1.2)
    if title:
        fig.suptitle(title, fontsize=7)
    add_panel_labels(axes, labels=["a", "b", "c"], x=-0.12, y=1.04, fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, out / fname, formats=FORMATS); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / f"{fname}_source_data.csv", index=False)


def production_reference():
    rows = []
    for res, key in PROD.items():
        p = REPO / f"results/similarity_analysis/with_dms/similarity_bootstrap_results_task_{key}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d["task_family"] = d.Task.map(fam_of)
        # production stored one row per task x type x bootstrap-iteration; task-level mean first, then family mean
        tl = d.groupby(["task_family", "Task", "Type"]).Similarity.mean().reset_index()
        fl = tl.groupby(["task_family", "Type"]).Similarity.mean().unstack()
        for f in fl.index:
            rows.append(dict(spatial_resolution=res, task_family=f, within_mean=fl.loc[f, "Within"], across_mean=fl.loc[f, "Across"]))
    return pd.DataFrame(rows)


def fig_task_level(task, out):
    t = task[task.variant == "clean_verified"]
    order = ["dms_loc", "nback_loc", "nback_ctg", "nback_obj", "interdms_loc_ABAB", "interdms_loc_ABBA", "interdms_ctg_ABAB", "interdms_ctg_ABBA",
             "interdms_obj_ABAB", "interdms_obj_ABBA", "ctxdm_lco", "ctxdm_col"]
    figsize, _, _ = get_figsize("nature", "double", 1, 3, height_per_row_mm=40)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    x = np.arange(len(order)); rows = []
    for ax, (res, lab) in zip(axes, RES):
        d = t[t.spatial_resolution == res].set_index("task").reindex(order)
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        ax.errorbar(x - 0.12, d.within_mean, yerr=d.within_sd_subjects, fmt="o", ms=3, color=C_WITHIN, lw=0.6, capsize=1.2, label="Within-subject (mean ± SD over 5 subjects)")
        ax.errorbar(x + 0.12, d.across_mean, yerr=d.across_sd_pairs, fmt="s", ms=3, color=C_ACROSS, lw=0.6, capsize=1.2, label="Across-subject (mean ± SD over 10 pairs)")
        for i in (0.5, 3.5, 9.5):
            ax.axvline(i, color="0.85", lw=0.5)
        ax.set_title(lab, fontsize=7, pad=2); apply_clean_spines(ax); ax.set_ylabel("Pearson r")
        rows.append(d.assign(spatial_resolution=res).reset_index())
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(d.task_label, rotation=40, ha="right", fontsize=5.5)
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper right")
    add_panel_labels(axes, labels=["a", "b", "c"], x=-0.06, y=1.02, fontsize=8)
    fig.tight_layout(h_pad=0.8)
    save_figure(fig, out / "figS_task_level", formats=FORMATS); plt.close(fig)
    pd.concat(rows)[["spatial_resolution", "task", "task_label", "task_family", "within_mean", "within_sd_subjects", "within_mc_sd_splits",
                     "across_mean", "across_sd_pairs", "across_mc_sd_splits"]].to_csv(out / "figS_task_level_source_data.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="default: <output_root>/similarity_analysis/regenerated_clean")
    ap.add_argument("--out", default=None, help="default: <output_root>/figures/fig8_regenerated_clean")
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--output_root", default=None)
    args = ap.parse_args()
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard("plot_fig8_regenerated_clean")
    args.results = str(paths.require_not_legacy(args.results or paths.output_dir("similarity_analysis", "regenerated_clean"), "--results"))
    args.out = str(paths.require_not_legacy(args.out or paths.output_dir("figures", "fig8_regenerated_clean"), "--out"))
    results = Path(args.results); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    task = pd.read_csv(results / "summary_task_level.csv"); fam = pd.read_csv(results / "summary_family_level.csv")
    fig_main(task, fam, out)
    fig_matrices(results, task, out)
    aud = [(v, n) for v, n in zip(["legacy_pickle", "fixed_with_duplicates", "clean_lookup", "clean_verified"],
                                  ["Original pickles (mis-indexed, duplicated)", "Fixed indexing, duplicates kept", "Fixed + de-duplicated (lookup labels)", "Final (block-file-verified labels)"])
           if v in set(fam.variant.unique())]   # legacy_pickle is absent when the invalid pickles are not supplied
    fig_variants(fam, out, [v for v, _ in aud], [n for _, n in aud], "figS_audit_variants", prod_ref=production_reference())
    sens_v = ["clean_verified", "clean_exclude_inconsistent", "clean_behaviour_relabel", "clean_verified_robust_psc", "clean_verified_sessionscale"]
    sens_n = ["Primary (clean, verified labels)", "Exclude behaviour-inconsistent runs", "Relabel by behaviour (sensitivity)", "Exclude extreme-PSC vertices", "Random per-session scaling"]
    have = set(fam.variant.unique())
    keep = [(v, n) for v, n in zip(sens_v, sens_n) if v in have]
    fig_variants(fam, out, [v for v, _ in keep], [n for _, n in keep], "figS_sensitivity")
    fig_task_level(task, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
