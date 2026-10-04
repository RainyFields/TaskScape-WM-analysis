#!/usr/bin/env python3
"""Style-matched Figure 8 from the corrected clean_verified similarity tables.

Visual language follows the June 2026 multipanel boxplot (task-colored within boxes,
subject markers, right-hand legends) but:

  * reads ONLY results/similarity_analysis/regenerated_clean/ (variant clean_verified);
  * boxplots are built from 5 subject-level estimates after averaging the 50 split-half
    iterations — never from the iterations themselves;
  * no paired t-test, stars, or significance brackets;
  * across-subject boxes are light grey (not a washed family colour);
  * family-level means are a separate larger-marker layer.

Outputs go to figures/fig8_regenerated_clean/style_matched_v2/ and do not overwrite
the existing canonical three-panel family figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
RESULTS = PROJECT / "results/similarity_analysis/regenerated_clean"
VARIANT = "clean_verified"

# Canonical task order from scripts/similarity_analysis_clean.py (do not reorder keys).
TASKS = [
    "dms_loc",
    "nback_loc", "nback_ctg", "nback_obj",
    "interdms_loc_ABAB", "interdms_loc_ABBA",
    "interdms_ctg_ABAB", "interdms_ctg_ABBA",
    "interdms_obj_ABAB", "interdms_obj_ABBA",
    "ctxdm_lco", "ctxdm_col",
]
# Axis labels name the variant only; family is encoded by colour (see legend).
PRETTY = {
    "dms_loc": "DMS",
    "nback_loc": "location",
    "nback_ctg": "category",
    "nback_obj": "identity",
    "interdms_loc_ABAB": "loc. ABAB",
    "interdms_loc_ABBA": "loc. ABBA",
    "interdms_ctg_ABAB": "ctg. ABAB",
    "interdms_ctg_ABBA": "ctg. ABBA",
    "interdms_obj_ABAB": "id. ABAB",
    "interdms_obj_ABBA": "id. ABBA",
    "ctxdm_lco": "LCI",
    "ctxdm_col": "CIL",
}
FULL_LABEL = {
    "dms_loc": "DMS",
    "nback_loc": "1-Back location",
    "nback_ctg": "1-Back category",
    "nback_obj": "1-Back identity",
    "interdms_loc_ABAB": "InterDMS location ABAB",
    "interdms_loc_ABBA": "InterDMS location ABBA",
    "interdms_ctg_ABAB": "InterDMS category ABAB",
    "interdms_ctg_ABBA": "InterDMS category ABBA",
    "interdms_obj_ABAB": "InterDMS identity ABAB",
    "interdms_obj_ABBA": "InterDMS identity ABBA",
    "ctxdm_lco": "CTXDM LCI",
    "ctxdm_col": "CTXDM CIL",
}
FAMILY = {
    "dms_loc": "DMS",
    "nback_loc": "1-Back", "nback_ctg": "1-Back", "nback_obj": "1-Back",
    "ctxdm_lco": "CTXDM", "ctxdm_col": "CTXDM",
    **{t: "InterDMS" for t in TASKS if t.startswith("interdms")},
}
FAMILIES = ["DMS", "1-Back", "InterDMS", "CTXDM"]
FAMILY_COLOR = {
    "DMS": "#377EB8",
    "1-Back": "#FFD21F",
    "InterDMS": "#A04A9B",
    "CTXDM": "#4899A5",
}
ACROSS_FACE = "#D0D0D0"
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
SUBJECT_MARKERS = {"sub-01": "o", "sub-02": "s", "sub-03": "^", "sub-05": "D", "sub-06": "v"}
RESOLUTIONS = [
    ("vertex", "Vertex Level"),
    ("region", "Region Level"),
    ("network", "Network Level"),
]

FIGSIZE = (7.5, 9.3)
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "legend.title_fontsize": 14,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
}


def load_clean(results: Path):
    w = pd.read_csv(results / "summary_subject_task.csv")
    a = pd.read_csv(results / "summary_pair_task.csv")
    fam = pd.read_csv(results / "summary_family_level.csv")
    task = pd.read_csv(results / "summary_task_level.csv")
    for df in (w, a, fam, task):
        if "variant" in df.columns:
            df.drop(index=df.index[df.variant != VARIANT], inplace=True)
    if w.empty or a.empty or fam.empty or task.empty:
        raise SystemExit(f"No {VARIANT} rows in the clean tables.")
    return w, a, fam, task


def subject_across(pairs: pd.DataFrame) -> pd.DataFrame:
    """One across-subject value per subject × task × resolution.

    For subject s, mean of the four unordered-pair estimates that include s
    (each pair estimate is already the mean of 50 split-half iterations).
    Averaging those five subject values recovers the 10-pair mean exactly,
    because each pair is counted once in each of its two members.
    """
    rows = []
    for res in [r for r, _ in RESOLUTIONS]:
        for task in TASKS:
            sub = pairs[(pairs.spatial_resolution == res) & (pairs.task == task)]
            for s in SUBJECTS:
                vals = sub[(sub.subject_1 == s) | (sub.subject_2 == s)]["mean"]
                if len(vals) != 4:
                    raise AssertionError(f"{res} {task} {s}: expected 4 pairs, got {len(vals)}")
                rows.append(dict(
                    spatial_resolution=res, task=task, task_family=FAMILY[task],
                    subject=s, comparison="across", r=float(vals.mean()),
                    n_pairs=int(len(vals)),
                ))
    return pd.DataFrame(rows)


def subject_within(within: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in within.itertuples():
        rows.append(dict(
            spatial_resolution=r.spatial_resolution, task=r.task, task_family=r.task_family,
            subject=r.subject, comparison="within", r=float(r.mean), n_pairs=np.nan,
        ))
    return pd.DataFrame(rows)


def task_x_positions():
    """x positions with extra gaps between the four families."""
    xs, x = [], 0.0
    last_fam = None
    for t in TASKS:
        fam = FAMILY[t]
        if last_fam is not None and fam != last_fam:
            x += 0.70
        xs.append(x)
        x += 1.15
        last_fam = fam
    return np.asarray(xs)


def style_boxplot(bp, facecolor, lw=1.6):
    box = bp["boxes"][0]
    box.set_facecolor(facecolor)
    box.set_edgecolor("black")
    box.set_linewidth(lw)
    box.set_alpha(1.0)
    for key in ("whiskers", "caps", "medians"):
        for line in bp[key]:
            line.set_color("black")
            line.set_linewidth(lw)
    for line in bp["medians"]:
        line.set_solid_capstyle("butt")


def draw_panel(ax, long_df, fam, res, xs, ylim, show_xticks):
    offset, width = 0.22, 0.36
    jitter = np.linspace(-0.05, 0.05, len(SUBJECTS))
    sub = long_df[long_df.spatial_resolution == res]
    fm = fam[fam.spatial_resolution == res].set_index("task_family")

    for fam_name in FAMILIES:
        idx = [i for i, t in enumerate(TASKS) if FAMILY[t] == fam_name]
        ax.axvspan(xs[idx[0]] - 0.55, xs[idx[-1]] + 0.55,
                   facecolor=FAMILY_COLOR[fam_name], alpha=0.06, zorder=0, lw=0)

    for i, task in enumerate(TASKS):
        fam_name = FAMILY[task]
        color = FAMILY_COLOR[fam_name]
        w = sub[(sub.task == task) & (sub.comparison == "within")].set_index("subject").loc[SUBJECTS, "r"].to_numpy()
        a = sub[(sub.task == task) & (sub.comparison == "across")].set_index("subject").loc[SUBJECTS, "r"].to_numpy()
        bp_w = ax.boxplot(
            [w], positions=[xs[i] - offset], widths=width, patch_artist=True,
            showfliers=False, manage_ticks=False, whis=(0, 100),
        )
        style_boxplot(bp_w, color)
        bp_a = ax.boxplot(
            [a], positions=[xs[i] + offset], widths=width, patch_artist=True,
            showfliers=False, manage_ticks=False, whis=(0, 100),
        )
        style_boxplot(bp_a, ACROSS_FACE)
        for j, s in enumerate(SUBJECTS):
            ax.scatter(
                xs[i] - offset + jitter[j], w[j],
                marker=SUBJECT_MARKERS[s], s=52, facecolors="white",
                edgecolors="black", linewidths=1.15, zorder=5,
            )
            ax.scatter(
                xs[i] + offset + jitter[j], a[j],
                marker=SUBJECT_MARKERS[s], s=52, facecolors="white",
                edgecolors="black", linewidths=1.15, zorder=5,
            )

    for fam_name in FAMILIES:
        idx = [i for i, t in enumerate(TASKS) if FAMILY[t] == fam_name]
        # Left of the family group so the 12 task boxes stay unobstructed.
        xc = float(xs[idx[0]]) - 0.62
        row = fm.loc[fam_name]
        ax.scatter(
            xc, float(row.within_mean),
            s=120, marker="o", facecolors=FAMILY_COLOR[fam_name],
            edgecolors="black", linewidths=1.2, zorder=4,
        )
        ax.scatter(
            xc, float(row.across_mean),
            s=100, marker="s", facecolors=ACROSS_FACE,
            edgecolors="black", linewidths=1.2, zorder=4,
        )

    ax.axhline(0.0, color="0.15", lw=1.2, zorder=0)
    ax.yaxis.grid(True, color="0.92", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(*ylim)
    ax.set_xlim(xs[0] - 1.05, xs[-1] + 0.75)
    ax.set_ylabel("Similarity (Pearson r)")
    ax.set_xticks(xs)
    if show_xticks:
        ax.set_xticklabels([PRETTY[t] for t in TASKS], rotation=36, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)


def draw_legends(ax):
    ax.set_axis_off()
    fam_h = [
        Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", linewidth=1.2, label=f)
        for f in FAMILIES
    ]
    cmp_h = [
        Patch(facecolor="#6A6A6A", edgecolor="black", linewidth=1.2, label="Within-subject"),
        Patch(facecolor=ACROSS_FACE, edgecolor="black", linewidth=1.2, label="Across-subject"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#377EB8",
               markeredgecolor="black", markeredgewidth=1.1, markersize=11, linestyle="None",
               label="Family mean (within)"),
        Line2D([], [], marker="s", color="w", markerfacecolor=ACROSS_FACE,
               markeredgecolor="black", markeredgewidth=1.1, markersize=10, linestyle="None",
               label="Family mean (across)"),
    ]
    sub_h = [
        Line2D([], [], marker=SUBJECT_MARKERS[s], color="w", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.15, markersize=8.5,
               linestyle="None", label=s.replace("sub-", "Sub "))
        for s in SUBJECTS
    ]
    kw = dict(frameon=False, borderaxespad=0.0, labelspacing=0.35, handletextpad=0.5)
    leg1 = ax.legend(handles=fam_h, title="Task family", loc="upper left",
                     bbox_to_anchor=(0.0, 1.02), **kw)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=cmp_h, title="Comparison", loc="upper left",
                     bbox_to_anchor=(0.0, 0.62), **kw)
    ax.add_artist(leg2)
    ax.legend(handles=sub_h, title="Subject", loc="upper left",
              bbox_to_anchor=(0.0, 0.26), **kw)


def validate(long_df, fam, task):
    """Plotted family means must match summary_family_level.csv to float tolerance."""
    failures = []
    for res, _ in RESOLUTIONS:
        for fam_name in FAMILIES:
            row = fam[(fam.spatial_resolution == res) & (fam.task_family == fam_name)].iloc[0]
            sub = long_df[(long_df.spatial_resolution == res) & (long_df.task_family == fam_name)]
            w_task = (
                sub[sub.comparison == "within"].groupby("task").r.mean().mean()
            )
            a_subj = sub[sub.comparison == "across"].groupby("subject").r.mean()
            # family within = mean of task-level means (each task = mean of 5 subjects)
            if not np.isclose(w_task, row.within_mean, rtol=0, atol=1e-12):
                failures.append(f"{res} {fam_name} within {w_task:.12f} != {row.within_mean:.12f}")
            # family across = mean of task-level 10-pair means = mean of 5 subject-across values
            # (each subject-across is the mean of 4 pairs; averaging subjects = averaging pairs)
            a_from_subj = (
                sub[sub.comparison == "across"].groupby("task").r.mean().mean()
            )
            if not np.isclose(a_from_subj, row.across_mean, rtol=0, atol=1e-12):
                failures.append(f"{res} {fam_name} across {a_from_subj:.12f} != {row.across_mean:.12f}")
            trow = task[(task.spatial_resolution == res) & (task.task_family == fam_name)]
            if not np.isclose(trow.within_mean.mean(), row.within_mean, rtol=0, atol=1e-12):
                failures.append(f"{res} {fam_name} task-table within mismatch")
            if not np.isclose(trow.across_mean.mean(), row.across_mean, rtol=0, atol=1e-12):
                failures.append(f"{res} {fam_name} task-table across mismatch")
    n_tasks = long_df.task.nunique()
    n_subj = set(long_df.subject.unique())
    if n_tasks != 12:
        failures.append(f"expected 12 tasks, got {n_tasks}")
    if n_subj != set(SUBJECTS):
        failures.append(f"subject set {n_subj}")
    if (long_df.r < 0).sum() == 0:
        failures.append("no negative subject-level values (unexpected for this dataset)")
    if failures:
        raise AssertionError("validation failed:\n  " + "\n  ".join(failures))
    return True


def save_all(fig, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig8_within_across_similarity_style_matched_v2"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            f"{stem}.{ext}", format=ext, dpi=600,
            facecolor="white", edgecolor="white", transparent=False,
            bbox_inches="tight", pad_inches=0.18,
        )
    # Force an opaque RGB PNG (no alpha) so Overleaf cannot treat it as transparent.
    from PIL import Image
    png = Path(f"{stem}.png")
    im = Image.open(png).convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1])
    bg.save(png, dpi=(600, 600))
    return stem


def write_readme(out: Path, ylim, n_neg):
    text = f"""# Figure 8 style-matched v2

Revised layout matching the June 2026 12-task boxplot (three stacked panels, family
colours, subject markers, right-hand legends). Numerical values are the corrected
`clean_verified` analysis only.

## Inputs (canonical, corrected)

- `{RESULTS}/summary_subject_task.csv` — within-subject *r*, already averaged over 50 splits
- `{RESULTS}/summary_pair_task.csv` — 10 unordered pair *r*, already averaged over 50 splits
- `{RESULTS}/summary_task_level.csv` — task-level means (validation only)
- `{RESULTS}/summary_family_level.csv` — family-level means (validation + summary markers)

Variant: `{VARIANT}` only. Legacy pickles, `with_dms/`, and `similarity_bootstrap_results_*.csv` are not read.

## Aggregation

1. Split iterations are averaged first, inside the clean analysis tables. This script never opens the 50-iteration long tables for plotting.
2. **Within-subject, one subject × task × resolution:** `summary_subject_task.mean` (mean of 50 split-half diagonals).
3. **Across-subject, one subject × task × resolution:** mean of the four pair estimates that include that subject. Each pair estimate is already `0.5 * [r(i_A, j_B) + r(j_A, i_B)]` averaged over 50 splits. Averaging the five subject values recovers the 10-pair mean (each pair is counted once for each of its two members).
4. Boxplots (median, hinges, whiskers to min/max) are built from those **five subject-level values**. They are not built from split iterations.
5. Family-mean markers are the `summary_family_level.csv` values (mean of the task-level estimates). They are a separate layer and do not replace the 12 task boxplots.

## Visual encoding

| Family | Colour |
|---|---|
| DMS | `#377EB8` |
| 1-Back | `#FFD21F` |
| InterDMS | `#A04A9B` |
| CTXDM | `#4899A5` |

- Within-subject box: family colour, black outline and median (1.6 pt).
- Across-subject box: `{ACROSS_FACE}`, black outline and median (1.6 pt).
- Subject markers (white fill, 1.15 pt black edge, ~7 pt): Sub 01 circle, Sub 02 square, Sub 03 up-triangle, Sub 05 diamond, Sub 06 down-triangle. Deterministic horizontal jitter `linspace(-0.045, 0.045, 5)` in subject order.
- Family-mean layer: filled circle (within) / square (across), ~10 pt, white edge, placed at the x-centre of each family group.

## Figure dimensions and type

- `figsize = {FIGSIZE}` inches (manuscript width unknown; Nature double column is 183 mm ≈ 7.2 in).
- Fonts: titles 16 pt, axis labels 15 pt, y-ticks 13 pt, x-ticks 12 pt, legend titles 14 pt, legend entries 12.5 pt.
- x-axis labels name the variant only (`location`, `loc. ABAB`, `LCI`, …). Family identity is the colour band / Task-family legend, so 12 pt labels do not collide. Full names are in the source CSV.
- Shared y-limits: `{ylim[0]:.3f}` to `{ylim[1]:.3f}` (includes all subject-level values; {n_neg} negative points).
- PNG saved with `facecolor="white", transparent=False`.

## Not included

No paired *t*-test, significance stars, split-iteration point cloud, or legacy numerical values.
"""
    (out / "README.md").write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(PROJECT / "figures/fig8_regenerated_clean/style_matched_v2"))
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)

    w_tab, pair_tab, fam, task = load_clean(results)
    long_df = pd.concat([subject_within(w_tab), subject_across(pair_tab)], ignore_index=True)
    validate(long_df, fam, task)

    xs = task_x_positions()
    ymin, ymax = float(long_df.r.min()), float(long_df.r.max())
    pad = max(0.06, 0.08 * (ymax - ymin))
    ylim = (min(-0.15, ymin - pad), min(1.05, ymax + pad))

    plt.rcParams.update(RC)
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.0, 0.30], height_ratios=[1, 1, 1],
        left=0.12, right=0.97, top=0.965, bottom=0.20, wspace=0.06, hspace=0.20,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for i, ((res, title), ax) in enumerate(zip(RESOLUTIONS, axes)):
        draw_panel(ax, long_df, fam, res, xs, ylim, show_xticks=(i == 2))
        ax.set_title(title, pad=8)
        ax.text(-0.11, 1.06, ["a", "b", "c"][i], transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")
    draw_legends(fig.add_subplot(gs[:, 1]))

    stem = save_all(fig, out)
    src = long_df.copy()
    src["task_label"] = src.task.map(FULL_LABEL)
    src["axis_label"] = src.task.map(PRETTY)
    src["variant"] = VARIANT
    src["aggregation"] = src.comparison.map({
        "within": "mean of 50 split-half diagonals for this subject",
        "across": "mean of 4 pair estimates that include this subject (pairs already mean of 50 splits)",
    })
    fam_src = fam[["spatial_resolution", "task_family", "within_mean", "across_mean", "n_tasks"]].copy()
    fam_src["level"] = "family_summary"
    src.to_csv(out / "fig8_within_across_similarity_style_matched_v2_source_data.csv", index=False)
    fam_src.to_csv(out / "fig8_family_means_source_data.csv", index=False)
    write_readme(out, ylim, int((long_df.r < 0).sum()))
    meta = dict(
        variant=VARIANT, figsize_in=list(FIGSIZE), ylim=list(ylim),
        n_subject_level_points=int(len(long_df)), n_negative=int((long_df.r < 0).sum()),
        tasks=TASKS, subjects=SUBJECTS, family_colors=FAMILY_COLOR,
        across_face=ACROSS_FACE, fonts=RC,
    )
    (out / "figure_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print("validation PASS")
    print("wrote", stem)
    print(f"ylim={ylim}  n_neg={(long_df.r < 0).sum()}  family means checked against summary_family_level.csv")


if __name__ == "__main__":
    main()
