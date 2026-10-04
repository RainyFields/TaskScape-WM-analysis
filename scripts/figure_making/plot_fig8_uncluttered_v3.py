#!/usr/bin/env python3
"""Uncluttered Figure 8 at manuscript includegraphics width.

The manuscript TeX is not in this repository. Nature Scientific Data full-width
figures use figure* + \\includegraphics[width=\\textwidth]. That text width is
183 mm (Nature double column; figure_style JOURNAL_SPECS['nature']['double']).

The PDF is generated at that physical width so Overleaf does not downscale type.
Requested on-page sizes are therefore the matplotlib point sizes (scale = 1).

Reads ONLY results/similarity_analysis/regenerated_clean/ (variant clean_verified).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, "/project/def-pbellec/xuan/figure_style")
from figure_utils import apply_clean_spines, mm_to_inches, save_figure

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
RESULTS = PROJECT / "results/similarity_analysis/regenerated_clean"
VARIANT = "clean_verified"

# Canvas is wider than Nature \textwidth (183 mm) so the 12-task axis can breathe.
# If later included at width=\textwidth, effective type = drawn × (183 / FIGURE_WIDTH_MM).
TEXTWIDTH_MM = 183.0
FIGURE_WIDTH_MM = 364.0  # 280 mm pre-permutation canvas × 1.3
N_ROWS = 3
PANEL_H_MM = 72.0
ROW_GAP_MM = 16.0
FIGURE_HEIGHT_MM = 3 * PANEL_H_MM + 2 * ROW_GAP_MM + 48.0  # panels + gaps + title/x-label margins

TASKS = [
    "dms_loc",
    "nback_loc", "nback_ctg", "nback_obj",
    "interdms_loc_ABAB", "interdms_loc_ABBA",
    "interdms_ctg_ABAB", "interdms_ctg_ABBA",
    "interdms_obj_ABAB", "interdms_obj_ABBA",
    "ctxdm_lco", "ctxdm_col",
]
PRETTY = {
    "dms_loc": "DMS",
    "nback_loc": "loc",
    "nback_ctg": "ctg",
    "nback_obj": "id",
    "interdms_loc_ABAB": "loc",
    "interdms_loc_ABBA": "loc",
    "interdms_ctg_ABAB": "ctg",
    "interdms_ctg_ABBA": "ctg",
    "interdms_obj_ABAB": "id",
    "interdms_obj_ABBA": "id",
    "ctxdm_lco": "LCI",
    "ctxdm_col": "CIL",
}
SEQUENCE = {
    "interdms_loc_ABAB": "ABAB",
    "interdms_loc_ABBA": "ABBA",
    "interdms_ctg_ABAB": "ABAB",
    "interdms_ctg_ABBA": "ABBA",
    "interdms_obj_ABAB": "ABAB",
    "interdms_obj_ABBA": "ABBA",
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
    ("vertex", "Vertex level"),
    ("region", "Region level"),
    ("network", "Network level"),
]

# On-page type at manuscript width (scale = 1).
FONT = {
    "title": 15.5,
    "ylabel": 14.5,
    "tick": 12.0,
    "legend_title": 13.0,
    "legend": 11.5,
    "panel": 16.5,
    "family_group": 12.0,
    "sequence": 11.5,
}

# Geometry in millimetres.
BOX_W_MM = 3.1
INTRA_TASK_MM = 1.6
INTER_TASK_MM = 5.8
INTER_FAMILY_MM = 14.0
INTERDMS_EXTRA_MM = 2.0

plt.style.use("/project/def-pbellec/xuan/figure_style/nature_style.mplstyle")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": FONT["title"],
    "axes.labelsize": FONT["ylabel"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],
    "legend.title_fontsize": FONT["legend_title"],
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


class MulticolorBox:
    def __init__(self, colors):
        self.colors = list(colors)


class HandlerMulticolorBox(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        colors = orig_handle.colors
        n = len(colors)
        return [
            Rectangle(
                (xdescent + i * width / n, ydescent), width / n, height,
                facecolor=c, edgecolor="black", linewidth=0.5, transform=trans,
            )
            for i, c in enumerate(colors)
        ]


def load_clean(results: Path):
    w = pd.read_csv(results / "summary_subject_task.csv")
    a = pd.read_csv(results / "summary_pair_task.csv")
    fam = pd.read_csv(results / "summary_family_level.csv")
    task = pd.read_csv(results / "summary_task_level.csv")
    for df in (w, a, fam, task):
        if "variant" in df.columns:
            df.drop(index=df.index[df.variant != VARIANT], inplace=True)
    if w.empty or a.empty:
        raise SystemExit(f"No {VARIANT} rows in the clean tables.")
    return w, a, fam, task


def subject_across(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for res, _ in RESOLUTIONS:
        for task in TASKS:
            sub = pairs[(pairs.spatial_resolution == res) & (pairs.task == task)]
            for s in SUBJECTS:
                vals = sub[(sub.subject_1 == s) | (sub.subject_2 == s)]["mean"]
                if len(vals) != 4:
                    raise AssertionError(f"{res} {task} {s}: expected 4 pairs, got {len(vals)}")
                rows.append(dict(
                    spatial_resolution=res, task=task, task_family=FAMILY[task],
                    subject=s, comparison="across", r=float(vals.mean()), n_pairs=4,
                ))
    return pd.DataFrame(rows)


def subject_within(within: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        dict(
            spatial_resolution=r.spatial_resolution, task=r.task, task_family=r.task_family,
            subject=r.subject, comparison="within", r=float(r.mean), n_pairs=np.nan,
        )
        for r in within.itertuples()
    ])


def task_x_positions_mm():
    step = 2.0 * BOX_W_MM + INTRA_TASK_MM + INTER_TASK_MM
    extra = INTER_FAMILY_MM - INTER_TASK_MM
    xs, x = [], BOX_W_MM + 0.5 * INTRA_TASK_MM
    last_fam = None
    for t in TASKS:
        fam = FAMILY[t]
        if last_fam is not None and fam != last_fam:
            x += extra
        elif last_fam == "InterDMS":
            x += INTERDMS_EXTRA_MM
        xs.append(x)
        x += step
        last_fam = fam
    return np.asarray(xs)


def panel_ylim(vals: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, 0.25)
    pad = 0.08 * span
    ymin = lo - pad
    if lo > -0.08:
        ymin = min(-0.10, ymin)
    ymax = min(1.12, hi + pad + 0.07)
    return ymin, ymax


def style_boxplot(bp, facecolor):
    box = bp["boxes"][0]
    box.set_facecolor(facecolor)
    box.set_edgecolor("black")
    box.set_linewidth(0.8)
    box.set_alpha(1.0)
    for key in ("whiskers", "caps", "medians"):
        for line in bp[key]:
            line.set_color("black")
            line.set_linewidth(0.8)


def draw_pair_bracket(ax, x1, x2, y, h, color="0.45"):
    """Light grouping cap over a within/across pair (no significance mark)."""
    ax.plot(
        [x1, x1, x2, x2], [y, y + h, y + h, y],
        color=color, lw=0.75, solid_capstyle="round", solid_joinstyle="round",
        clip_on=False, zorder=6,
    )


def draw_panel(ax, long_df, res, xs, ylim, show_xticks, show_ylabel):
    offset = BOX_W_MM + 0.5 * INTRA_TASK_MM
    jitter = np.linspace(-0.32, 0.32, len(SUBJECTS))
    sub = long_df[long_df.spatial_resolution == res]
    yspan = ylim[1] - ylim[0]
    bracket_h = 0.022 * yspan

    for i, task in enumerate(TASKS):
        color = FAMILY_COLOR[FAMILY[task]]
        w = sub[(sub.task == task) & (sub.comparison == "within")].set_index("subject").loc[SUBJECTS, "r"].to_numpy()
        a = sub[(sub.task == task) & (sub.comparison == "across")].set_index("subject").loc[SUBJECTS, "r"].to_numpy()
        bp_w = ax.boxplot(
            [w], positions=[xs[i] - offset], widths=BOX_W_MM, patch_artist=True,
            showfliers=False, manage_ticks=False, whis=(0, 100),
        )
        style_boxplot(bp_w, color)
        bp_a = ax.boxplot(
            [a], positions=[xs[i] + offset], widths=BOX_W_MM, patch_artist=True,
            showfliers=False, manage_ticks=False, whis=(0, 100),
        )
        style_boxplot(bp_a, ACROSS_FACE)
        for j, s in enumerate(SUBJECTS):
            kw = dict(
                marker=SUBJECT_MARKERS[s], s=16, facecolors="white",
                edgecolors="black", linewidths=0.6, zorder=5,
            )
            ax.scatter(xs[i] - offset + jitter[j], w[j], **kw)
            ax.scatter(xs[i] + offset + jitter[j], a[j], **kw)
        x1, x2 = xs[i] - offset, xs[i] + offset
        y0 = max(float(w.max()), float(a.max())) + 0.028 * yspan
        draw_pair_bracket(ax, x1, x2, y0, bracket_h)

    ax.axhline(0.0, color="0.35", lw=0.8, linestyle="--", zorder=0)
    ax.set_ylim(*ylim)
    ax.set_xlim(xs[0] - BOX_W_MM - INTRA_TASK_MM - 1.0, xs[-1] + BOX_W_MM + INTRA_TASK_MM + 1.0)
    if show_ylabel:
        ax.set_ylabel("Similarity (Pearson r)", fontsize=FONT["ylabel"])
    ax.set_xticks(xs)
    if show_xticks:
        ax.set_xticklabels([PRETTY[t] for t in TASKS], ha="center", fontsize=FONT["tick"])
        for i, task in enumerate(TASKS):
            seq = SEQUENCE.get(task)
            if seq:
                ax.text(
                    xs[i], -0.12, seq, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=FONT["sequence"], clip_on=False,
                )
        for fam_name in FAMILIES:
            idx = [i for i, t in enumerate(TASKS) if FAMILY[t] == fam_name]
            xc = 0.5 * (xs[idx[0]] + xs[idx[-1]])
            ax.text(
                xc, -0.28, fam_name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FONT["family_group"],
                fontweight="normal", clip_on=False,
            )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    apply_clean_spines(ax, linewidth=0.9)


def draw_legends(ax):
    ax.set_axis_off()
    fam_h = [
        Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", linewidth=0.6, label=f)
        for f in FAMILIES
    ]
    within_h = MulticolorBox([FAMILY_COLOR[f] for f in FAMILIES])
    across_h = Patch(facecolor=ACROSS_FACE, edgecolor="black", linewidth=0.6)
    sub_h = [
        Line2D(
            [], [], marker=SUBJECT_MARKERS[s], color="w", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=0.7, markersize=6.5,
            linestyle="None", label=s.replace("sub-", "Sub "),
        )
        for s in SUBJECTS
    ]
    kw = dict(
        frameon=False, borderaxespad=0.0, labelspacing=0.35, handletextpad=0.45,
        fontsize=FONT["legend"], title_fontsize=FONT["legend_title"],
    )
    leg1 = ax.legend(handles=fam_h, title="Task family", loc="upper left",
                     bbox_to_anchor=(0.0, 1.02), **kw)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=[within_h, across_h],
        labels=["Within-subject", "Across-subject"],
        title="Comparison", loc="upper left", bbox_to_anchor=(0.0, 0.58),
        handler_map={MulticolorBox: HandlerMulticolorBox()},
        handleheight=1.4,
        **kw,
    )
    leg2._legend_box.align = "left"
    ax.add_artist(leg2)
    ax.legend(handles=sub_h, title="Subject", loc="upper left",
              bbox_to_anchor=(0.0, 0.30), **kw)


def validate(long_df, fam, task):
    failures = []
    for res, _ in RESOLUTIONS:
        for fam_name in FAMILIES:
            row = fam[(fam.spatial_resolution == res) & (fam.task_family == fam_name)].iloc[0]
            sub = long_df[(long_df.spatial_resolution == res) & (long_df.task_family == fam_name)]
            w_task = sub[sub.comparison == "within"].groupby("task").r.mean().mean()
            a_task = sub[sub.comparison == "across"].groupby("task").r.mean().mean()
            if not np.isclose(w_task, row.within_mean, rtol=0, atol=1e-12):
                failures.append(f"{res} {fam_name} within mismatch")
            if not np.isclose(a_task, row.across_mean, rtol=0, atol=1e-12):
                failures.append(f"{res} {fam_name} across mismatch")
    if long_df.task.nunique() != 12 or set(long_df.subject) != set(SUBJECTS):
        failures.append("task/subject set")
    if (long_df.r < 0).sum() == 0:
        failures.append("no negative values")
    if failures:
        raise AssertionError("validation failed:\n  " + "\n  ".join(failures))


def flatten_png(path: Path):
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1])
    bg.save(path, dpi=(600, 600))


def pdf_size_mm(pdf: Path) -> tuple[float, float]:
    try:
        from pypdf import PdfReader
        box = PdfReader(str(pdf)).pages[0].mediabox
        return float(box.width) * 25.4 / 72.0, float(box.height) * 25.4 / 72.0
    except Exception:
        pass
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        for line in out.splitlines():
            if line.startswith("Page size:"):
                parts = line.split()
                return float(parts[2]) * 25.4 / 72.0, float(parts[4]) * 25.4 / 72.0
    except Exception:
        pass
    raw = pdf.read_bytes()
    i = raw.find(b"/MediaBox")
    if i < 0:
        raise RuntimeError(f"Could not read PDF page size: {pdf}")
    chunk = raw[i:i + 80].decode("latin1", errors="ignore")
    nums = [float(x) for x in chunk.replace("[", " ").replace("]", " ").split() if x.replace(".", "", 1).isdigit()]
    return nums[2] * 25.4 / 72.0, nums[3] * 25.4 / 72.0


def rasterize_pdf_at_width(pdf: Path, dest: Path, width_mm: float, dpi: int = 600) -> Path:
    width_px = int(round(width_mm / 25.4 * dpi))
    for cmd in (
        ["pdftocairo", "-png", "-singlefile", "-r", str(dpi), str(pdf), str(dest.with_suffix(""))],
        ["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf), str(dest.with_suffix(""))],
    ):
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            produced = dest.with_suffix(".png")
            if produced.exists() and produced != dest:
                produced.replace(dest)
            flatten_png(dest)
            return dest
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    try:
        import fitz
        doc = fitz.open(pdf)
        page = doc[0]
        zoom = (width_px / float(page.rect.width))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(dest))
        flatten_png(dest)
        return dest
    except Exception:
        pass
    from PIL import Image
    # Fallback: render matplotlib PNG is already 600 dpi at figsize width.
    src = pdf.with_suffix(".png")
    im = Image.open(src)
    dest.write_bytes(src.read_bytes())
    return dest


def write_docs(out: Path, ylim, n_neg, figsize, scale, pdf_w_mm, pdf_h_mm):
    caption = """# Proposed Figure 8 caption

**Figure 8 | Within- and across-subject spatial-pattern similarity of delay-period activation.** Pearson correlations of delay-period beta maps at (a) the vertex level (58,396 valid cortical vertices), (b) the region level (360 Glasser parcels) and (c) the network level (12 Cole–Anticevic networks). For each of the 12 task variants, each participant’s unique correct delay trials were randomly partitioned into two non-overlapping halves, averaged, z-scored across spatial units and correlated. This random split-half partition was repeated for 50 random split-half iterations; each plotted value is the mean of those 50 iterations for that participant (or participant pair). The 50 iterations are Monte Carlo partitions used to stabilise the estimate; they are not independent statistical observations.

Boxes summarise the five subject-level estimates (median, hinges and whiskers to the minimum and maximum). Open markers identify participants: circle, Sub 01; square, Sub 02; upward triangle, Sub 03; diamond, Sub 05; downward triangle, Sub 06. Within-subject boxes (left of each task) use the task-family colour (DMS, blue; 1-Back, yellow; InterDMS, purple; CTXDM, teal) and show the split-half correlation of each participant’s own two halves (the diagonal of the 5 × 5 matrix). Across-subject boxes (right of each task, light grey) show, for each participant, the mean of that participant’s four symmetrised pairwise correlations with the other participants, where each pair \\((i,j)\\) is \(0.5\\,[r(i_\\mathrm{A},j_\\mathrm{B})+r(j_\\mathrm{A},i_\\mathrm{B})]\\). Light brackets group each within- / across-subject pair and do not encode significance. Axis tick labels name the variant (loc, location; ctg, category; id, identity; ABAB / ABBA, InterDMS sequence order; LCI / CIL, CTXDM stimulus-order variants); family names under the axis mark the four groups. Negative Pearson correlations were retained. *Y*-axis ranges are panel-specific.

Source data: `figures/fig8_regenerated_clean/uncluttered_v3/fig8_within_across_similarity_uncluttered_v3_source_data.csv`.
"""
    (out / "figure8_caption_proposed.md").write_text(caption)

    eff = {k: v * scale for k, v in FONT.items()}
    smallest = min(eff.values())
    report = f"""# Manuscript-width type check

## Width used

The manuscript `.tex` is not in this repository. Nature Scientific Data places
full-width 3-panel figures in `figure*` with

```latex
\\includegraphics[width=\\textwidth]{{...}}
```

The PDF canvas is **{FIGURE_WIDTH_MM:.0f} × {FIGURE_HEIGHT_MM:.0f} mm** (wider than
Nature `\\textwidth` = {TEXTWIDTH_MM:.0f} mm) with {ROW_GAP_MM:.0f} mm between
stacked panels.

| Quantity | Value |
|---|---|
| PDF MediaBox | {pdf_w_mm:.2f} × {pdf_h_mm:.2f} mm |
| `figsize` | {figsize[0]:.4f} × {figsize[1]:.4f} in |
| Scale at native canvas size | 1.000 |
| Scale if included at `width=\\textwidth` ({TEXTWIDTH_MM:.0f} mm) | {TEXTWIDTH_MM / pdf_w_mm:.3f} |

## Effective type after scaling (pt)

| Element | Drawn | Effective |
|---|---:|---:|
| Panel titles | {FONT['title']:.1f} | {eff['title']:.2f} |
| Shared y-axis label | {FONT['ylabel']:.1f} | {eff['ylabel']:.2f} |
| y / x tick labels | {FONT['tick']:.1f} | {eff['tick']:.2f} |
| InterDMS sequence row | {FONT['sequence']:.1f} | {eff['sequence']:.2f} |
| Family group labels | {FONT['family_group']:.1f} | {eff['family_group']:.2f} |
| Legend titles | {FONT['legend_title']:.1f} | {eff['legend_title']:.2f} |
| Legend entries | {FONT['legend']:.1f} | {eff['legend']:.2f} |
| Panel letters | {FONT['panel']:.1f} | {eff['panel']:.2f} |

**Smallest effective font size after scaling: {smallest:.2f} pt** (legend entries / sequence row).

## Caption / Methods / Results wording

- Caption, Methods, and Results all say **50 random split-half iterations**, not bootstraps.
- Within-subject: diagonal of the 5 × 5 matrix (each participant’s half A vs half B).
- Across-subject pair: \(0.5[r(i_A,j_B)+r(j_A,i_B)]\); plotted subject-level value is the mean of the four pairs that include that participant.
- Boxes summarise the five subject-level estimates. Negative *r* retained. No *p*-values.

{n_neg} negative subject-level points. Panel ylims: {ylim}.
"""
    (out / "FONT_SCALE_REPORT.md").write_text(report)
    return smallest, eff


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(PROJECT / "figures/fig8_regenerated_clean/uncluttered_v3"))
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    w_tab, pair_tab, fam, task = load_clean(results)
    long_df = pd.concat([subject_within(w_tab), subject_across(pair_tab)], ignore_index=True)
    validate(long_df, fam, task)

    xs = task_x_positions_mm()
    ylims = {
        res: panel_ylim(long_df.loc[long_df.spatial_resolution == res, "r"].to_numpy())
        for res, _ in RESOLUTIONS
    }
    figsize = (mm_to_inches(FIGURE_WIDTH_MM), mm_to_inches(FIGURE_HEIGHT_MM))
    panel_w_mm = FIGURE_WIDTH_MM
    panel_h_mm = PANEL_H_MM

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(
        N_ROWS, 2, width_ratios=[1.0, 0.18], height_ratios=[1, 1, 1],
        left=0.07, right=0.995, top=0.96, bottom=0.16,
        wspace=2.5 / panel_w_mm, hspace=ROW_GAP_MM / panel_h_mm,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(N_ROWS)]
    letters = ["a", "b", "c"]
    for i, ((res, title), ax) in enumerate(zip(RESOLUTIONS, axes)):
        draw_panel(ax, long_df, res, xs, ylims[res], show_xticks=(i == N_ROWS - 1), show_ylabel=(i == 1))
        ax.set_title(title, pad=4, fontsize=FONT["title"])
        ax.text(
            -0.10, 1.04, letters[i],
            transform=ax.transAxes,
            fontsize=FONT["panel"],
            fontweight="bold",
            va="bottom",
            ha="left",
            clip_on=False,
        )
    draw_legends(fig.add_subplot(gs[:, 1]))

    stem = out / "fig8_within_across_similarity_uncluttered_v3"
    save_figure(fig, stem, formats=("svg", "pdf", "png"))
    flatten_png(stem.with_suffix(".png"))

    pdf = stem.with_suffix(".pdf")
    pdf_w_mm, pdf_h_mm = pdf_size_mm(pdf)
    scale = 1.0
    scale_if_textwidth = TEXTWIDTH_MM / pdf_w_mm
    preview = out / "fig8_within_across_similarity_uncluttered_v3_at_manuscript_width.png"
    rasterize_pdf_at_width(pdf, preview, pdf_w_mm, dpi=600)

    src = long_df.copy()
    src["task_label"] = src.task.map(FULL_LABEL)
    src["axis_label"] = src.task.map(PRETTY)
    src["variant"] = VARIANT
    src["aggregation"] = src.comparison.map({
        "within": "mean of 50 random split-half diagonals for this subject",
        "across": "mean of 4 pair estimates that include this subject (pairs already mean of 50 random split-half iterations)",
    })
    src.to_csv(out / "fig8_within_across_similarity_uncluttered_v3_source_data.csv", index=False)
    ylim_report = {k: [float(a), float(b)] for k, (a, b) in ylims.items()}
    smallest, eff = write_docs(out, ylim_report, int((long_df.r < 0).sum()), figsize, scale, pdf_w_mm, pdf_h_mm)
    (out / "figure_config.json").write_text(json.dumps(dict(
        variant=VARIANT,
        includegraphics="native canvas; optional width=\\textwidth would scale down",
        figure_width_mm=FIGURE_WIDTH_MM,
        textwidth_mm=TEXTWIDTH_MM,
        row_gap_mm=ROW_GAP_MM,
        pdf_mediabox_mm=[pdf_w_mm, pdf_h_mm],
        scale_on_page=scale,
        scale_if_textwidth=scale_if_textwidth,
        figsize_in=list(figsize),
        fonts_drawn_pt=FONT,
        fonts_effective_pt=eff,
        smallest_effective_pt=smallest,
        ylim=ylim_report,
        n_negative=int((long_df.r < 0).sum()),
        family_colors=FAMILY_COLOR,
    ), indent=2) + "\n")
    plt.close(fig)
    print("validation PASS")
    print(f"PDF {pdf_w_mm:.2f} x {pdf_h_mm:.2f} mm; scale={scale:.4f}")
    print(f"smallest effective font {smallest:.2f} pt")
    print("wrote", stem)


if __name__ == "__main__":
    main()
