#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster condition-by-condition heatmap and summarize clusters.

Inputs (from your previous step):
  - *_cond_similarity_mean_matrix_per_<unit>.npy   (R_mean: C x C)
  - *_cond_names_per_<unit>.json                   (["condA", "condB", ...])

Outputs:
  - <subject>_clustered_heatmap_per_<unit>.png
  - <subject>_clusters_list_per_<unit>.png
  - <subject>_cluster_assignments_per_<unit>.csv
  - <subject>_cluster_traits_per_<unit>.json
  - <subject>_cluster_traits_per_<unit>.txt
  - <subject>_reordered_matrix_per_<unit>.npy
  - <subject>_reordered_cond_names_per_<unit>.json

Usage:
    python hierarchical_clustering_conditions.py \
    --subject sub-03 \
    --matrix_npy /project/def-pbellec/xuan/fmri_dataset_project/results/condition_similarity/sub-06_cond_similarity_mean_matrix_per_network.npy \
    --cond_json  /project/def-pbellec/xuan/fmri_dataset_project/results/condition_similarity/sub-06_cond_names_per_network.json \
    --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/clustered \
    --unit network \
    --method average \
    --n_clusters 40 \
    --metadata_csv /optional/condition_metadata.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


# ----------------------- utilities -----------------------

def to_jsonable(x):
    """Recursively convert numpy types and keys to JSON-safe types."""
    if isinstance(x, dict):
        # stringify keys for portability
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def sanitize_corr(R):
    """Clamp to [-1,1], fill bad entries, symmetrize."""
    R = np.array(R, dtype=np.float64)
    C = R.shape[0]
    R = np.clip(R, -1.0, 1.0, out=R)

    # Diagonal: set NaN diag to 1
    diag = np.diag(R).copy()
    diag[~np.isfinite(diag)] = 1.0
    np.fill_diagonal(R, diag)

    # Off-diagonal NaNs -> row-wise means, fallback to 0
    mask = np.isfinite(R)
    if not np.all(mask):
        row_means = np.where(
            np.any(mask, axis=1, keepdims=True),
            np.nanmean(np.where(mask, R, np.nan), axis=1, keepdims=True),
            0.0
        )
        R = np.where(mask, R, row_means)
        R = np.nan_to_num(R, nan=0.0)

    # Force symmetry
    R = 0.5 * (R + R.T)
    # Final clamp
    R = np.clip(R, -1.0, 1.0, out=R)
    return R


def correlation_to_distance(R):
    """Distance from correlation: d = 1 - r  in [0,2]."""
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return D


def hierarchical_order(D, method="average", optimal_ordering=True):
    """
    D: square distance matrix
    Returns:
        order: list of leaf indices
        Z: linkage matrix
    """
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method, optimal_ordering=optimal_ordering)
    dend = dendrogram(Z, no_plot=True)
    order = dend["leaves"]
    return order, Z


def plot_clustered_heatmap(R, cond_names, Z, order, out_png, title="", vmin=-1, vmax=1):
    """
    Plot only a clustered heatmap (no dendrograms, no text labels).
    """
    R_ord = R[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(
        R_ord, vmin=vmin, vmax=vmax,
        origin="upper", interpolation="nearest"
    )

    # Remove tick labels (clean look)
    ax.set_xticks([])
    ax.set_yticks([])

    # Optional title and colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation", rotation=90)
    if title:
        ax.set_title(title, fontsize=12, pad=12)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return R_ord, [cond_names[i] for i in order]


def tokenize(name):
    """Simple tokenization: split on non-alphanumeric, keep tokens >=2 chars, lowercase."""
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", name) if len(t) >= 2]


def simple_token_enrichment(cond_names, cluster_ids):
    """
    For each cluster, find tokens over-represented vs global.
    Returns dict: cluster_id -> {"size": n, "top_tokens": [(token, lift, p_cluster, p_global), ...]}
    """
    all_tokens = []
    per_cond_tokens = []
    for name in cond_names:
        toks = tokenize(name)
        per_cond_tokens.append(toks)
        # unique per condition to avoid duplicated words dominating
        all_tokens.extend(set(toks))

    global_counts = Counter(all_tokens)
    N = len(cond_names)
    global_freq = {t: c / N for t, c in global_counts.items()}

    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        clusters[int(cid)].append(i)

    out = {}
    for cid, idxs in clusters.items():
        c_tokens = []
        for i in idxs:
            c_tokens.extend(set(per_cond_tokens[i]))
        c_counts = Counter(c_tokens)
        cN = len(idxs)
        c_freq = {t: c / cN for t, c in c_counts.items()} if cN > 0 else {}

        scored = []
        for t, pf in c_freq.items():
            pg = global_freq.get(t, 0.0)
            lift = (pf - pg) / (pg + 1e-8)
            scored.append((t, float(lift), float(pf), float(pg)))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[int(cid)] = {"size": cN, "top_tokens": scored[:15]}
    return out


def metadata_enrichment(cond_names, cluster_ids, meta_csv_path):
    """
    If a metadata CSV is supplied (with 'condition' + categorical columns),
    compute simple enrichment per cluster for each categorical field.

    Returns: dict[cid][field] -> [(value, lift, p_cluster, p_global)]
    """
    import pandas as pd
    meta = pd.read_csv(meta_csv_path)
    if "condition" not in meta.columns:
        raise ValueError("Metadata CSV must have a 'condition' column.")
    meta = meta.set_index("condition")
    meta = meta.reindex(cond_names)

    # Identify categorical columns; also coerce small-cardinality numerics to categorical
    cat_cols = [c for c in meta.columns if meta[c].dtype == "object" or str(meta[c].dtype).startswith("category")]
    for c in meta.columns:
        if c not in cat_cols and meta[c].nunique(dropna=True) <= 10:
            meta[c] = meta[c].astype("category")
            cat_cols.append(c)

    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        clusters[int(cid)].append(i)

    N = len(cond_names)
    out = {}
    for cid, idxs in clusters.items():
        out[int(cid)] = {"size": len(idxs)}
        for col in cat_cols:
            series = meta[col]
            overall_counts = series.value_counts(dropna=False)
            overall_freq = (overall_counts / N).to_dict()

            sub_counts = series.iloc[idxs].value_counts(dropna=False)
            cN = len(idxs)
            sub_freq = (sub_counts / cN).to_dict() if cN > 0 else {}

            scored = []
            for val, pf in sub_freq.items():
                pg = float(overall_freq.get(val, 0.0))
                lift = (float(pf) - pg) / (pg + 1e-8)
                scored.append((str(val), float(lift), float(pf), float(pg)))
            scored.sort(key=lambda x: x[1], reverse=True)
            out[int(cid)][col] = scored[:15]
    return out


def render_cluster_list_png(cond_names_ord, cluster_ids_ord, out_png, title="Condition clusters"):
    """Render a readable PNG listing clusters and their condition names."""
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(cond_names_ord) + 2))
    ax.axis("off")

    lines = [title, ""]
    prev_cid = None
    for name, cid in zip(cond_names_ord, cluster_ids_ord):
        if cid != prev_cid:
            if prev_cid is not None:
                lines.append("")
            lines.append(f"Cluster {cid}:")
            prev_cid = cid
        lines.append(f"  - {name}")

    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            fontsize=10, family="monospace")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--matrix_npy", required=True, help="Path to *_mean_matrix_*.npy")
    ap.add_argument("--cond_json", required=True, help="Path to *_cond_names_*.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--unit", default="voxel", help="voxel|region|network (for file naming only)")
    ap.add_argument("--method", default="average",
                    choices=["single","complete","average","weighted","ward","median","centroid"])
    ap.add_argument("--n_clusters", type=int, default=None,
                    help="If set, cut dendrogram into this many clusters.")
    ap.add_argument("--max_d", type=float, default=None,
                    help="Alternative: cut at distance threshold.")
    ap.add_argument("--metadata_csv", type=str, default=None,
                    help="Optional: CSV with column 'condition' and categorical traits.")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    R = np.load(args.matrix_npy)
    cond_names = json.loads(Path(args.cond_json).read_text())
    if not (R.shape[0] == R.shape[1] == len(cond_names)):
        raise SystemExit("Matrix size and condition list length must match.")

    # Sanitize correlation & compute distance
    R = sanitize_corr(R)
    D = correlation_to_distance(R)

    # Clustering & order
    order, Z = hierarchical_order(D, method=args.method, optimal_ordering=True)

    # Clustered heatmap
    title = f"{args.subject} clustered condition similarity (per-{args.unit})"
    heatmap_png = outdir / f"{args.subject}_clustered_heatmap_per_{args.unit}.png"
    R_ord, cond_ord = plot_clustered_heatmap(
        R, cond_names, Z, order, str(heatmap_png), title=title, vmin=-1, vmax=1
    )

    # Choose cluster cut
    if args.n_clusters is not None:
        cluster_ids = fcluster(Z, args.n_clusters, criterion="maxclust")
    elif args.max_d is not None:
        cluster_ids = fcluster(Z, args.max_d, criterion="distance")
    else:
        # Simple heuristic: elbow on the last 10% of merge heights
        last = Z[-max(5, len(Z)//10):, 2]
        if len(last) >= 2:
            diffs = np.diff(last)
            k = int(np.argmax(diffs) + 1)
            n_merge_total = Z.shape[0]
            n_clusters = (n_merge_total + 1) - (len(last) - k)
            n_clusters = max(1, int(n_clusters))
            cluster_ids = fcluster(Z, n_clusters, criterion="maxclust")
        else:
            cluster_ids = fcluster(Z, 1, criterion="maxclust")

    # Reorder cluster ids to match heatmap order; cast to native int
    cluster_ids_ord = [int(cluster_ids[i]) for i in order]

    # Save reordered matrix and names
    np.save(outdir / f"{args.subject}_reordered_matrix_per_{args.unit}.npy", R_ord)
    Path(outdir / f"{args.subject}_reordered_cond_names_per_{args.unit}.json").write_text(
        json.dumps(cond_ord, indent=2)
    )

    # Save cluster assignments (CSV)
    csv_path = outdir / f"{args.subject}_cluster_assignments_per_{args.unit}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "cluster_id", "order_index"])
        for rank, (name, cid) in enumerate(zip(cond_ord, cluster_ids_ord)):
            w.writerow([name, cid, rank])

    # PNG list of clusters and names
    clusters_list_png = outdir / f"{args.subject}_clusters_list_per_{args.unit}.png"
    render_cluster_list_png(
        cond_ord, cluster_ids_ord, str(clusters_list_png),
        title=f"{args.subject} condition clusters (per-{args.unit})"
    )

    # Trait analysis
    traits = {}
    traits["token_enrichment"] = simple_token_enrichment(cond_ord, cluster_ids_ord)

    if args.metadata_csv:
        try:
            traits["metadata_enrichment"] = metadata_enrichment(cond_ord, cluster_ids_ord, args.metadata_csv)
        except Exception as e:
            traits["metadata_enrichment_error"] = str(e)

    traits_json = outdir / f"{args.subject}_cluster_traits_per_{args.unit}.json"
    Path(traits_json).write_text(json.dumps(to_jsonable(traits), indent=2))

    # Human-readable TXT summary
    txt_lines = []
    txt_lines.append(f"Subject: {args.subject} | unit: {args.unit}\n")
    txt_lines.append("Cluster token enrichment (top):\n")
    for cid in sorted(traits["token_enrichment"].keys(), key=lambda x: int(x)):
        info = traits["token_enrichment"][cid]
        size = info.get("size", "?")
        txt_lines.append(f"- Cluster {cid} (n={size}):")
        tops = info.get("top_tokens", [])
        if not tops:
            txt_lines.append("    (no salient tokens)")
        else:
            for t, lift, pf, pg in tops[:8]:
                txt_lines.append(f"    {t:>12s}  lift={lift:+.2f}  p_cluster={pf:.2f}  p_global={pg:.2f}")

    if "metadata_enrichment" in traits:
        txt_lines.append("\nMetadata enrichment (top per field):")
        for cid in sorted(traits["metadata_enrichment"].keys(), key=lambda x: int(x)):
            fields = traits["metadata_enrichment"][cid]
            size = fields.get("size", "?")
            txt_lines.append(f"- Cluster {cid} (n={size}):")
            for field, vals in fields.items():
                if field == "size" or not vals:
                    continue
                best = vals[0]
                txt_lines.append(
                    f"    {field}: {best[0]} (lift={best[1]:+.2f}, p_c={best[2]:.2f}, p_g={best[3]:.2f})"
                )

    traits_txt = outdir / f"{args.subject}_cluster_traits_per_{args.unit}.txt"
    Path(traits_txt).write_text("\n".join(txt_lines))

    print(f"✅ Clustered heatmap: {heatmap_png}")
    print(f"✅ Cluster list PNG: {clusters_list_png}")
    print(f"✅ Cluster assignments CSV: {csv_path}")
    print(f"✅ Traits JSON: {traits_json}")
    print(f"✅ Traits TXT:  {traits_txt}")


if __name__ == "__main__":
    main()
