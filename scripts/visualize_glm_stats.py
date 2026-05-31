#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def decode_list(values):
    out = []
    for v in list(values):
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def finite_valid(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def hist_with_lines(ax, values, bins, color, xlabel, title, xlim=None, fwhm_hist=0.3):
    v = finite_valid(values)
    if v.size == 0:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return np.nan, np.nan, np.nan

    counts, edges, _ = ax.hist(v, bins=bins, color=color, alpha=0.75, edgecolor="none")
    centers = 0.5 * (edges[1:] + edges[:-1])
    if fwhm_hist > 0:
        sigma = fwhm_hist / 2.354820045
        smooth_counts = gaussian_filter1d(counts.astype(np.float64), sigma=max(sigma, 1e-3))
        ax.plot(centers, smooth_counts, color="black", linewidth=1.0, alpha=0.7)

    mean = float(np.nanmean(v))
    med = float(np.nanmedian(v))
    std = float(np.nanstd(v))
    ax.axvline(mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean:.3f}")
    ax.axvline(med, color="gray", linestyle=":", linewidth=1.0, label=f"median={med:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("grayordinate count")
    ax.set_title(f"{title}\nmean={mean:.3f}  median={med:.3f}  std={std:.3f}")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(fontsize=8)
    return mean, med, std


def make_parser():
    p = argparse.ArgumentParser(description="Visualize GLM R2 and beta statistics from *_betas.h5.")
    p.add_argument("--h5", required=True, help="Path to _betas.h5 file")
    p.add_argument("--out_dir", default=None, help="Output directory (default: same directory as h5)")
    p.add_argument("--subject", default="sub-XX", help="Subject ID for titles")
    p.add_argument("--task", default="task", help="Task label for titles")
    p.add_argument("--run", default="run", help="Run label for titles")
    p.add_argument("--fwhm_hist", type=float, default=0.3, help="Smoothing FWHM for histogram KDE line")
    p.add_argument("--parcel_labels", default=None, help="Optional Glasser parcel labels path")
    return p


def main():
    args = make_parser().parse_args()
    h5_path = Path(args.h5)
    out_dir = Path(args.out_dir) if args.out_dir else h5_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        betas = f["betas"][:]  # (P x n_trials)
        col_names = decode_list(f["design_col_names"][:])
        trial_names = decode_list(f.attrs["task_regressor_names"])
        valid = f["valid_mask"][:].astype(bool)
        r2_full = f["r2_full"][:].astype(np.float32)
        r2_noise = f["r2_noise"][:].astype(np.float32)
        r2_task = f["r2_task"][:].astype(np.float32)
        _psc_mu = f["psc_baseline_mean"][:]

    # Separate encoding and delay betas
    enc_idx = [i for i, n in enumerate(trial_names) if "encoding" in n]
    del_idx = [i for i, n in enumerate(trial_names) if "delay" in n]
    betas_enc = betas[:, enc_idx]
    betas_del = betas[:, del_idx]
    mean_enc = np.nanmean(betas_enc, axis=1).astype(np.float32)
    mean_del = np.nanmean(betas_del, axis=1).astype(np.float32)

    # Mask invalids
    for arr in [r2_full, r2_noise, r2_task, mean_enc, mean_del]:
        arr[~valid] = np.nan

    n_valid = int(np.sum(valid))

    # Figure 1 — R² distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    m_full, _, _ = hist_with_lines(
        axes[0], r2_full[valid], bins=100, color="#4878CF",
        xlabel="R² value", title="R²_full", xlim=(-0.1, 1.0), fwhm_hist=args.fwhm_hist
    )
    m_noise, _, _ = hist_with_lines(
        axes[1], r2_noise[valid], bins=100, color="#6ACC65",
        xlabel="R² value", title="R²_noise", xlim=(-0.1, 1.0), fwhm_hist=args.fwhm_hist
    )
    m_task, _, _ = hist_with_lines(
        axes[2], r2_task[valid], bins=100, color="#D65F5F",
        xlabel="R² value", title="R²_task", xlim=(-0.1, 1.0), fwhm_hist=args.fwhm_hist
    )
    fig.suptitle(
        f"{args.subject} | {args.task} | {args.run} — R² distributions (n_valid={n_valid})",
        fontsize=12,
    )
    fig.tight_layout()
    fig1 = out_dir / "r2_distributions.png"
    fig.savefig(fig1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2 — R²_task detail
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rt = finite_valid(r2_task[valid])
    if rt.size:
        axes[0].axvspan(-1.0, 0.01, color="gray", alpha=0.15, label="noise floor")
        axes[0].axvspan(0.01, 0.05, color="gold", alpha=0.15, label="low")
        axes[0].axvspan(0.05, 0.15, color="limegreen", alpha=0.12, label="moderate")
        axes[0].axvspan(0.15, max(0.3, float(rt.max()) + 0.05), color="deepskyblue", alpha=0.10, label="high")
        axes[0].hist(rt, bins=100, color="#D65F5F", alpha=0.80, edgecolor="none")
        axes[0].axvline(np.nanmean(rt), color="black", linestyle="--", linewidth=1.2, label=f"mean={np.nanmean(rt):.3f}")
        axes[0].axvline(np.nanmedian(rt), color="gray", linestyle=":", linewidth=1.0, label=f"median={np.nanmedian(rt):.3f}")
        axes[0].set_xlim(-0.05, max(0.3, float(rt.max()) + 0.05))
        axes[0].legend(fontsize=8)
        sorted_r2 = np.sort(rt)
        cdf = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2)
        axes[1].plot(sorted_r2, cdf, color="#D65F5F", linewidth=1.5)
        axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=1)
        axes[1].axhline(0.9, color="gray", linestyle=":", linewidth=1)
    else:
        axes[0].text(0.5, 0.5, "No finite R²_task", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No finite R²_task", ha="center", va="center", transform=axes[1].transAxes)
    axes[0].set_xlabel("R²_task")
    axes[0].set_ylabel("grayordinate count")
    axes[0].set_title("R²_task histogram")
    axes[1].set_xlabel("R²_task")
    axes[1].set_ylabel("cumulative fraction of grayordinates")
    axes[1].set_title("CDF of R²_task")
    fig.suptitle(f"R²_task | {args.subject} {args.task} {args.run}", fontsize=13)
    fig.tight_layout()
    fig2 = out_dir / "r2_task_detail.png"
    fig.savefig(fig2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3 — Encoding beta
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    me = mean_enc[valid]
    md = mean_del[valid]
    mean_enc_val, _, _ = hist_with_lines(
        axes[0], me, bins=100, color="#E24B4A",
        xlabel="mean encoding β (% BOLD)",
        title="Mean encoding beta per grayordinate",
        xlim=(-5, 10) if np.nanmax(np.abs(me)) <= 10 else None,
        fwhm_hist=args.fwhm_hist,
    )
    axes[0].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].scatter(me, md, alpha=0.05, s=1, color="gray")
    sc = axes[1].scatter(
        me, md, c=r2_task[valid], cmap="viridis", alpha=0.3, s=1,
        vmin=0, vmax=np.nanpercentile(r2_task[valid], 95),
    )
    plt.colorbar(sc, ax=axes[1], label="R²_task")
    axes[1].set_xlabel("mean encoding β (% BOLD)")
    axes[1].set_ylabel("mean delay β (% BOLD)")
    axes[1].set_title("Encoding vs Delay beta per grayordinate")
    fig.tight_layout()
    fig3 = out_dir / "beta_encoding.png"
    fig.savefig(fig3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 4 — Delay beta
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mean_del_val, _, _ = hist_with_lines(
        axes[0], md, bins=100, color="#2E75B6",
        xlabel="mean delay β (% BOLD)",
        title="Mean delay beta per grayordinate",
        xlim=(-5, 10) if np.nanmax(np.abs(md)) <= 10 else None,
        fwhm_hist=args.fwhm_hist,
    )
    axes[0].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].scatter(me, md, alpha=0.05, s=1, color="gray")
    sc = axes[1].scatter(
        me, md, c=r2_task[valid], cmap="viridis", alpha=0.3, s=1,
        vmin=0, vmax=np.nanpercentile(r2_task[valid], 95),
    )
    plt.colorbar(sc, ax=axes[1], label="R²_task")
    axes[1].set_xlabel("mean encoding β (% BOLD)")
    axes[1].set_ylabel("mean delay β (% BOLD)")
    axes[1].set_title("Encoding vs Delay beta per grayordinate")
    fig.tight_layout()
    fig4 = out_dir / "beta_delay.png"
    fig.savefig(fig4, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Saved histogram/stat figures:\n  {fig1}\n  {fig2}\n  {fig3}\n  {fig4}")
    print(f"[stats] r2_full mean={np.nanmean(r2_full[valid]):.4f}")
    print(f"[stats] r2_noise mean={np.nanmean(r2_noise[valid]):.4f}")
    print(f"[stats] r2_task mean={np.nanmean(r2_task[valid]):.4f}")
    print(f"[stats] mean encoding beta (valid)={mean_enc_val:.4f}")
    print(f"[stats] mean delay beta (valid)={mean_del_val:.4f}")

    # Attempt brain surface maps (best-effort fallback)
    try:
        from nilearn import datasets, plotting, surface

        fsavg = datasets.fetch_surf_fsaverage("fsaverage5")
        metrics = [
            ("brain_r2task", r2_task, "hot", (0, 0.3)),
            ("brain_enc_beta", mean_enc, "RdBu_r", (-3, 3)),
            ("brain_del_beta", mean_del, "RdBu_r", (-3, 3)),
        ]
        # Dense grayordinates are stored as left then right cortex.
        # Split equally by hemisphere and resample by nearest index to fsaverage5.
        n_total = int(r2_task.shape[0])
        n_left = n_total // 2
        n_right = n_total - n_left
        mesh_map = {
            "left": (fsavg["infl_left"], fsavg["sulc_left"]),
            "right": (fsavg["infl_right"], fsavg["sulc_right"]),
        }
        target_n = {
            "left": surface.load_surf_mesh(fsavg["infl_left"]).coordinates.shape[0],
            "right": surface.load_surf_mesh(fsavg["infl_right"]).coordinates.shape[0],
        }
        for name, metric, cmap, vrange in metrics:
            hemi_data = {
                "left": metric[:n_left],
                "right": metric[n_left:n_left + n_right],
            }
            for hemi in ["left", "right"]:
                data = np.asarray(hemi_data[hemi], dtype=np.float32)
                if data.shape[0] != target_n[hemi]:
                    # nearest-neighbor index mapping fallback
                    src_idx = np.linspace(0, max(0, data.shape[0] - 1), target_n[hemi]).astype(int)
                    data = data[src_idx]
                out_png = out_dir / f"{name}_{hemi}.png"
                plotting.plot_surf_stat_map(
                    surf_mesh=mesh_map[hemi][0],
                    stat_map=data,
                    bg_map=mesh_map[hemi][1],
                    hemi=hemi,
                    colorbar=True,
                    cmap=cmap,
                    vmax=vrange[1],
                    threshold=0.001,
                    title=f"{name} {hemi}",
                    output_file=str(out_png),
                )
        print("[✓] Brain surface plots saved.")
    except Exception as e:
        print(f"[!] Brain surface plot failed: {e}")
        print("    Histogram figures were saved successfully.")
        print("    To enable brain plots, ensure nilearn >= 0.10 and fsaverage5 data can be downloaded.")


if __name__ == "__main__":
    main()
