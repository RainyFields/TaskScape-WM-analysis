#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/sub-01/ses-001/func")
MASK_PATH = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/valid_vertex_mask.npy")
OUT_FIG = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/glm_sanity/sub-01/ses-001/beta_consistency.png")


def decode_names(raw_names):
    out = []
    for n in raw_names:
        if isinstance(n, bytes):
            out.append(n.decode("utf-8"))
        else:
            out.append(str(n))
    return out


def per_vertex_corr(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    corrs = np.full((B1.shape[0],), np.nan, dtype=np.float32)
    for v in range(B1.shape[0]):
        x = B1[v]
        y = B2[v]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        corrs[v] = pearsonr(x, y)[0]
    return corrs


def trial_average_betas(betas: np.ndarray, names: list[str]) -> np.ndarray:
    trial_nums = [n.split("_")[0] for n in names]  # trial001
    unique_trials = list(dict.fromkeys(trial_nums))
    avg_betas = []
    for t in unique_trials:
        idx = [i for i, n in enumerate(names) if n.startswith(t)]
        avg_betas.append(betas[:, idx].mean(axis=1))
    return np.stack(avg_betas, axis=1)


def condition_average_betas(betas: np.ndarray, names: list[str]) -> np.ndarray:
    enc_idx = [i for i, n in enumerate(names) if n.endswith("encoding")]
    del_idx = [i for i, n in enumerate(names) if n.endswith("delay")]
    return np.stack(
        [
            betas[:, enc_idx].mean(axis=1),
            betas[:, del_idx].mean(axis=1),
        ],
        axis=1,
    )


def print_stats(prefix: str, corrs: np.ndarray, thr_a: float, thr_b: float | None = None):
    print(prefix)
    print(f"  Mean r = {np.nanmean(corrs):.3f}")
    print(f"  Median r = {np.nanmedian(corrs):.3f}")
    print(f"  % vertices r > {thr_a:.1f}: {np.nanmean(corrs > thr_a) * 100:.1f}%")
    if thr_b is not None:
        print(f"  % vertices r > {thr_b:.1f}: {np.nanmean(corrs > thr_b) * 100:.1f}%")


def main():
    files = sorted(BASE.glob("sub-01_ses-001_task-interdms_run-*_betas.h5"))
    runs = []
    betas_per_run = {}
    names_per_run = {}

    for h5_path in files:
        run = h5_path.stem.replace("sub-01_ses-001_task-interdms_", "").replace("_betas", "")
        with h5py.File(h5_path, "r") as f:
            betas_per_run[run] = f["betas"][:]
            names_per_run[run] = decode_names(f.attrs["task_regressor_names"])
        runs.append(run)

    runs = sorted(runs)
    print(f"Runs found: {runs}")
    if len(runs) < 2:
        raise RuntimeError("Need at least 2 runs for consistency.")

    mask = np.load(MASK_PATH).astype(bool)
    for run in runs:
        betas_per_run[run] = betas_per_run[run][mask, :]
        print(f"{run}: betas shape after mask = {betas_per_run[run].shape}")

    pairs = list(combinations(runs, 2))
    level_to_corrs = {"item-level": [], "trial-averaged": [], "condition-level": []}
    pair_labels = []

    # Level 1
    for r1, r2 in pairs:
        B1 = betas_per_run[r1]
        B2 = betas_per_run[r2]
        K_shared = min(B1.shape[1], B2.shape[1])
        B1s = B1[:, :K_shared]
        B2s = B2[:, :K_shared]
        corrs = per_vertex_corr(B1s, B2s)
        print()
        print_stats(f"[Item-level] {r1} vs {r2}:", corrs, thr_a=0.5, thr_b=0.3)
        level_to_corrs["item-level"].append(corrs)
        pair_labels.append(f"{r1} vs {r2}")

    # Level 2
    for r1, r2 in pairs:
        B1 = trial_average_betas(betas_per_run[r1], names_per_run[r1])
        B2 = trial_average_betas(betas_per_run[r2], names_per_run[r2])
        K_shared = min(B1.shape[1], B2.shape[1])
        corrs = per_vertex_corr(B1[:, :K_shared], B2[:, :K_shared])
        print()
        print_stats(f"[Trial-averaged] {r1} vs {r2}:", corrs, thr_a=0.5)
        level_to_corrs["trial-averaged"].append(corrs)

    # Level 3
    for r1, r2 in pairs:
        B1 = condition_average_betas(betas_per_run[r1], names_per_run[r1])
        B2 = condition_average_betas(betas_per_run[r2], names_per_run[r2])
        corrs = per_vertex_corr(B1, B2)
        print()
        print_stats(f"[Condition-level] {r1} vs {r2}:", corrs, thr_a=0.7)
        level_to_corrs["condition-level"].append(corrs)

    # Figure: 3 rows x 3 columns
    levels = ["item-level", "trial-averaged", "condition-level"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    for row, level in enumerate(levels):
        for col in range(3):
            ax = axes[row, col]
            if col >= len(level_to_corrs[level]):
                ax.axis("off")
                continue
            corrs = level_to_corrs[level][col]
            med = float(np.nanmedian(corrs))
            ax.hist(corrs[np.isfinite(corrs)], bins=50, range=(-1, 1), alpha=0.7)
            ax.axvline(med, linestyle="--", linewidth=1.5)
            ax.set_xlim(-1, 1)
            ax.set_title(f"{level} {pair_labels[col]}, median r={med:.3f}")
            ax.set_xlabel("Correlation (r)")
            ax.set_ylabel("Vertex count")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)
    print()
    print(f"Saved figure: {OUT_FIG}")


if __name__ == "__main__":
    main()
