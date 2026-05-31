#!/usr/bin/env python3
"""Verify single-file CIFTI smoothing quality."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.cifti2.cifti2_axes import SeriesAxis

GLASSER_LABEL_PATH = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii"
)


def _load_time_by_vertex(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    axis0 = img.header.get_axis(0)
    axis1 = img.header.get_axis(1)

    if isinstance(axis0, SeriesAxis):
        arr = data
    elif isinstance(axis1, SeriesAxis):
        arr = data.T
    else:
        arr = data if data.shape[0] <= data.shape[1] else data.T

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CIFTI data, got {arr.shape} for {path}")
    return arr


def _load_valid_vertex_mask(label_path: Path) -> np.ndarray:
    img = nib.load(str(label_path))
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    labels = np.rint(data).astype(np.int64).reshape(-1)
    return labels > 0


def _metrics(time_by_vertex: np.ndarray) -> dict[str, np.ndarray | float]:
    mean_signal = np.mean(time_by_vertex, axis=0)
    stdev = np.std(time_by_vertex, axis=0, ddof=1)
    tsnr = mean_signal / stdev
    global_mean = np.mean(time_by_vertex, axis=1)
    return {
        "stdev": stdev,
        "tsnr": tsnr,
        "mean_stdev": float(np.mean(stdev)),
        "max_stdev": float(np.max(stdev)),
        "mean_tsnr": float(np.mean(tsnr)),
        "global_mean": global_mean,
    }


def _hist(a: np.ndarray, b: np.ndarray, out: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(a, bins=100, alpha=0.6, label="Original", density=True)
    plt.hist(b, bins=100, alpha=0.6, label="Smoothed", density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _line(a: np.ndarray, b: np.ndarray, out: Path) -> None:
    n_tp = min(100, a.shape[0], b.shape[0])
    x = np.arange(n_tp)
    plt.figure(figsize=(10, 5))
    plt.plot(x, a[:n_tp], label="Original", linewidth=2)
    plt.plot(x, b[:n_tp], label="Smoothed", linewidth=2)
    plt.title("Mean BOLD signal across valid vertices (first 100 timepoints)")
    plt.xlabel("Timepoint")
    plt.ylabel("Mean BOLD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--smoothed", required=True, type=Path)
    parser.add_argument("--figdir", required=True, type=Path)
    args = parser.parse_args()

    if args.original.resolve() == args.smoothed.resolve():
        print("ERROR: input path == output path. Aborting.")
        return 1

    args.figdir.mkdir(parents=True, exist_ok=True)

    orig = _load_time_by_vertex(args.original)
    sm = _load_time_by_vertex(args.smoothed)

    if orig.shape != sm.shape:
        print(f"ERROR: shape mismatch: original={orig.shape}, smoothed={sm.shape}")
        return 1

    if not GLASSER_LABEL_PATH.exists():
        print(f"ERROR: Missing Glasser label file: {GLASSER_LABEL_PATH}")
        return 1

    parcel_mask = _load_valid_vertex_mask(GLASSER_LABEL_PATH)
    if parcel_mask.size != orig.shape[1]:
        print(
            "ERROR: Vertex count mismatch between dtseries and Glasser labels: "
            f"dtseries vertices={orig.shape[1]}, labels={parcel_mask.size}"
        )
        return 1

    stdev_all = np.std(orig, axis=0, ddof=1)
    variance_mask = stdev_all > 1e-6
    valid_mask = parcel_mask & variance_mask

    n_total = int(valid_mask.size)
    n_parcel_excluded = int(np.sum(~parcel_mask))
    n_zero_var_in_parcels = int(np.sum(parcel_mask & ~variance_mask))
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        print("ERROR: No valid vertices found after applying two-condition mask.")
        return 1

    orig_masked = orig[:, valid_mask]
    sm_masked = sm[:, valid_mask]

    mo = _metrics(orig_masked)
    ms = _metrics(sm_masked)

    print("\n=== Smoothing Verification Summary ===")
    print(f"Medial wall vertices excluded: {n_parcel_excluded}")
    print(f"Zero-variance vertices excluded (within parcels): {n_zero_var_in_parcels}")
    print(f"Final valid vertices: {n_valid} (of total {n_total})")
    print(f"{'Metric':<30}{'Original':>16}{'Smoothed':>16}")
    print("-" * 62)
    print(f"{'Mean per-vertex STDEV':<30}{mo['mean_stdev']:>16.6f}{ms['mean_stdev']:>16.6f}")
    print(f"{'Max per-vertex STDEV':<30}{mo['max_stdev']:>16.6f}{ms['max_stdev']:>16.6f}")
    print(f"{'Mean per-vertex tSNR':<30}{mo['mean_tsnr']:>16.6f}{ms['mean_tsnr']:>16.6f}")

    stdev_fig = args.figdir / "hist_per_vertex_stdev_original_vs_smoothed.png"
    tsnr_fig = args.figdir / "hist_per_vertex_tsnr_original_vs_smoothed.png"
    line_fig = args.figdir / "line_mean_bold_first100_original_vs_smoothed.png"

    _hist(mo["stdev"], ms["stdev"], stdev_fig, "Per-vertex STDEV", "STDEV")
    _hist(mo["tsnr"], ms["tsnr"], tsnr_fig, "Per-vertex tSNR", "tSNR")
    _line(mo["global_mean"], ms["global_mean"], line_fig)

    print("\nSaved figures:")
    print(stdev_fig)
    print(tsnr_fig)
    print(line_fig)

    stdev_ok = ms["mean_stdev"] < mo["mean_stdev"]
    tsnr_ok = ms["mean_tsnr"] > mo["mean_tsnr"]

    if not stdev_ok:
        print("\n!!! WARNING: Smoothed mean STDEV is NOT lower than original !!!")
        return 2
    if not tsnr_ok:
        print("\n!!! WARNING: Smoothed mean tSNR is NOT higher than original !!!")
        return 2

    print("\nVerification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
