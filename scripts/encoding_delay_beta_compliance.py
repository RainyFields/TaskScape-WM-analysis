#!/usr/bin/env python3
"""Shared compliance checks for encoding-delay trial-level GLM beta H5 files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

N_VERTICES = 64984

# Canonical pipeline: glm_analysis.py on bold_sm4 + preprocess_y_fixed_psc
EXPECTED_PSC_EQUATION = "100 * (Y - mean(Y_full)) / mean(Y_full)"
EXPECTED_SMOOTHING = "external_wb_command_cifti_smoothing_4mm_FWHM"
EXPECTED_REGRESSOR_LEVEL = "trial"


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def is_compliant_h5(h5_path: Path) -> tuple[bool, dict[str, Any]]:
    """Return (compliant, diagnostics) for a single *_betas.h5 file."""
    diag: dict[str, Any] = {"path": str(h5_path)}
    try:
        with h5py.File(h5_path, "r") as f:
            diag["psc_equation"] = _decode_attr(f.attrs.get("psc_equation"))
            diag["smoothing"] = _decode_attr(f.attrs.get("smoothing"))
            diag["estimator"] = _decode_attr(f.attrs.get("estimator"))
            diag["regressor_level"] = _decode_attr(f.attrs.get("regressor_level"))
            diag["has_psc_baseline_mean"] = "psc_baseline_mean" in f
            diag["has_betas"] = "betas" in f
            if "betas" in f:
                diag["betas_shape"] = tuple(f["betas"].shape)
            else:
                diag["betas_shape"] = None
    except OSError as exc:
        diag["error"] = str(exc)
        return False, diag

    reasons: list[str] = []
    if diag.get("psc_equation") != EXPECTED_PSC_EQUATION:
        reasons.append("psc_equation")
    if diag.get("smoothing") != EXPECTED_SMOOTHING:
        reasons.append("smoothing")
    if not diag.get("has_psc_baseline_mean"):
        reasons.append("psc_baseline_mean")
    if not diag.get("has_betas"):
        reasons.append("betas")
    elif diag["betas_shape"][0] != N_VERTICES:
        reasons.append("betas_vertex_count")
    reg_level = diag.get("regressor_level")
    if reg_level is not None and reg_level != EXPECTED_REGRESSOR_LEVEL:
        reasons.append("regressor_level")

    diag["fail_reasons"] = reasons
    return len(reasons) == 0, diag


def iter_run_beta_h5(beta_root: Path):
    """Yield H5 paths under subject/session/func, excluding grouped_betas and results."""
    skip_dir_names = {"grouped_betas", "results"}
    for h5 in sorted(beta_root.rglob("*_betas.h5")):
        parts = h5.parts
        if any(p in skip_dir_names for p in parts):
            continue
        if any(p.startswith("_quarantine") for p in parts):
            continue
        yield h5


def run_bundle_paths(h5_path: Path) -> list[Path]:
    """Sibling artifacts for one GLM run (same basename)."""
    base = h5_path.name.replace("_betas.h5", "")
    parent = h5_path.parent
    names = [
        f"{base}_betas.h5",
        f"{base}_design.csv",
        f"{base}_design_matrix.csv",
    ]
    return [parent / n for n in names if (parent / n).exists()]
