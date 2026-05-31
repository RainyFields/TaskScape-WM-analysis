#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import numpy as np
import h5py


def print_result(label: str, status: str, msg: str):
    print(f"[{status}] {label}: {msg}")


def main():
    ap = argparse.ArgumentParser(description="Run sanity checks on a GLM _betas.h5 file.")
    ap.add_argument("h5_path", help="Path to *_betas.h5 file")
    args = ap.parse_args()

    passed = 0
    total = 6
    hard_fail = False

    with h5py.File(args.h5_path, "r") as f:
        if "psc_baseline_mean" not in f:
            print_result("CHECK 1 — PSC baseline mean", "FAIL", "psc_baseline_mean dataset missing")
            hard_fail = True
            mu_mean = np.nan
        else:
            mu = f["psc_baseline_mean"][:]
            mu_mean = float(np.nanmean(mu))
            if not np.isfinite(mu_mean) or mu_mean < 10.0:
                print_result(
                    "CHECK 1 — PSC baseline mean",
                    "FAIL",
                    f"nanmean={mu_mean} (near-zero or non-finite baseline)",
                )
                hard_fail = True
            elif 100.0 <= mu_mean <= 50000.0:
                print_result("CHECK 1 — PSC baseline mean", "PASS", f"nanmean={mu_mean:.3f}")
                passed += 1
            else:
                print_result(
                    "CHECK 1 — PSC baseline mean",
                    "FAIL",
                    f"nanmean={mu_mean:.3f} (outside [100, 50000])",
                )
                hard_fail = True

        if "betas" not in f:
            print_result("CHECK 2 — Beta range", "FAIL", "betas dataset missing")
            print_result("CHECK 3 — Beta std", "FAIL", "betas dataset missing")
            print_result("CHECK 4 — Near-zero fraction", "FAIL", "betas dataset missing")
            print_result("CHECK 5 — No NaN or Inf in betas", "FAIL", "betas dataset missing")
            hard_fail = True
            betas = None
        else:
            betas = f["betas"][:]
            absmax = float(np.nanmax(np.abs(betas)))
            bstd = float(np.nanstd(betas))
            near_zero_frac = float(np.mean(np.abs(betas) < 1e-6))
            all_finite = bool(np.all(np.isfinite(betas)))

            # CHECK 2 — Beta range
            if absmax < 1e-4 or absmax > 500:
                print_result("CHECK 2 — Beta range", "FAIL", f"absmax={absmax:.6g}")
                hard_fail = True
            elif 0.01 <= absmax < 200:
                print_result("CHECK 2 — Beta range", "PASS", f"absmax={absmax:.6g}")
                if absmax > 20:
                    print("  ⚠ absmax > 20 — check whether outliers are in valid_mask=False grayordinates")
                passed += 1
            else:
                print_result("CHECK 2 — Beta range", "FAIL", f"absmax={absmax:.6g} (outside [0.01, 200])")
                hard_fail = True

            # CHECK 3 — Beta std
            if bstd < 1e-4:
                print_result("CHECK 3 — Beta std", "FAIL", f"std={bstd:.6g}")
                hard_fail = True
            elif bstd > 0.01:
                print_result("CHECK 3 — Beta std", "PASS", f"std={bstd:.6g}")
                passed += 1
            else:
                print_result("CHECK 3 — Beta std", "FAIL", f"std={bstd:.6g} (<= 0.01)")
                hard_fail = True

            # CHECK 4 — Near-zero fraction
            if near_zero_frac < 0.3:
                print_result("CHECK 4 — Near-zero fraction", "PASS", f"fraction={near_zero_frac:.4f}")
                passed += 1
            else:
                print_result("CHECK 4 — Near-zero fraction", "FAIL", f"fraction={near_zero_frac:.4f}")
                hard_fail = True

            # CHECK 5 — No NaN or Inf in betas
            if all_finite:
                print_result("CHECK 5 — No NaN or Inf in betas", "PASS", "all finite")
                passed += 1
            else:
                print_result("CHECK 5 — No NaN or Inf in betas", "FAIL", "contains NaN or Inf")
                hard_fail = True

        # CHECK 6 — PSC equation metadata
        eq = f.attrs.get("psc_equation", "")
        if isinstance(eq, bytes):
            eq = eq.decode("utf-8")
        eq = str(eq)
        eq_low = eq.lower()
        if ("mean(y_raw)" in eq_low) or ("y_hp / mean" in eq_low):
            print_result("CHECK 6 — PSC equation metadata", "PASS", eq)
            passed += 1
        elif "mean(y_hp)" in eq_low:
            print_result("CHECK 6 — PSC equation metadata", "WARN", eq)
        else:
            print_result("CHECK 6 — PSC equation metadata", "FAIL", eq if eq else "missing")
            hard_fail = True

    print(f"\n{passed}/{total} checks passed")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
