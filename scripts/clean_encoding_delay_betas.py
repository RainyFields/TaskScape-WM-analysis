#!/usr/bin/env python3
"""
Quarantine non-compliant encoding-delay trial-level GLM outputs and clear derived PKLs.

Compliant run = spatially smoothed bold_sm4 input + PSC-normalized glm_analysis.py output
(see encoding_delay_beta_compliance.py).

After cleaning, re-run GLM with scripts/submit_glm_clean_regenerate.sh, then grouped betas with
scripts/submit_regenerate_grouped_betas.sh.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from encoding_delay_beta_compliance import (
    EXPECTED_PSC_EQUATION,
    EXPECTED_SMOOTHING,
    is_compliant_h5,
    iter_run_beta_h5,
    run_bundle_paths,
)

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
DEFAULT_BETA_ROOT = PROJECT / "data/encoding_delay/trial_level_betas"
DEFAULT_QUARANTINE = DEFAULT_BETA_ROOT / "_quarantine_pre_sm4_psc_cleanup"
GROUPED_DIR = DEFAULT_BETA_ROOT / "grouped_betas/task_relevant_only"


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beta-root", type=Path, default=DEFAULT_BETA_ROOT)
    p.add_argument("--quarantine-dir", type=Path, default=DEFAULT_QUARANTINE)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Move non-compliant files and delete grouped PKLs (default: dry-run only).",
    )
    p.add_argument(
        "--also-clean-legacy-root",
        action="store_true",
        help="Also quarantine non-compliant files under data/trial_level_betas (legacy path).",
    )
    return p.parse_args()


def quarantine_path(quarantine_root: Path, src: Path, beta_root: Path) -> Path:
    rel = src.relative_to(beta_root)
    return quarantine_root / rel


def clean_beta_root(beta_root: Path, quarantine_root: Path, execute: bool) -> dict:
    compliant: list[str] = []
    quarantined_runs: list[dict] = []
    by_subj = defaultdict(lambda: {"total": 0, "compliant": 0, "quarantined": 0})

    subj_re = re.compile(r"sub-\d+")
    for h5 in iter_run_beta_h5(beta_root):
        m = subj_re.search(str(h5.relative_to(beta_root)))
        subj = m.group(0) if m else "unknown"

        by_subj[subj]["total"] += 1
        ok, diag = is_compliant_h5(h5)
        if ok:
            by_subj[subj]["compliant"] += 1
            compliant.append(str(h5.relative_to(beta_root)))
            continue

        by_subj[subj]["quarantined"] += 1
        bundle = run_bundle_paths(h5)
        entry = {
            "h5": str(h5.relative_to(beta_root)),
            "fail_reasons": diag.get("fail_reasons", []),
            "psc_equation": diag.get("psc_equation"),
            "smoothing": diag.get("smoothing"),
            "estimator": diag.get("estimator"),
            "files": [str(p.relative_to(beta_root)) for p in bundle],
        }
        quarantined_runs.append(entry)

        if execute:
            for src in bundle:
                dst = quarantine_path(quarantine_root, src, beta_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    removed_pkls: list[str] = []
    if beta_root.resolve() == DEFAULT_BETA_ROOT.resolve() and GROUPED_DIR.exists():
        for pkl in sorted(GROUPED_DIR.glob("*_task_condition_betas.pkl")):
            removed_pkls.append(str(pkl.relative_to(PROJECT / "data")))
            if execute:
                q_dst = quarantine_root / "grouped_betas" / "task_relevant_only" / pkl.name
                q_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pkl), str(q_dst))

    return {
        "beta_root": str(beta_root),
        "compliant_runs": len(compliant),
        "quarantined_runs": len(quarantined_runs),
        "by_subject": dict(by_subj),
        "quarantined_run_details": quarantined_runs,
        "removed_grouped_pkls": removed_pkls,
        "compliant_run_paths": compliant,
    }


def main():
    args = get_args()
    beta_root = args.beta_root.resolve()
    quarantine_root = args.quarantine_dir.resolve()

    roots = [beta_root]
    if args.also_clean_legacy_root:
        legacy = (PROJECT / "data/trial_level_betas").resolve()
        if legacy != beta_root and legacy.exists():
            roots.append(legacy)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "expected_psc_equation": EXPECTED_PSC_EQUATION,
        "expected_smoothing": EXPECTED_SMOOTHING,
        "roots": [],
    }

    for root in roots:
        qdir = quarantine_root if root == beta_root else quarantine_root.parent / f"_legacy_{root.name}"
        summary = clean_beta_root(root, qdir, args.execute)
        summary["quarantine_dir"] = str(qdir)
        report["roots"].append(summary)

    manifest_path = beta_root / "cleanup_manifest.json"
    if args.execute:
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    else:
        manifest_path = beta_root / "cleanup_manifest_dryrun.json"
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")

    mode = "EXECUTED" if args.execute else "DRY-RUN"
    print(f"[{mode}] Cleanup report: {manifest_path}")
    for s in report["roots"]:
        print(
            f"  {s['beta_root']}: compliant={s['compliant_runs']}, "
            f"to_quarantine={s['quarantined_runs']}, grouped_pkls={len(s['removed_grouped_pkls'])}"
        )
    if not args.execute:
        print("Re-run with --execute to apply changes, then submit GLM regeneration jobs.")


if __name__ == "__main__":
    main()
