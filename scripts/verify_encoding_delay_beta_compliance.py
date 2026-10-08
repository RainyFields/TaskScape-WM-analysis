#!/usr/bin/env python3
"""Report compliance of encoding-delay trial-level beta H5 files (exit 1 if any fail)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding_delay_beta_compliance import is_compliant_h5, iter_run_beta_h5
from config import paths  # noqa: E402  TaskScape-WM inputs only; rejects the legacy tree

# Input (TaskScape-WM): derivatives/glm/<sub>/<ses>/func/*_run-NN_betas.h5 (attrs psc_equation/smoothing, psc_baseline_mean).
# legacy equivalent: data/encoding_delay/trial_level_betas
# Output: <output_root>/encoding_delay_beta_compliance/compliance_report.json
# legacy equivalent: data/encoding_delay/trial_level_betas/compliance_report.json (written next to the inputs;
# the dataset is read-only, so the report now goes under the output root).


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta-root", type=Path, default=None, help="default: <dataset>/derivatives/glm")
    p.add_argument("--dataset_root", default=None, help="TaskScape-WM root (default: config.paths / $TASKSCAPE_DATASET_ROOT)")
    p.add_argument("--output_root", default=None, help="analysis output root (default: config.paths / $TASKSCAPE_OUTPUT_ROOT)")
    args = p.parse_args()
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard("verify_encoding_delay_beta_compliance")
    args.beta_root = paths.require_not_legacy(args.beta_root or paths.glm_dir(), "--beta-root")
    report_dir = paths.output_dir("encoding_delay_beta_compliance")

    by_subj = defaultdict(lambda: {"total": 0, "ok": 0})
    failures = []
    for h5 in iter_run_beta_h5(args.beta_root):
        rel = str(h5.relative_to(args.beta_root))
        parts = h5.parts
        subj = next((x for x in parts if x.startswith("sub-")), "unknown")
        by_subj[subj]["total"] += 1
        ok, diag = is_compliant_h5(h5)
        if ok:
            by_subj[subj]["ok"] += 1
        else:
            failures.append({"path": rel, "reasons": diag.get("fail_reasons"), "psc": diag.get("psc_equation")})

    report = {"by_subject": dict(by_subj), "n_failures": len(failures), "failures": failures[:20]}
    out = report_dir / "compliance_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
