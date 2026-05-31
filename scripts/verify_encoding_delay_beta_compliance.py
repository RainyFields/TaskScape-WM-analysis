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

from encoding_delay_beta_compliance import is_compliant_h5, iter_run_beta_h5

DEFAULT_ROOT = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta-root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()

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
    out = args.beta_root / "compliance_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
