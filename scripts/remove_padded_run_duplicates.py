#!/usr/bin/env python3
"""
Remove zero-padded run labels (run-01, run-02, ...) when standard run-N exists.

Standard naming matches glasser_resampled BOLD and glm_analysis.py: run-1, run-2, ...
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RUN_PADDED_RE = re.compile(r"_(run-0\d+)(_|\.)")


def padded_to_standard(name: str) -> str | None:
    m = RUN_PADDED_RE.search(name)
    if not m:
        return None
    padded = m.group(1)
    n = int(padded.split("-")[1])
    return name.replace(padded, f"run-{n}", 1)


def iter_padded_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and RUN_PADDED_RE.search(p.name):
            yield p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas"
        ),
    )
    p.add_argument(
        "--also-legacy",
        action="store_true",
        help="Also clean data/trial_level_betas when encoding_delay has standard run-N.",
    )
    p.add_argument("--execute", action="store_true", help="Delete files (default: dry-run).")
    args = p.parse_args()

    encoding_root = Path(
        "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas"
    )

    to_delete: list[Path] = []
    for f in iter_padded_files(args.root):
        std_name = padded_to_standard(f.name)
        if std_name is None:
            continue
        std_path = f.parent / std_name
        if std_path.exists():
            to_delete.append(f)
            continue
        # Quarantine-only padded copies: drop if standard exists anywhere in encoding_delay
        if "_quarantine" in f.parts:
            rel = f.relative_to(args.root)
            # .../sub-01/ses-001/func/file -> build encoding path without quarantine
            parts = list(rel.parts)
            qidx = next(i for i, p in enumerate(parts) if p.startswith("_quarantine"))
            enc_rel = Path(*parts[qidx + 1 :])
            enc_std = encoding_root / enc_rel.parent / std_name
            if enc_std.exists():
                to_delete.append(f)

    if args.also_legacy:
        legacy_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas")
        if legacy_root.exists():
            for f in iter_padded_files(legacy_root):
                std_name = padded_to_standard(f.name)
                if std_name is None:
                    continue
                # sub-02/ses-010/func/... -> encoding_delay mirror
                rel = f.relative_to(legacy_root)
                enc_std = encoding_root / rel.parent / std_name
                if enc_std.exists():
                    to_delete.append(f)

    print(f"{'EXECUTE' if args.execute else 'DRY-RUN'}: {len(to_delete)} files to delete")
    for f in sorted(to_delete)[:20]:
        print(f"  {f}")
    if len(to_delete) > 20:
        print(f"  ... and {len(to_delete) - 20} more")

    if args.execute:
        for f in to_delete:
            f.unlink()
        print("Done.")


if __name__ == "__main__":
    main()
