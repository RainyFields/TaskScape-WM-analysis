#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply wb_command -cifti-smoothing to BOLD dtseries files."
    )
    p.add_argument("--subj", required=True, help="Subject label, e.g. sub-01")
    p.add_argument(
        "--fmri_root",
        required=True,
        help="Root directory containing ses-*/func/*.dtseries.nii (or subj/ses-*/func/*.dtseries.nii)",
    )
    p.add_argument(
        "--surf_root",
        required=True,
        help="Directory containing midthickness surfaces (*.surf.gii)",
    )
    p.add_argument("--fwhm", type=float, default=4.0, help="Smoothing FWHM in mm.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names to process. Default: all discovered tasks.",
    )
    return p.parse_args()


def fwhm_tag(fwhm: float) -> str:
    if float(fwhm).is_integer():
        return f"{int(fwhm)}mm"
    return f"{str(fwhm).replace('.', 'p')}mm"


def find_subject_root(fmri_root: Path, subj: str) -> Path:
    subj_dir = fmri_root / subj
    if subj_dir.exists():
        return subj_dir
    return fmri_root


def find_surfaces(surf_root: Path):
    left_matches = sorted(surf_root.rglob("*hemi-L*midthickness*.surf.gii"))
    right_matches = sorted(surf_root.rglob("*hemi-R*midthickness*.surf.gii"))
    if not left_matches:
        raise FileNotFoundError(f"No left midthickness surface found in {surf_root}")
    if not right_matches:
        raise FileNotFoundError(f"No right midthickness surface found in {surf_root}")
    return left_matches[0], right_matches[0]


def extract_task(name: str):
    m = re.search(r"_task-([^_]+)_", name)
    return m.group(1) if m else None


def build_output_name(input_name: str, smooth_tag: str) -> str:
    suffix = "_bold.dtseries.nii"
    token = f"_smooth-{smooth_tag}{suffix}"
    if not input_name.endswith(suffix):
        return input_name
    return input_name.replace(suffix, token)


def main():
    args = parse_args()

    fmri_root = Path(args.fmri_root)
    surf_root = Path(args.surf_root)
    subject_root = find_subject_root(fmri_root, args.subj)
    smooth_tag = fwhm_tag(args.fwhm)

    surf_l, surf_r = find_surfaces(surf_root)
    print(f"[i] Using surfaces:\n  L: {surf_l}\n  R: {surf_r}")

    dtseries = sorted(
        subject_root.glob(f"ses-*/func/{args.subj}_*_space-Glasser64k_bold.dtseries.nii")
    )
    if not dtseries:
        print("No matching dtseries files found.")
        print("Smoothed 0 / 0 files. Errors: 0.")
        return

    tasks_wanted = set(args.tasks) if args.tasks else None
    if tasks_wanted is None:
        discovered = sorted({t for t in (extract_task(p.name) for p in dtseries) if t})
        print(f"[i] Tasks discovered: {discovered}")
    else:
        print(f"[i] Tasks requested: {sorted(tasks_wanted)}")

    selected = []
    for p in dtseries:
        task = extract_task(p.name)
        if tasks_wanted is None or (task in tasks_wanted):
            selected.append(p)

    total = len(selected)
    smoothed = 0
    errors = 0

    for input_path in selected:
        output_path = input_path.with_name(build_output_name(input_path.name, smooth_tag))
        if output_path.exists() and not args.overwrite:
            print(f"[↷] Exists, skipping: {output_path.name}")
            continue

        cmd = [
            "wb_command",
            "-cifti-smoothing",
            str(input_path),
            str(args.fwhm),
            str(args.fwhm),
            "COLUMN",
            str(output_path),
            "-left-surface",
            str(surf_l),
            "-right-surface",
            str(surf_r),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            errors += 1
            print(f"ERROR: {result.stderr}")
        else:
            smoothed += 1
            print(f"[✓] Smoothed: {output_path.name}")

    print(f"Smoothed {smoothed} / {total} files. Errors: {errors}.")


if __name__ == "__main__":
    main()
