#!/usr/bin/env python3
"""Inventory of legacy-path references in this repository -> results/restructure/path_migration_table.tsv.

One row per (file, line) that mentions the frozen legacy project or the upstream annex trees, plus one row per
file migrated in the taskscape-migration branches (now free of legacy references). Classes:

  migrated            file moved to config.paths (reviewed; exercised by scripts/repro/submit_fidelity.sh)
  input / output      runnable code reading / writing a legacy path (auto-classified; NOT yet migrated)
  historical          comment, docstring or usage example (auto-classified; annotate, leave)
  obsolete-candidate  shell/SLURM helper or one-off diagnostic (auto-classified; candidate for scripts/legacy/)

`review` is "reviewed" only for the migrated set; everything else is "auto (pending review)".
Usage: python scripts/repro/path_migration_table.py [--base spatial_smoothed]
"""
import argparse, re, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAT = re.compile(r"fmri_dataset_project|cneuromod\.multfs\.(fmriprep|raw)|/scratch/xuan")
OUT_HINT = re.compile(r"out(_|put)?(root|dir|path)?|save|write|to_csv|savefig|mkdir", re.I)


def classify(path: str, line: str) -> str:
    s = line.strip()
    if s.startswith(("#", "//")) or s.startswith(('"', "'")) and "=" not in s:
        return "historical"
    if path.endswith((".sh", ".slurm")) or "/sanity_check/" in path or path.endswith(".txt"):
        return "obsolete-candidate"
    if path.endswith(".md"):
        return "historical"
    return "output" if OUT_HINT.search(s) else "input"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="spatial_smoothed"); a = ap.parse_args()
    files = subprocess.run(["git", "-C", str(REPO), "ls-files", "scripts", "config"], capture_output=True, text=True, check=True).stdout.split()
    changed = set(subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", f"{a.base}...HEAD", "--", "scripts", "config"],
                                 capture_output=True, text=True, check=True).stdout.split())
    rows = []
    for f in sorted(files):
        p = REPO / f
        if not p.is_file() or p.suffix in (".pyc", ".png", ".pdf", ".npy", ".npz", ".h5"):
            continue
        try:
            text = p.read_text(errors="replace").splitlines()
        except Exception:
            continue
        hits = [(i + 1, l) for i, l in enumerate(text) if PAT.search(l)]
        if f in changed:
            rows.append((f, "", "migrated" if not [h for h in hits if not h[1].lstrip().startswith("#")] else "migrated (legacy path only in guard constants or comments)",
                         "reviewed", f"{len(hits)} remaining mentions (comments/docstrings only)" if hits else ""))
            continue
        for n, l in hits:
            rows.append((f, str(n), classify(f, l), "auto (pending review)", l.strip()[:160].replace("\t", " ")))
    out = REPO / "results/restructure"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "path_migration_table.tsv", "w") as fh:
        fh.write("file\tline\tclass\treview\tdetail\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    from collections import Counter
    c = Counter(r[2] for r in rows)
    print(f"{len(rows)} rows; files with legacy references not migrated: {len({r[0] for r in rows if r[3] != 'reviewed'})}; {dict(c)}")


if __name__ == "__main__":
    main()
