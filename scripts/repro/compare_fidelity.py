#!/usr/bin/env python3
"""Migration fidelity: pre-migration outputs vs the same pipeline re-run on TaskScape-WM inputs.

Target max |delta| = 0 for every numeric value. Only fields listed in IGNORE_* (run-specific provenance:
paths, hostnames, timings, interpreter versions) are excluded, and every exclusion is written to the report.

Subcommands (all write <out>/<name>_comparison.tsv + .json):
  glm      new GLM H5 files vs the dataset's derivatives/glm H5 (= pre-migration files, hardlinked)
  clean    rebuilt clean delay betas vs derivatives/glm-clean-delay (pre-migration products)
  tables   a directory of CSV/NPY/NPZ/JSON outputs vs a reference directory (tracked tables at a tag)
Read-only on both sides. Reference directories come from `git archive <tag>` snapshots of this repo or from
the dataset, never from the legacy project tree (config.paths guard).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import paths  # noqa: E402

IGNORE_JSON_KEYS = {"elapsed_s", "elapsed_sec", "hostname", "slurm_job_id", "command", "python", "numpy", "pandas",
                    "git_commit_worktree", "labels_file", "input_dir", "out_dir", "atlas_path", "source_h5", "created",
                    "timestamp", "timestamp_utc"}
IGNORE_CSV_COLS = {"source_beta_file", "source_design_file", "source_events_file"}  # re-rooted provenance paths
IGNORE_H5_ATTRS = {"labels_file"}


def num_diff(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        an, bn = np.isnan(a.astype(float)), np.isnan(b.astype(float))
        if not np.array_equal(an, bn):
            return False, float("inf")
        d = np.abs(a.astype(float) - b.astype(float))[~an]
        m = float(d.max()) if d.size else 0.0
        return m == 0.0, m
    eq = np.array_equal(a, b)
    return eq, 0.0 if eq else float("inf")


DROP_VARIANT: str | None = None   # --drop_variant: reference rows/columns of a variant deliberately not re-run


def cmp_csv(new: Path, ref: Path) -> dict:
    if ref.read_text().startswith(",,"):   # two-row header pivot (variant_comparison_family_level.csv)
        a = pd.read_csv(new, header=[0, 1], index_col=[0, 1], float_precision="round_trip")
        b = pd.read_csv(ref, header=[0, 1], index_col=[0, 1], float_precision="round_trip")
        n_drop = 0
        if DROP_VARIANT:
            keep = [c for c in b.columns if c[1] != DROP_VARIANT]; n_drop = b.shape[1] - len(keep); b = b[keep]
        if list(a.columns) != list(b.columns) or not a.index.equals(b.index):
            return {"ok": False, "max_abs": float("inf"), "note": "pivot columns/index differ", "dropped_ref_columns": n_drop}
        e, m = num_diff(a.to_numpy(float), b.to_numpy(float))
        return {"ok": e, "max_abs": m, "dropped_ref_columns": n_drop}
    a, b = pd.read_csv(new, float_precision="round_trip"), pd.read_csv(ref, float_precision="round_trip")
    n_drop = 0
    if DROP_VARIANT and "variant" in b.columns:
        n_drop = int((b["variant"] == DROP_VARIANT).sum())
        b = b[b["variant"] != DROP_VARIANT].reset_index(drop=True)
    out = {"rows_new": len(a), "rows_ref": len(b), "dropped_ref_rows": n_drop, "ignored_columns": sorted(IGNORE_CSV_COLS & set(b.columns))}
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        out.update(ok=False, max_abs=float("inf"), note=f"columns/rows differ: {sorted(set(a.columns) ^ set(b.columns))}")
        return out
    worst, ok, bad = 0.0, True, []
    for c in a.columns:
        if c in IGNORE_CSV_COLS:
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            e, m = num_diff(x.to_numpy(), y.to_numpy())
        else:
            e = x.astype(str).equals(y.astype(str)); m = 0.0 if e else float("inf")
        if not e:
            ok = False; bad.append(c)
        worst = max(worst, m)
    out.update(ok=ok, max_abs=worst, differing_columns=bad)
    return out


def _json_walk(x, y, path, diffs, worst):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in IGNORE_JSON_KEYS:
                continue
            if k not in x or k not in y:
                diffs.append(f"{path}/{k}: missing"); continue
            worst = _json_walk(x[k], y[k], f"{path}/{k}", diffs, worst)
    elif isinstance(x, list) and isinstance(y, list) and DROP_VARIANT in y and DROP_VARIANT not in x:
        worst = _json_walk(x, [v for v in y if v != DROP_VARIANT], path, diffs, worst)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            worst = _json_walk(u, v, f"{path}[{i}]", diffs, worst)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        d = abs(float(x) - float(y)) if not (np.isnan(float(x)) and np.isnan(float(y))) else 0.0
        if d:
            diffs.append(f"{path}: {x} vs {y}")
        worst = max(worst, d)
    elif x != y:
        diffs.append(f"{path}: {str(x)[:80]} vs {str(y)[:80]}"); worst = float("inf")
    return worst


def cmp_json(new: Path, ref: Path) -> dict:
    diffs: list[str] = []
    worst = _json_walk(json.loads(new.read_text()), json.loads(ref.read_text()), "", diffs, 0.0)
    return {"ok": not diffs, "max_abs": worst, "differences": diffs[:20], "ignored_keys": sorted(IGNORE_JSON_KEYS)}


def cmp_npz(new: Path, ref: Path) -> dict:
    a = np.load(new, allow_pickle=True); b = np.load(ref, allow_pickle=True)
    if new.suffix == ".npy":
        e, m = num_diff(a, b); return {"ok": e, "max_abs": m}
    keys = sorted(set(a.files) | set(b.files)); worst, bad = 0.0, []
    for k in keys:
        if k not in a.files or k not in b.files:
            bad.append(f"{k}: missing"); worst = float("inf"); continue
        e, m = num_diff(a[k], b[k]); worst = max(worst, m)
        if not e:
            bad.append(k)
    return {"ok": not bad, "max_abs": worst, "differing_keys": bad}


def cmp_h5(new: Path, ref: Path, datasets: list[str] | None = None) -> dict:
    out, worst, bad = {}, 0.0, []
    with h5py.File(new, "r") as fa, h5py.File(ref, "r") as fb:
        names = datasets or sorted(set(fa.keys()) | set(fb.keys()))
        for k in names:
            if k not in fa or k not in fb:
                bad.append(f"{k}: missing"); worst = float("inf"); continue
            x, y = fa[k][()], fb[k][()]
            if x.dtype.kind in "SOU" or y.dtype.kind in "SOU":
                e = [str(v) for v in np.ravel(x)] == [str(v) for v in np.ravel(y)]; m = 0.0 if e else float("inf")
            else:
                e, m = num_diff(x, y)
            worst = max(worst, m)
            if not e:
                bad.append(k)
        for k in sorted(set(fa.attrs) | set(fb.attrs)):
            if k in IGNORE_H5_ATTRS:
                continue
            u, v = fa.attrs.get(k), fb.attrs.get(k)
            if not np.array_equal(np.asarray(u), np.asarray(v)):
                bad.append(f"attr:{k}")
    out.update(ok=not bad, max_abs=worst, differing=bad, ignored_attrs=sorted(IGNORE_H5_ATTRS))
    return out


def compare_file(new: Path, ref: Path) -> dict:
    s = new.name
    if s.endswith(".csv"):
        return cmp_csv(new, ref)
    if s.endswith(".json"):
        return cmp_json(new, ref)
    if s.endswith((".npz", ".npy")):
        return cmp_npz(new, ref)
    if s.endswith(".h5"):
        return cmp_h5(new, ref)
    same = new.read_bytes() == ref.read_bytes()
    return {"ok": same, "max_abs": 0.0 if same else float("inf"), "note": "byte comparison"}


def write(rows: list[dict], out: Path, name: str) -> dict:
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_comparison.tsv", sep="\t", index=False)
    cmp_ = df[df["ok"].notna()] if len(df) else df
    summ = {"name": name, "n_compared": len(cmp_), "n_identical": int(cmp_["ok"].astype(bool).sum()) if len(cmp_) else 0,
            "n_excluded": int(len(df) - len(cmp_)), "excluded": df.loc[df["ok"].isna(), "file"].tolist() if len(df) else [],
            "max_abs_overall": float(pd.to_numeric(cmp_["max_abs"]).max()) if len(cmp_) else None,
            "all_identical": bool(len(cmp_) and cmp_["ok"].astype(bool).all()),
            "differing": cmp_.loc[~cmp_["ok"].astype(bool), "file"].tolist()[:50] if len(cmp_) else []}
    (out / f"{name}_comparison.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["glm", "clean", "tables"])
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None, help="tables: reference dir; glm/clean: default from the dataset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--glob", nargs="+", default=["**/*"])
    ap.add_argument("--drop_variant", default=None, help="variant deliberately not re-run (e.g. legacy_pickle); dropped from the reference and reported")
    ap.add_argument("--exclude", nargs="*", default=[], help="substrings of relative paths to skip (reported)")
    a = ap.parse_args()
    global DROP_VARIANT
    DROP_VARIANT = a.drop_variant
    if a.drop_variant:
        a.exclude = [*a.exclude, f"_{a.drop_variant}"]   # per-variant files (e.g. matrices/*_legacy_pickle.npz)
    paths.configure(None, None)
    paths.install_legacy_guard(f"compare_fidelity_{a.mode}")
    rows = []
    if a.mode == "glm":
        # new: <new>/trial_level_betas/<sub>/<ses>/func/<stem>_run-NN_betas.h5 ; ref: dataset derivatives/glm (same names)
        for h5 in sorted(a.new.rglob("*_betas.h5")):
            sub, ses = h5.parts[-4], h5.parts[-3]
            ref = paths.glm_dir() / sub / ses / "func" / h5.name
            r = {"file": h5.name, "ref": str(ref)}
            r.update(cmp_h5(h5, ref) if ref.exists() else {"ok": False, "max_abs": float("inf"), "note": "no reference"})
            rows.append(r)
        write(rows, a.out, a.name or "glm")
    elif a.mode == "clean":
        ref_root = a.ref or paths.clean_delay_dir()
        for f in sorted(a.new.glob("sub-*_*")):
            sub = f.name.split("_")[0]
            ref = ref_root / sub / f.name if (ref_root / sub).is_dir() else ref_root / f.name
            r = {"file": f.name, "ref": str(ref)}
            r.update(compare_file(f, ref) if ref.exists() else {"ok": False, "max_abs": float("inf"), "note": "no reference"})
            rows.append(r)
        write(rows, a.out, a.name or "clean")
    else:
        refs = sorted({p for g in a.glob for p in a.ref.glob(g) if p.is_file()})
        for ref in refs:
            rel = ref.relative_to(a.ref)
            if any(x in str(rel) for x in a.exclude):
                rows.append({"file": str(rel), "ok": None, "max_abs": None, "note": "EXCLUDED by --exclude (not compared)"})
                continue
            new = a.new / rel
            r = {"file": str(rel)}
            r.update(compare_file(new, ref) if new.exists() else {"ok": False, "max_abs": float("inf"), "note": "missing in new"})
            rows.append(r)
        write(rows, a.out, a.name or "tables")


if __name__ == "__main__":
    main()
