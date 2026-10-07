#!/usr/bin/env python3
"""Authoritative run -> task-variant labels by matching each acquired run's stimulus sequence to the
PsychoPy block files that the session runner actually loads.

Sources
  raw events   : cneuromod.multfs.raw/sub-*/ses-*/func/*_events.tsv  (one row per trial, stimulus codes)
  block files  : task_stimuli/data/multfs/updated_cond_file/blockfiles/*.csv (+ pilot_DMS_loc.csv)
  lookup table : data/study_designs/{subj}_design_design_with_converted.tsv  (used by the legacy grouping code)
  runner design: task_stimuli/data/multfs/study_designs/{subj}_design.tsv    (what the runner reads)

The runner (task_stimuli/src/sessions/ses-multfs.py) derives feature and ABAB/ABBA order from the block
file *name*, so the matched block file identifies the task variant the participant actually performed.
Trials are presented in random order, so runs are matched as multisets of trial tuples.

Outputs (in --out_dir)
  run_labels_verified.csv      one row per acquired run (all tasks) with lookup / blockfile / behaviour labels
  lookup_mismatch_report.csv   runs where any two label sources disagree (or matching failed)
  lookup_mismatch_report.md    human-readable version

Usage
  python scripts/verify_run_labels_from_blockfiles.py --out_dir results/similarity_analysis/regenerated_clean
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import paths  # TaskScape-WM inputs only; rejects the legacy tree
STIM = paths.TASK_STIMULI   # EXTERNAL stimulus repository (revision recorded in config.paths)
BLOCKDIR = STIM / "updated_cond_file/blockfiles"
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_condition_identification as tci

TASK_RE = re.compile(r"sub-(\d+)_ses-(\d+)_task-(\w+?)_run-(\d+)_events\.tsv$")
FE = {"loc": "locmod", "obj": "objmod", "ctg": "ctgmod"}
STIM_COL_RE = re.compile(r"^(stim|loc|objmod|locmod|ctgmod|angmod|objlocmod|ref|action)\d*$")


def md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-Markdown table (avoids the optional `tabulate` dependency)."""
    df = df.reset_index() if df.index.name is not None else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def stim_columns(df):
    return sorted(c for c in df.columns if STIM_COL_RE.match(str(c).strip()))


def trial_multiset(df, cols):
    rows = df[cols].astype(str).apply(lambda r: "|".join(r.values), axis=1)
    return sorted(rows.tolist())


def load_blockfiles():
    blocks = {}
    for f in sorted(BLOCKDIR.glob("*.csv")):
        name = f.stem
        m = re.match(r"^(.*?)_?block_(\d+)$", name)
        label = m.group(1) if m else name
        df = pd.read_csv(f)
        blocks[name] = dict(label=label, df=df, cols=stim_columns(df))
    df = pd.read_csv(STIM / "updated_cond_file/pilot_DMS_loc.csv")
    blocks["pilot_DMS_loc"] = dict(label="dms_loc", df=df, cols=stim_columns(df))
    return blocks


def behaviour_agreement(df, label):
    """Fraction of responses consistent with `label` (interdms / nback only)."""
    def responses(k):
        c = f"response_{k}"
        return df[c].astype(str).str.strip().str.lower() if c in df else None
    if label.startswith("interdms"):
        _, feat, order = label.split("_"); feat = FE[feat]
        items = list(({2: 0, 3: 1} if order == "ABAB" else {2: 1, 3: 0}).items())
    elif label.startswith("nback"):
        feat = FE[label.split("_")[1]]; items = [(k, k - 1) for k in range(1, 6)]
    else:
        return np.nan
    ag = tot = 0
    for k, ref in items:
        r = responses(k)
        if r is None: continue
        same = (df[f"{feat}{k+1}"].values == df[f"{feat}{ref+1}"].values); ok = r.isin(["b", "x"]).values
        if not ok.any(): continue
        ag += max(((r.values == "b") == same)[ok].mean(), ((r.values == "x") == same)[ok].mean()) * ok.sum(); tot += ok.sum()
    return ag / tot if tot else np.nan


def behaviour_label(df, task):
    cands = ([f"interdms_{f}_{o}" for f in FE for o in ("ABAB", "ABBA")] if task == "interdms"
             else [f"nback_{f}" for f in FE] if task == "1back" else [])
    if not cands:
        return None, np.nan
    accs = {c: behaviour_agreement(df, c) for c in cands}
    best = max(accs, key=lambda c: np.nan_to_num(accs[c], nan=-1))
    return best, accs[best]


def runner_design_label(subj, ses, task, run):
    d = pd.read_csv(STIM / f"study_designs/{subj}_design.tsv", sep="\t")
    d.columns = d.columns.str.strip()
    rows = d[d.session.astype(int) == int(ses)]
    fam = {"1back": "nback", "interdms": "interdms", "ctxdm": "ctxdm"}.get(task)
    if fam is None:
        return "dms_loc" if task == "dms" else None
    blocks = [b.strip() for b in rows.block_file_name if b.strip().startswith(fam)]
    if int(run) - 1 < len(blocks):
        return tci._strip_block_suffix(blocks[int(run) - 1])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--output_root", default=None)
    args = ap.parse_args()
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard("verify_run_labels_from_blockfiles")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    blocks = load_blockfiles()
    print(f"{len(blocks)} block files loaded")
    rows = []
    for subj in SUBJECTS:
        lookup = pd.read_csv(paths.study_designs_dir() / f"{subj}_design_design_with_converted.tsv", sep="\t")
        glm_runs = {p.name.replace("_betas.h5", "") for p in (paths.glm_dir() / subj).glob("ses-*/func/*_betas.h5")}
        # original raw events (verbatim copies in the dataset), sorted by path as RAW.glob(...) was
        for ev in sorted(paths.raw_events_files(subj), key=lambda p: (p.parent.parent.name, p.name)):
            m = TASK_RE.search(ev.name)
            if not m or "__dup" in ev.name:
                continue
            ses, task, run = m.group(2), m.group(3), m.group(4)
            df = pd.read_csv(ev, sep="\t", encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            # lookup label as the legacy code resolves it
            try:
                lookup_label = tci.get_detailed_task_name(subj, f"ses-{ses}", task, f"run-{run}", lookup, strict=True)
            except Exception as e:
                lookup_label = f"LOOKUP_FAIL:{type(e).__name__}"
            # block-file matching (order-free)
            best = None
            for bname, b in blocks.items():
                cols = [c for c in b["cols"] if c in df.columns]
                if len(cols) < 4 or len(cols) < len(b["cols"]) * 0.8:
                    continue
                ev_ms = trial_multiset(df, cols); bl_ms = trial_multiset(b["df"], cols)
                inter = len(set(ev_ms) & set(bl_ms))
                frac = inter / max(len(set(bl_ms)), 1)
                exact = ev_ms == bl_ms
                cand = (exact, frac, len(cols), bname)
                if best is None or cand > best:
                    best = cand
            exact, frac, ncols, bname = best if best else (False, 0.0, 0, None)
            block_label = blocks[bname]["label"] if bname else None
            # second-best to detect ambiguity
            beh_label, beh_acc = behaviour_label(df, task)
            lookup_acc = behaviour_agreement(df, lookup_label) if lookup_label in (beh_label or "",) or task in ("interdms", "1back") else np.nan
            in_glm = f"{subj}_ses-{ses}_task-{task}_run-{int(run):02d}" in glm_runs   # dataset zero-pads run labels
            rows.append(dict(
                subject=subj, session=f"ses-{ses}", run=f"run-{int(run):02d}", bold_task=task, in_glm_outputs=in_glm,
                n_trials_events=len(df), lookup_label=lookup_label, runner_design_label=runner_design_label(subj, ses, task, run),
                blockfile_match=bname, blockfile_label=block_label, blockfile_exact=bool(exact),
                blockfile_trial_overlap=round(float(frac), 3), n_match_columns=ncols,
                behaviour_label=beh_label, behaviour_agreement=None if np.isnan(beh_acc) else round(float(beh_acc), 3),
                lookup_behaviour_agreement=None if (lookup_acc is None or np.isnan(lookup_acc)) else round(float(lookup_acc), 3),
            ))
    tab = pd.DataFrame(rows)

    # final label + action
    def decide(r):
        if r.blockfile_exact and r.blockfile_label:
            if r.blockfile_label == r.lookup_label:
                return r.blockfile_label, "keep", "block-file match confirms lookup label"
            return r.blockfile_label, "relabel_verified", "exact stimulus-sequence match to a different block file (authoritative)"
        if r.blockfile_label and r.blockfile_trial_overlap >= 0.9:
            if r.blockfile_label == r.lookup_label:
                return r.blockfile_label, "keep", f"partial block-file match ({r.blockfile_trial_overlap:.2f} trials) agrees with lookup"
            return r.blockfile_label, "relabel_verified_partial", f"partial block-file match ({r.blockfile_trial_overlap:.2f}) to a different block"
        return r.lookup_label, "unresolved_keep_lookup", "no block file matched; lookup label kept, run flagged"
    dec = tab.apply(decide, axis=1, result_type="expand"); dec.columns = ["final_label", "action", "justification"]
    tab = pd.concat([tab, dec], axis=1)
    tab["resolved"] = tab.action.isin(["keep", "relabel_verified", "relabel_verified_partial"])
    # Presented design (block file) is authoritative for the label; but flag runs where the participant's
    # responses fit a DIFFERENT rule much better than the presented one (probable rule/instruction error).
    fa = tab.apply(lambda r: r.behaviour_agreement if r.behaviour_label == r.final_label else r.lookup_behaviour_agreement
                   if r.final_label == r.lookup_label else np.nan, axis=1)
    tab["final_label_behaviour_agreement"] = fa
    tab["behaviour_inconsistent"] = (tab.behaviour_label.notna() & (tab.behaviour_label != tab.final_label)
                                     & (pd.to_numeric(tab.behaviour_agreement) >= 0.85)
                                     & ((pd.to_numeric(tab.behaviour_agreement) - pd.to_numeric(fa)) >= 0.20))
    tab.to_csv(out / "run_labels_verified.csv", index=False)

    dis = tab[(tab.lookup_label != tab.blockfile_label) | (~tab.blockfile_exact)
              | (tab.behaviour_label.notna() & (tab.behaviour_label != tab.final_label))]
    dis.to_csv(out / "lookup_mismatch_report.csv", index=False)

    n_exact = int(tab.blockfile_exact.sum()); n = len(tab)
    lines = ["# Run-to-condition label verification (block-file matching)", "",
             f"Runs checked: {n} (all acquired runs with an events file, excluding `__dup` files).",
             f"Exact block-file match: {n_exact}/{n}. Lookup label confirmed by block file: {int((tab.lookup_label == tab.blockfile_label).sum())}/{n}.",
             f"Relabelled from authoritative block file: {int(tab.action.str.startswith('relabel').sum())}. Unresolved: {int((~tab.resolved).sum())}.", "",
             "Authoritative source: `task_stimuli/data/multfs/updated_cond_file/blockfiles/*.csv`, loaded by "
             "`task_stimuli/src/sessions/ses-multfs.py`; the runner derives feature and ABAB/ABBA order from the block-file name.", "",
             "## Runs where sources disagree or matching is not exact", ""]
    cols = ["subject", "session", "run", "bold_task", "in_glm_outputs", "lookup_label", "runner_design_label", "blockfile_label",
            "blockfile_exact", "blockfile_trial_overlap", "behaviour_label", "behaviour_agreement", "lookup_behaviour_agreement",
            "final_label", "final_label_behaviour_agreement", "behaviour_inconsistent", "action", "justification"]
    if len(dis):
        lines.append(md_table(dis[cols]))
    else:
        lines.append("none")
    lines += ["", "## Action summary", "", md_table(tab.action.value_counts().rename_axis("action").to_frame("n_runs")), "",
              "## Per-subject relabels", "", md_table(tab[tab.action.str.startswith("relabel")].groupby("subject").size().to_frame("n_relabelled"))]
    (out / "lookup_mismatch_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(tab.action.value_counts())
    print(dis[cols].to_string(index=False))


if __name__ == "__main__":
    main()
