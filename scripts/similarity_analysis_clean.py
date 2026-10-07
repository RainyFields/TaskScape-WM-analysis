#!/usr/bin/env python3
"""Within- vs across-subject spatial-pattern similarity of delay-period activation maps (Figure 8, regenerated).

Two stages (both meant for SLURM compute nodes):

  --stage halves --subj sub-01
      For one subject and every data variant, draw 50 random non-overlapping split-half partitions of the
      task's trials (seeded, recorded), average each half -> two vertex-level maps (valid vertices only) and
      save them to --work_dir/halves_<variant>_<subj>.npz together with the half assignment of every record.

  --stage similarity
      Load all subjects' half maps; for each variant, spatial resolution (vertex / 360 Glasser regions / 12
      Cole-Anticevic networks), task and split: z-score each subject's half map across units, Pearson-correlate
      half A of subject i with half B of subject j (5 x 5 matrix). Within = diagonal, across = 10 unordered
      pairs symmetrised as 0.5*(r_ij + r_ji). Writes long-form tables, full 5x5 matrices, task-/family-level
      summaries, negative/positive controls and sensitivity variants.

Data variants (all from data/encoding_delay/clean_delay_betas unless stated)
  clean_verified            primary: fixed indexing, one record per physical delay beta, block-file-verified labels
  clean_lookup              same, but run labels from the legacy lookup table
  clean_exclude_inconsistent primary minus runs whose behaviour fits another rule (run_labels_verified.behaviour_inconsistent)
  clean_behaviour_relabel   primary with those runs relabelled by behaviour (sensitivity only)
  fixed_with_duplicates     fixed indexing, lookup labels, legacy pair+single duplication reproduced by weights
  legacy_pickle             the mis-indexed production pickles (grouped_betas/task_relevant_only), production grouping code
  clean_verified_familypooled primary, trials pooled within task family (separate estimand)
  clean_verified_sessionscale primary with a random positive per-session scalar applied to trials (invariance check)
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, nibabel as nib

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS)); sys.path.insert(0, str(_SCRIPTS / "figure_making"))
from utils import load_valid_vertex_mask, apply_mask_to_betas_dict
from config import paths  # TaskScape-WM inputs only; rejects the legacy tree
import plot_per_task_within_across_subj_similarity as prod   # production grouping (legacy variant only)

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
TASKS = ["dms_loc", "nback_loc", "nback_ctg", "nback_obj",
         "interdms_loc_ABAB", "interdms_loc_ABBA", "interdms_ctg_ABAB", "interdms_ctg_ABBA",
         "interdms_obj_ABAB", "interdms_obj_ABBA", "ctxdm_lco", "ctxdm_col"]
FAMILY = {"dms_loc": "DMS", "nback_loc": "1-Back", "nback_ctg": "1-Back", "nback_obj": "1-Back", "ctxdm_lco": "CTXDM", "ctxdm_col": "CTXDM",
          **{t: "InterDMS" for t in TASKS if t.startswith("interdms")}}
FAMILIES = ["DMS", "1-Back", "InterDMS", "CTXDM"]
PRETTY = {"dms_loc": "DMS", "nback_loc": "1-Back location", "nback_ctg": "1-Back category", "nback_obj": "1-Back identity",
          "ctxdm_lco": "CTXDM LCI", "ctxdm_col": "CTXDM CIL",
          **{f"interdms_{f}_{o}": f"InterDMS {n} {o}" for f, n in (("loc", "location"), ("ctg", "category"), ("obj", "identity")) for o in ("ABAB", "ABBA")}}
BASE_SEED = 20260926
N_SPLITS = 50
EXTREME_PSC = 100.0         # record excluded (robust variant) if the 95th percentile of |beta| across cortex exceeds this (PSC)
BASELINE_FLOOR = 1.0        # vertex excluded (robust variant) if its PSC baseline (run mean signal) is below this in any run
# 1.0 (not 1000): a floor of 1000 emptied Glasser parcels (job 21874759). Near-zero baselines are the
# PSC-undefined case; typical valid baselines are several thousand.
RESOLUTIONS = ["vertex", "region", "network"]
CLEAN: Path = None       # set in main(): --clean_dir (default <dataset>/derivatives/glm-clean-delay/<sub>/)
PKL_DIR: Path | None = None  # legacy_pickle variant only when --legacy_pickle_dir is given (invalid published input;
                             # not part of the TaskScape-WM dataset -> skipped by default)


def clean_file(subj: str, name: str) -> Path:
    # dataset layout keeps one folder per subject; a flat directory (build_clean_delay_betas --out_dir) also works
    p = CLEAN / subj / f"{subj}_{name}"
    return p if p.exists() else CLEAN / f"{subj}_{name}"


def split_rng(split, subj_idx, task_idx):
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, split, subj_idx, task_idx]))


def half_maps(X, w, split, subj_idx, task_idx):
    """X: (n, units) trials; w: integer weights (record multiplicity). Returns (2, units) means and assignment."""
    rows = np.repeat(np.arange(X.shape[0]), w)          # duplicated rows appear several times, as in legacy pooling
    rng = split_rng(split, subj_idx, task_idx)
    perm = rng.permutation(len(rows)); mid = len(rows) // 2
    a, b = rows[perm[:mid]], rows[perm[mid:]]
    assign = np.full(X.shape[0], -1, np.int8)
    assign[np.unique(a)] = 0; assign[np.unique(b)] = 1
    both = np.intersect1d(np.unique(a), np.unique(b)); assign[both] = 2     # only possible with duplicates (weights > 1)
    return np.stack([X[a].mean(0), X[b].mean(0)]), assign


# ------------------------------------------------------------------------------------------------ stage 1
def stage_halves(args):
    subj = args.subj; si = SUBJECTS.index(subj)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    mask = load_valid_vertex_mask(paths.valid_vertex_mask())
    t0 = time.time()
    prov = pd.read_csv(clean_file(subj, "provenance.csv"))
    ret = prov[prov.retained].reset_index(drop=True)
    assert (ret.record_id.values == np.arange(len(ret))).all()
    with h5py.File(clean_file(subj, "delay_betas.h5"), "r") as f:
        X = f["betas"][()][:, mask]
    print(f"[{subj}] clean betas {X.shape} loaded in {time.time()-t0:.0f}s")
    labs = pd.read_csv(args.labels); labs = labs[labs.subject == subj]
    flag = labs[labs.behaviour_inconsistent].set_index(["session", "run"])
    key = list(zip(ret.session, ret.run))
    inconsistent = np.array([k in flag.index for k in key])
    beh_label = np.array([flag.loc[k, "behaviour_label"] if k in flag.index else l for k, l in zip(key, ret.task_variant)])
    ses_idx = pd.factorize(ret.session)[0]
    ses_scale = np.random.default_rng(BASE_SEED + 7).lognormal(0.0, 0.3, size=ses_idx.max() + 1).astype(np.float32)

    # per-record robust QC (|beta| percentiles across valid vertices); records whose 95th percentile exceeds
    # EXTREME_PSC are whole-map blow-ups (implausible BOLD) and are excluded in the robust sensitivity variant only
    absX = np.abs(X)
    p50, p95, pmax = np.percentile(absX, 50, axis=1), np.percentile(absX, 95, axis=1), absX.max(1); del absX
    qc_dir = Path(args.out_dir) / "qc"; qc_dir.mkdir(parents=True, exist_ok=True)
    rq = ret[["subject", "session", "run", "task_variant", "trial_number", "stim_order", "record_id"]].copy()
    rq["abs_beta_p50"], rq["abs_beta_p95"], rq["abs_beta_max"] = p50, p95, pmax
    rq["extreme_record"] = p95 > EXTREME_PSC
    rq.to_csv(qc_dir / f"{subj}_record_qc.csv", index=False)
    print(f"[{subj}] extreme records (p95 |beta| > {EXTREME_PSC}): {int(rq.extreme_record.sum())} / {len(rq)}")

    variants = {
        "clean_verified_robust": (ret.task_variant.values, (~rq.extreme_record.values).astype(int), X),
        "clean_verified": (ret.task_variant.values, np.ones(len(ret), int), X),
        "clean_lookup": (ret.label_lookup.values, np.ones(len(ret), int), X),
        "clean_exclude_inconsistent": (ret.task_variant.values, (~inconsistent).astype(int), X),
        "clean_behaviour_relabel": (beh_label, np.ones(len(ret), int), X),
        "fixed_with_duplicates": (ret.label_lookup.values, ret.legacy_multiplicity.values.astype(int), X),
        "clean_verified_familypooled": (ret.task_variant.map(FAMILY).values, np.ones(len(ret), int), X),
        "clean_verified_sessionscale": (ret.task_variant.values, np.ones(len(ret), int), None),   # X scaled lazily
    }
    for name, (labels, w, Xv) in variants.items():
        if Xv is None:
            Xv = X * ses_scale[ses_idx][:, None]
        task_list = FAMILIES if name.endswith("familypooled") else TASKS
        H = np.zeros((len(task_list), N_SPLITS, 2, X.shape[1]), np.float32)
        A = np.full((len(task_list), N_SPLITS, len(ret)), -1, np.int8)
        n_used = np.zeros(len(task_list), int)
        for ti, task in enumerate(task_list):
            sel = np.where((labels == task) & (w > 0))[0]
            n_used[ti] = int(w[sel].sum())
            if len(sel) < 4:
                H[ti] = np.nan; continue
            tidx = ti if task_list is TASKS else 100 + ti
            for sp in range(N_SPLITS):
                h, a = half_maps(Xv[sel], w[sel], sp, si, tidx)
                H[ti, sp] = h; A[ti, sp, sel] = a
        np.savez(work / f"halves_{name}_{subj}.npz", halves=H, assign=A, tasks=np.array(task_list), n_trials=n_used,
                 record_ids=ret.record_id.values, vertex_ids=np.where(mask)[0])
        print(f"[{subj}] {name}: n per task {dict(zip(task_list, n_used))} ({time.time()-t0:.0f}s)")
    del X

    # legacy production pickle -> production grouping code (mis-indexed + duplicated, as published)
    if PKL_DIR is None:
        print(f"[{subj}] legacy_pickle variant skipped (no --legacy_pickle_dir; invalid input not in TaskScape-WM)")
        return
    with open(PKL_DIR / f"{subj}_task_condition_betas.pkl", "rb") as f:
        raw = pickle.load(f)
    raw = apply_mask_to_betas_dict(raw, mask)
    grouped, _ = prod.group_betas(raw, "task"); del raw
    H = np.full((len(TASKS), N_SPLITS, 2, int(mask.sum())), np.nan, np.float32); n_used = np.zeros(len(TASKS), int)
    for ti, task in enumerate(TASKS):
        if task not in grouped or len(grouped[task]) < 4:
            continue
        M = np.stack([np.asarray(v).reshape(-1) for v in grouped[task]], 0).astype(np.float32); n_used[ti] = len(M)
        for sp in range(N_SPLITS):
            H[ti, sp], _ = half_maps(M, np.ones(len(M), int), sp, si, ti)
    np.savez(work / f"halves_legacy_pickle_{subj}.npz", halves=H, tasks=np.array(TASKS), n_trials=n_used, vertex_ids=np.where(mask)[0])
    print(f"[{subj}] legacy_pickle: n per task {dict(zip(TASKS, n_used))} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ stage 2
def pearson_rows(A, B):
    Az = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-8)
    Bz = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-8)
    return Az @ Bz.T / A.shape[1]


class Aggregator:
    """vertex (masked) -> 360 region means -> 12 network means (unweighted mean of regions, as in production)."""
    def __init__(self, mask, vertex_keep=None):
        lab = np.asarray(nib.load(str(paths.glasser_dlabel())).get_fdata()).reshape(-1).astype(int)[mask]
        net = np.loadtxt(paths.parcel_networks()).astype(int)
        keep = np.ones(lab.size, bool) if vertex_keep is None else vertex_keep
        self.keep = keep
        self.region_ids = np.arange(1, 361)
        self.R = np.zeros((360, lab.size), np.float32)
        empty = []
        for j, rid in enumerate(self.region_ids):
            m = (lab == rid) & keep
            if m.sum() == 0:
                empty.append(int(rid)); continue
            self.R[j, m] = 1.0 / m.sum()
        if empty and vertex_keep is None:
            raise AssertionError(f"empty parcel(s) under the shared valid-vertex mask: {empty}")
        if empty:
            print(f"[Aggregator] skipping {len(empty)} empty parcels after extra vertex mask: {empty[:12]}{'...' if len(empty) > 12 else ''}")
        self.network_ids = np.arange(1, 13)
        self.N = np.zeros((12, 360), np.float32)
        for j, nid in enumerate(self.network_ids):
            m = net == nid; self.N[j, m] = 1.0 / m.sum()
        self.n_vertex = int(keep.sum())

    def transform(self, H, res):          # H (..., n_vertex_masked)
        if res == "vertex":
            return H[..., self.keep]
        reg = H @ self.R.T
        return reg if res == "region" else reg @ self.N.T


def load_halves(work, variant):
    out = {}
    for s in SUBJECTS:
        z = np.load(work / f"halves_{variant}_{s}.npz", allow_pickle=True)
        out[s] = dict(H=z["halves"], tasks=list(z["tasks"]), n=z["n_trials"])
    return out


def similarity_from_halves(Hs, agg, res):
    """Hs: dict subj -> (n_tasks, n_splits, 2, V). Returns r (n_tasks, n_splits, 5, 5) at resolution res."""
    T = {s: agg.transform(Hs[s], res) for s in SUBJECTS}
    nt, ns = next(iter(T.values())).shape[:2]
    r = np.full((nt, ns, 5, 5), np.nan, np.float32)
    for ti in range(nt):
        for sp in range(ns):
            A = np.stack([T[s][ti, sp, 0] for s in SUBJECTS]); B = np.stack([T[s][ti, sp, 1] for s in SUBJECTS])
            if np.isnan(A).any() or np.isnan(B).any():
                continue
            r[ti, sp] = pearson_rows(A.astype(np.float64), B.astype(np.float64))
    return r


def tables_from_r(r, tasks, variant, res, subjects=SUBJECTS):
    within, across = [], []
    fam = lambda t: FAMILY.get(t, t)
    for ti, t in enumerate(tasks):
        for sp in range(r.shape[1]):
            M = r[ti, sp]
            if np.isnan(M).all():
                continue
            for i, s in enumerate(subjects):
                within.append(dict(variant=variant, subject=s, task=t, task_family=fam(t), spatial_resolution=res, split_iteration=sp, within_similarity=float(M[i, i])))
            for i in range(5):
                for j in range(i + 1, 5):
                    across.append(dict(variant=variant, subject_1=subjects[i], subject_2=subjects[j], task=t, task_family=fam(t), spatial_resolution=res,
                                       split_iteration=sp, across_similarity=float(0.5 * (M[i, j] + M[j, i]))))
    return pd.DataFrame(within), pd.DataFrame(across)


def summarise(within, across):
    """Average over splits first, then over subjects / pairs, then over tasks within family."""
    g = ["variant", "spatial_resolution", "task", "task_family"]
    w_st = within.groupby(g + ["subject"]).within_similarity.agg(mean="mean", sd_splits="std").reset_index()
    a_pt = across.groupby(g + ["subject_1", "subject_2"]).across_similarity.agg(mean="mean", sd_splits="std").reset_index()
    w_task = w_st.groupby(g).agg(within_mean=("mean", "mean"), within_sd_subjects=("mean", "std"), within_min=("mean", "min"), within_max=("mean", "max"), n_subjects=("mean", "size")).reset_index()
    a_task = a_pt.groupby(g).agg(across_mean=("mean", "mean"), across_sd_pairs=("mean", "std"), across_min=("mean", "min"), across_max=("mean", "max"), n_pairs=("mean", "size")).reset_index()
    # Monte-Carlo SD of the task-level estimate: SD over splits of the per-split mean across subjects/pairs
    w_mc = within.groupby(g + ["split_iteration"]).within_similarity.mean().groupby(level=list(range(4))).std().rename("within_mc_sd_splits").reset_index()
    a_mc = across.groupby(g + ["split_iteration"]).across_similarity.mean().groupby(level=list(range(4))).std().rename("across_mc_sd_splits").reset_index()
    task = w_task.merge(a_task, on=g).merge(w_mc, on=g).merge(a_mc, on=g)
    task["task_label"] = task.task.map(lambda t: PRETTY.get(t, t))
    gf = ["variant", "spatial_resolution", "task_family"]
    fam = task.groupby(gf).agg(within_mean=("within_mean", "mean"), within_sd_tasks=("within_mean", "std"), across_mean=("across_mean", "mean"),
                               across_sd_tasks=("across_mean", "std"), n_tasks=("task", "size")).reset_index()
    return w_st, a_pt, task, fam


def stage_similarity(args):
    work = Path(args.work_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "matrices").mkdir(exist_ok=True)
    mask = load_valid_vertex_mask(paths.valid_vertex_mask())
    agg = Aggregator(mask)
    t0 = time.time()
    variants = ["clean_verified", "clean_lookup", "clean_exclude_inconsistent", "clean_behaviour_relabel", "fixed_with_duplicates",
                "legacy_pickle", "clean_verified_familypooled", "clean_verified_sessionscale"]
    if not all((work / f"halves_legacy_pickle_{s}.npz").exists() for s in SUBJECTS):
        variants.remove("legacy_pickle")
        print("[similarity] legacy_pickle variant omitted (halves not built; legacy pickles are not a TaskScape-WM input)")
    W, A, ntr = [], [], []
    halves_primary = None
    for v in variants:
        Hd = load_halves(work, v)
        Hs = {s: Hd[s]["H"] for s in SUBJECTS}; tasks = Hd[SUBJECTS[0]]["tasks"]
        for s in SUBJECTS:
            for ti, t in enumerate(tasks):
                ntr.append(dict(variant=v, subject=s, task=t, n_trials_pooled=int(Hd[s]["n"][ti])))
        mats = {}
        for res in RESOLUTIONS:
            r = similarity_from_halves(Hs, agg, res); mats[res] = r
            w, a = tables_from_r(r, tasks, v, res); W.append(w); A.append(a)
        np.savez_compressed(out / "matrices" / f"similarity_matrices_{v}.npz", tasks=np.array(tasks), subjects=np.array(SUBJECTS), **mats)
        if v == "clean_verified":
            halves_primary = Hs
        print(f"[{v}] done ({time.time()-t0:.0f}s)")
    W = pd.concat(W, ignore_index=True); A = pd.concat(A, ignore_index=True)
    W.to_csv(out / "within_similarity_long.csv", index=False); A.to_csv(out / "across_similarity_long.csv", index=False)
    pd.DataFrame(ntr).to_csv(out / "n_trials_per_variant.csv", index=False)
    w_st, a_pt, task, fam = summarise(W, A)
    w_st.to_csv(out / "summary_subject_task.csv", index=False); a_pt.to_csv(out / "summary_pair_task.csv", index=False)
    task.to_csv(out / "summary_task_level.csv", index=False); fam.to_csv(out / "summary_family_level.csv", index=False)

    # ---------------------------------------------------------------- controls on the primary variant
    Hs = halves_primary; ctrl_rows = []
    def run_ctrl(name, Hc, subjects=SUBJECTS, note=""):
        for res in RESOLUTIONS:
            r = similarity_from_halves(Hc, agg, res)
            w, a = tables_from_r(r, TASKS, name, res, subjects)
            ws, as_, tk, fm = summarise(w, a)
            for row in fm.itertuples():
                ctrl_rows.append(dict(control=name, spatial_resolution=res, task_family=row.task_family, within_mean=row.within_mean, across_mean=row.across_mean, note=note))
            # pair-specific summaries for the manipulated subject
            for s2 in SUBJECTS[1:]:
                sub = as_[((as_.subject_1 == SUBJECTS[0]) & (as_.subject_2 == s2))]
                ctrl_rows.append(dict(control=name, spatial_resolution=res, task_family="all", within_mean=ws[ws.subject == SUBJECTS[0]]["mean"].mean(),
                                      across_mean=sub["mean"].mean(), note=f"pair {SUBJECTS[0]}-{s2} (task-averaged)"))
            np.savez_compressed(out / "matrices" / f"control_{name}_{res}.npz", r=r, tasks=np.array(TASKS))
    # baseline (for paired comparison)
    run_ctrl("baseline_primary", Hs, note="clean_verified, unmodified")
    # 1. spatial shuffle of sub-01 (same vertex permutation for both halves and all tasks)
    perm = np.random.default_rng(BASE_SEED + 11).permutation(int(mask.sum()))
    Hc = dict(Hs); Hc[SUBJECTS[0]] = Hs[SUBJECTS[0]][..., perm]
    run_ctrl("spatial_shuffle_sub01", Hc, note="sub-01 vertex order permuted (seed BASE+11); within(sub-01) unchanged, across(sub-01,*) expected ~0")
    # 2. task-label shuffle of sub-01 (derangement of the 12 task labels; both halves moved together)
    rng = np.random.default_rng(BASE_SEED + 13)
    while True:
        tperm = rng.permutation(len(TASKS))
        if (tperm != np.arange(len(TASKS))).all():
            break
    Hc = dict(Hs); Hc[SUBJECTS[0]] = Hs[SUBJECTS[0]][tperm]
    run_ctrl("task_shuffle_sub01", Hc, note=f"sub-01 task labels deranged {tperm.tolist()} (seed BASE+13); within unchanged, across tests task specificity")
    # 3. clone: sub-02 replaced by sub-01 (halves from the next split iteration -> independent trial partition)
    Hc = dict(Hs); Hc[SUBJECTS[1]] = np.roll(Hs[SUBJECTS[0]], -1, axis=1)
    run_ctrl("clone_sub01_as_sub02", Hc, note="sub-02 := sub-01 with split i+1; across(sub-01,sub-02) expected ~ within(sub-01)")
    # 4. positive affine transform of sub-03 maps
    Hc = dict(Hs); Hc[SUBJECTS[2]] = Hs[SUBJECTS[2]] * np.float32(3.7) + np.float32(12.0)
    run_ctrl("affine_sub03", Hc, note="sub-03 maps * 3.7 + 12; all correlations must be unchanged")
    ctrl = pd.DataFrame(ctrl_rows)
    # paired deltas vs baseline
    base = ctrl[ctrl.control == "baseline_primary"].set_index(["spatial_resolution", "task_family", "note"])
    ctrl.to_csv(out / "controls_summary.csv", index=False)
    aff = ctrl[ctrl.control == "affine_sub03"].set_index(["spatial_resolution", "task_family"])
    bl = ctrl[ctrl.control == "baseline_primary"].set_index(["spatial_resolution", "task_family"])
    # exact affine check on the matrices
    dmax = 0.0
    for res in RESOLUTIONS:
        ra = np.load(out / "matrices" / f"control_affine_sub03_{res}.npz")["r"]; rb = np.load(out / "matrices" / f"control_baseline_primary_{res}.npz")["r"]
        dmax = max(dmax, float(np.nanmax(np.abs(ra - rb))))
    print(f"affine invariance max |dr| = {dmax:.2e}")

    # 5. extreme PSC values: locate + report; robust rerun = (i) drop records with p95 |beta| > EXTREME_PSC (whole-map blow-ups,
    #    computed in stage 1 -> variant clean_verified_robust) and (ii) drop vertices whose PSC baseline < BASELINE_FLOOR in any run
    #    of any subject (PSC ill-defined when the mean signal is near zero). Nothing is winsorized; the primary variant is untouched.
    ext_rows, summ_rows = [], []; bad_union = np.zeros(int(mask.sum()), bool)
    for s in SUBJECTS:
        q = np.load(clean_file(s, "vertex_qc.npz")); vmax = q["vertex_max_abs"][mask]; vbase = q["vertex_min_baseline"][mask]
        rq = pd.read_csv(out / "qc" / f"{s}_record_qc.csv")
        bad = vbase < BASELINE_FLOOR; bad_union |= bad
        top = rq.sort_values("abs_beta_max", ascending=False).head(20)
        for r in top.itertuples():
            v = int(q["record_argmax_vertex_masked"][int(r.record_id)])
            ext_rows.append(dict(subject=s, session=r.session, run=r.run, task_variant=r.task_variant, trial_number=r.trial_number, stim_order=r.stim_order,
                                 record_id=int(r.record_id), abs_beta_max=float(r.abs_beta_max), abs_beta_p95=float(r.abs_beta_p95), abs_beta_p50=float(r.abs_beta_p50),
                                 argmax_vertex=v, argmax_vertex_min_psc_baseline=float(q["vertex_min_baseline"][v]),
                                 likely_origin="near-zero PSC baseline at vertex" if q["vertex_min_baseline"][v] < BASELINE_FLOOR else "trial-level noise / spike (baseline normal)"))
        by_run = rq.groupby(["session", "run"]).extreme_record.sum().sort_values(ascending=False).head(5)
        summ_rows.append(dict(subject=s, n_records=len(rq), n_extreme_records=int(rq.extreme_record.sum()), frac_extreme_records=round(float(rq.extreme_record.mean()), 3),
                              n_vertices_low_baseline=int(bad.sum()), median_vertex_min_baseline=float(np.median(vbase)), vertex_max_abs_beta=float(vmax.max()),
                              vertex_max_abs_p50=float(np.median(vmax)), runs_with_most_extreme_records="; ".join(f"{k[0]}/{k[1]}:{int(v)}" for k, v in by_run.items())))
    pd.DataFrame(ext_rows).to_csv(out / "extreme_psc_top_records.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(out / "extreme_psc_report.csv", index=False)
    print(pd.DataFrame(summ_rows).to_string(index=False))
    print(f"robust variant excludes {int(bad_union.sum())} of {int(mask.sum())} vertices (baseline < {BASELINE_FLOOR} in some run of some subject)")
    agg_rob = Aggregator(mask, vertex_keep=~bad_union)
    Hrob = {s: h["H"] for s, h in load_halves(work, "clean_verified_robust").items()}
    Wr, Ar = [], []
    for res in RESOLUTIONS:
        r = similarity_from_halves(Hrob, agg_rob, res); w, a = tables_from_r(r, TASKS, "clean_verified_robust_psc", res); Wr.append(w); Ar.append(a)
        np.savez_compressed(out / "matrices" / f"similarity_matrices_clean_verified_robust_psc_{res}.npz", r=r, tasks=np.array(TASKS))
    Wr = pd.concat(Wr); Ar = pd.concat(Ar)
    _, _, task_r, fam_r = summarise(Wr, Ar)
    task = pd.concat([task, task_r], ignore_index=True); fam = pd.concat([fam, fam_r], ignore_index=True)
    task.to_csv(out / "summary_task_level.csv", index=False); fam.to_csv(out / "summary_family_level.csv", index=False)
    pd.concat([W, Wr]).to_csv(out / "within_similarity_long.csv", index=False); pd.concat([A, Ar]).to_csv(out / "across_similarity_long.csv", index=False)

    # variant comparison table (family level, all resolutions)
    piv = fam.pivot_table(index=["spatial_resolution", "task_family"], columns="variant", values=["within_mean", "across_mean"])
    piv.to_csv(out / "variant_comparison_family_level.csv")
    json.dump(dict(base_seed=BASE_SEED, n_splits=N_SPLITS, split_seed="SeedSequence([BASE_SEED, split, subject_index, task_index]) (task_index+100 for family-pooled)",
                   session_scale_seed=BASE_SEED + 7, spatial_shuffle_seed=BASE_SEED + 11, task_shuffle_seed=BASE_SEED + 13, task_shuffle_permutation=tperm.tolist(),
                   extreme_psc_record_p95_threshold=EXTREME_PSC, baseline_floor=BASELINE_FLOOR, n_vertices_excluded_robust=int(bad_union.sum()), n_valid_vertices=int(mask.sum()),
                   affine_max_abs_dr=dmax, subjects=SUBJECTS, tasks=TASKS, families=FAMILIES, variants=variants + ["clean_verified_robust_psc"],
                   network_aggregation="unweighted mean of the 360 region means (production convention)", z_score="per subject map, across units, ddof=0, +1e-8"),
              open(out / "analysis_config.json", "w"), indent=2)
    print(fam[fam.variant == "clean_verified"].to_string(index=False))
    print(f"done in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["halves", "similarity"], required=True)
    ap.add_argument("--subj")
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--output_root", default=None)
    ap.add_argument("--clean_dir", default=None, help="default: <dataset>/derivatives/glm-clean-delay")
    ap.add_argument("--legacy_pickle_dir", default=None, help="optional; the invalid published pickles (comparison only)")
    ap.add_argument("--work_dir", default=None, help="default: <output_root>/fig8_clean/halves")
    ap.add_argument("--out_dir", default=None, help="default: <output_root>/similarity_analysis/regenerated_clean")
    ap.add_argument("--labels", default=str(paths.REPO_ROOT / "results/similarity_analysis/regenerated_clean/run_labels_verified.csv"))
    args = ap.parse_args()
    global CLEAN, PKL_DIR
    paths.configure(args.dataset_root, args.output_root)
    paths.install_legacy_guard(f"similarity_analysis_clean_{args.stage}_{args.subj or 'all'}")
    CLEAN = paths.require_not_legacy(Path(args.clean_dir or paths.clean_delay_dir()), "--clean_dir")
    PKL_DIR = Path(args.legacy_pickle_dir) if args.legacy_pickle_dir else None
    args.work_dir = str(paths.require_not_legacy(args.work_dir or paths.output_dir("fig8_clean", "halves"), "--work_dir"))
    args.out_dir = str(paths.require_not_legacy(args.out_dir or paths.output_dir("similarity_analysis", "regenerated_clean"), "--out_dir"))
    paths.require_not_legacy(args.labels, "--labels")
    if args.stage == "halves":
        stage_halves(args)
    else:
        stage_similarity(args)


if __name__ == "__main__":
    main()
