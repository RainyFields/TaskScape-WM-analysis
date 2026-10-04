#!/usr/bin/env python3
"""Build the clean, de-duplicated, provenance-tracked delay-period beta dataset for one subject.

Replaces the task-pooled use of `grouped_betas/task_relevant_only/*_task_condition_betas.pkl`
(mis-indexed + duplicated; see AGENTS.md gotchas).

For every run in data/encoding_delay/trial_level_betas/<subj>/ses-*/func/:
  * the design CSV row i corresponds to GLM beta column i (glm_analysis.py writes trial_index = i);
    this is asserted, and NO re-indexing is ever done after filtering;
  * every design row of phase 'delay' becomes one provenance record; the beta vector is stored once,
    only for retained records (is_correct is not False);
  * the legacy pair/single-frame annotations are kept as metadata (`condition_pair_key`,
    `condition_single_key`, `legacy_multiplicity`) but never duplicate the vector;
  * run -> task-variant labels come from run_labels_verified.csv (block-file verified); the lookup label,
    the behaviour label and the resolution status are kept alongside.

Validation (job fails on violation): design/GLM column alignment, vector-by-vector equality with the
source GLM column, delay-only phase, uniqueness of (subject, session, run, trial_index, phase), no
NaN/Inf, identical CIFTI brain-model axis across runs.

Outputs (in --out_dir)
  <subj>_delay_betas.h5      datasets: betas (n_retained x 64984 float32), record_id (int), attrs
  <subj>_provenance.csv      one row per delay design row (retained and excluded), record_id for retained
  <subj>_validation.json     counts + assertion results
  <subj>_vertex_qc.npz       per-vertex max |beta| (retained records), min PSC baseline over runs, etc.

Usage
  python scripts/build_clean_delay_betas.py --subj sub-01 \
      --labels results/similarity_analysis/regenerated_clean/run_labels_verified.csv \
      --out_dir data/encoding_delay/clean_delay_betas
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, nibabel as nib

PROJECT = Path("/project/def-pbellec/xuan/fmri_dataset_project")
sys.path.insert(0, str(PROJECT / "scripts"))
import task_condition_identification as tci

BETA_ROOT = PROJECT / "data/encoding_delay/trial_level_betas"
FMRI_ROOT = PROJECT / "data/glasser_resampled"
N_GRAYORDINATES = 64984

TASK_FAMILY = {"dms_loc": "DMS", "nback_loc": "1-Back", "nback_ctg": "1-Back", "nback_obj": "1-Back",
               "ctxdm_col": "CTXDM", "ctxdm_lco": "CTXDM",
               **{f"interdms_{f}_{o}": "InterDMS" for f in ("loc", "obj", "ctg") for o in ("ABAB", "ABBA")}}


def is_negative(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v) is False
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"false", "b", "0"}


def axis_signature(dtseries_path: Path) -> str:
    img = nib.load(str(dtseries_path))
    ax = img.header.get_axis(1)
    h = hashlib.sha1()
    h.update(np.asarray(ax.vertex).astype(np.int64).tobytes())
    h.update("|".join(ax.name).encode())
    h.update(json.dumps(dict(ax.nvertices), sort_keys=True).encode())
    return h.hexdigest()


def find_dtseries(subj, ses, bold_task, run_int):
    cands = list((FMRI_ROOT / subj / ses).glob(f"{subj}_{ses}_task-{bold_task}_run-*_space-Glasser64k_bold_sm4.dtseries.nii"))
    for c in cands:
        r = c.name.split("_run-")[1].split("_")[0]
        if int(r) == run_int:
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", required=True)
    ap.add_argument("--labels", required=True, help="run_labels_verified.csv from verify_run_labels_from_blockfiles.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--beta_root", default=str(BETA_ROOT))
    args = ap.parse_args()
    subj = args.subj
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    beta_root = Path(args.beta_root)

    labels = pd.read_csv(args.labels)
    labels = labels[labels.subject == subj].set_index(["session", "bold_task", "run"])

    records, vectors = [], []
    run_summaries = []
    axis_sigs = {}
    vertex_max_abs = np.zeros(N_GRAYORDINATES, np.float32)
    vertex_min_baseline = np.full(N_GRAYORDINATES, np.inf, np.float32)
    n_glm_columns_total = 0
    t0 = time.time()

    for h5_path in sorted((beta_root / subj).glob("ses-*/func/*_betas.h5")):
        subj_id, ses, bold_task, run, base = tci.parse_run_tokens(h5_path)
        run_int = int(run.split("-")[1])
        design_csv = h5_path.with_name(f"{base}_design.csv")
        if not design_csv.exists():
            raise FileNotFoundError(design_csv)
        lab = labels.loc[(ses, bold_task, f"run-{run_int:02d}")]
        if isinstance(lab, pd.DataFrame):
            raise ValueError(f"multiple label rows for {subj} {ses} run-{run_int:02d}")

        df = pd.read_csv(design_csv)
        df = tci._canonicalize_design_columns(df)
        with h5py.File(h5_path, "r") as f:
            B = f["betas"][()]                                   # (P, K) float32
            baseline = f["psc_baseline_mean"][()].reshape(-1)    # (P,)
            attrs = dict(f.attrs)
        P, K = B.shape
        # ---- alignment assertions (design row i <-> GLM task column i)
        if P != N_GRAYORDINATES:
            raise AssertionError(f"{h5_path.name}: {P} grayordinates != {N_GRAYORDINATES}")
        if K != len(df):
            raise AssertionError(f"{h5_path.name}: betas columns {K} != design rows {len(df)}")
        if not np.array_equal(df["trial_index"].values, np.arange(K)):
            raise AssertionError(f"{design_csv.name}: trial_index is not 0..K-1")
        if attrs.get("regressor_level") != "trial":
            raise AssertionError(f"{h5_path.name}: regressor_level={attrs.get('regressor_level')}")
        if int(attrs["task_col_end"]) - int(attrs["task_col_start"]) != K:
            raise AssertionError(f"{h5_path.name}: task column span != K")
        n_glm_columns_total += K

        # ---- CIFTI axis signature of the source time series (header only)
        ts = find_dtseries(subj, ses, bold_task, run_int)
        if ts is None:
            raise FileNotFoundError(f"no dtseries for {subj} {ses} {bold_task} run-{run_int}")
        axis_sigs[h5_path.name] = axis_signature(ts)
        vertex_min_baseline = np.minimum(vertex_min_baseline, baseline.astype(np.float32))

        final_label = str(lab["final_label"]); family = TASK_FAMILY[final_label]
        pairs = tci.determine_task_type(final_label)
        feats = tci.determine_feature_type(final_label)
        df["type_l"] = df["type"].astype(str).str.lower()
        df["stim_order"] = pd.to_numeric(df["stim_order"], errors="coerce").astype("Int64")
        n_enc = int(df.type_l.str.contains("encod").sum())
        n_del = int(df.type_l.str.contains("delay").sum())
        n_del_retained = n_del_excluded = 0
        legacy_total = 0

        for _, trial_df in df.groupby("trialNumber"):
            enc = trial_df[trial_df.type_l.str.contains("encod")]
            dly = trial_df[trial_df.type_l.str.contains("delay")]
            enc_by_k = {int(r.stim_order): r for r in enc.itertuples() if pd.notna(r.stim_order)}
            for r in dly.itertuples():
                if pd.isna(r.stim_order):
                    continue
                k = int(r.stim_order)
                # legacy annotations that pointed at THIS delay beta
                single_key = pair_key = None; mult = 0
                if (k, k) in pairs and k in enc_by_k:
                    single_key = tci.build_condition_key(final_label, df.loc[enc_by_k[k].Index], df.loc[enc_by_k[k].Index], feats, condition_type="single_frame"); mult += 1
                if (k - 1, k) in pairs and (k - 1) in enc_by_k and k in enc_by_k:
                    pair_key = tci.build_condition_key(final_label, df.loc[enc_by_k[k - 1].Index], df.loc[enc_by_k[k].Index], feats); mult += 1
                legacy_total += mult
                retained = not is_negative(r.is_correct)
                col = int(r.trial_index)
                rec = dict(
                    subject=subj, session=ses, run=f"run-{run_int:02d}", bold_task=bold_task,
                    task_variant=final_label, task_family=family,
                    label_lookup=lab["lookup_label"], label_blockfile=lab["blockfile_label"], label_behaviour=lab["behaviour_label"],
                    label_action=lab["action"], label_resolved=bool(lab["resolved"]),
                    trial_number=int(r.trialNumber), stim_order=k, phase="delay",
                    trial_index=col, glm_column_full=int(attrs["task_col_start"]) + col,
                    is_correct=str(r.is_correct), retained=retained,
                    location=getattr(r, "location", None), object=getattr(r, "object", None), category=getattr(r, "category", None),
                    onset_time=getattr(r, "onset_time", None), offset_time=getattr(r, "offset_time", None),
                    condition_single_key=single_key, condition_pair_key=pair_key, legacy_multiplicity=mult,
                    source_beta_file=str(h5_path.relative_to(PROJECT)), source_design_file=str(design_csv.relative_to(PROJECT)),
                    source_events_file=f"data/reformated_behavior/{subj}/{ses}/func/{subj}_{ses}_task-{bold_task}_run-{run_int:02d}_events.tsv",
                )
                if retained:
                    vec = B[:, col]
                    if not np.all(np.isfinite(vec)):
                        raise AssertionError(f"non-finite beta in {h5_path.name} column {col}")
                    rec["record_id"] = len(vectors)
                    vectors.append(vec.astype(np.float32))
                    vertex_max_abs = np.maximum(vertex_max_abs, np.abs(vec))
                    n_del_retained += 1
                else:
                    rec["record_id"] = -1
                    n_del_excluded += 1
                records.append(rec)
        run_summaries.append(dict(subject=subj, session=ses, run=f"run-{run_int:02d}", task_variant=final_label, task_family=family,
                                  glm_columns=K, encoding_rows=n_enc, delay_rows=n_del, delay_retained=n_del_retained,
                                  delay_excluded_incorrect=n_del_excluded, legacy_records=legacy_total,
                                  duplicates_removed=legacy_total - n_del_retained if True else None))
    print(f"[{subj}] scanned {len(run_summaries)} runs in {time.time()-t0:.0f}s; records={len(records)} retained={len(vectors)}")

    prov = pd.DataFrame(records)
    # ---- uniqueness assertion
    # NOTE: run numbers restart for every BOLD task within a session (e.g. ses-001 has ctxdm run-01 AND
    # interdms run-01), so the physical key must include the BOLD task / source file.
    key = prov[["subject", "session", "bold_task", "run", "trial_index", "phase"]].astype(str).agg("|".join, axis=1)
    if key.duplicated().any():
        raise AssertionError(f"duplicate physical records: {key[key.duplicated()].tolist()[:5]}")
    if (prov.phase != "delay").any():
        raise AssertionError("non-delay phase in delay dataset")
    betas = np.stack(vectors, 0)                                  # (n, P)
    if not np.all(np.isfinite(betas)):
        raise AssertionError("NaN/Inf in stored betas")
    # physically identical vectors (should not happen: different GLM columns)
    hashes = pd.Series([hashlib.sha1(betas[i].tobytes()).hexdigest() for i in range(betas.shape[0])])
    n_identical_vectors = int(hashes.duplicated().sum())
    if len(set(axis_sigs.values())) != 1:
        raise AssertionError(f"CIFTI axis differs across runs: {set(axis_sigs.values())}")

    # ---- write H5
    h5_out = out / f"{subj}_delay_betas.h5"
    with h5py.File(h5_out, "w") as f:
        f.create_dataset("betas", data=betas, chunks=(min(64, betas.shape[0]), N_GRAYORDINATES))
        f.create_dataset("record_id", data=np.arange(betas.shape[0]))
        f.attrs["subject"] = subj
        f.attrs["n_records"] = betas.shape[0]
        f.attrs["n_grayordinates"] = N_GRAYORDINATES
        f.attrs["cifti_axis_sha1"] = next(iter(axis_sigs.values()))
        f.attrs["phase"] = "delay"
        f.attrs["units"] = "PSC (glm_analysis.py: 100*(Y-mean)/mean), OLS trial-level betas, 4 mm FWHM smoothed"
        f.attrs["labels_file"] = str(Path(args.labels).resolve())
        f.attrs["created_by"] = "scripts/build_clean_delay_betas.py"
    prov.to_csv(out / f"{subj}_provenance.csv", index=False)

    # ---- vector-by-vector re-verification against the source GLM files (re-open everything)
    n_checked = n_ok = 0
    ret = prov[prov.retained].copy()
    with h5py.File(h5_out, "r") as fo:
        stored = fo["betas"]
        for src, grp in ret.groupby("source_beta_file"):
            with h5py.File(PROJECT / src, "r") as fs:
                Bsrc = fs["betas"][()]
            dsrc = pd.read_csv(PROJECT / grp.source_design_file.iloc[0]); dsrc.columns = dsrc.columns.str.strip()
            for r in grp.itertuples():
                v = stored[int(r.record_id)]
                ok = np.array_equal(v, Bsrc[:, int(r.trial_index)]) and ("delay" in str(dsrc.iloc[int(r.trial_index)]["type"]).lower())
                n_checked += 1; n_ok += int(ok)
    if n_ok != n_checked:
        raise AssertionError(f"vector verification failed: {n_checked - n_ok} of {n_checked} mismatched")

    # ---- summaries
    per_task = ret.groupby("task_variant").agg(n_retained=("record_id", "size"), n_runs=("run", lambda s: s.groupby([ret.loc[s.index, 'session'], s]).ngroups))
    per_task["legacy_records"] = ret.groupby("task_variant").legacy_multiplicity.sum()
    per_task["duplicates_removed"] = per_task.legacy_records - per_task.n_retained
    per_task["excluded_incorrect"] = prov[~prov.retained].groupby("task_variant").size().reindex(per_task.index).fillna(0).astype(int)
    per_task["n_glm_columns"] = pd.DataFrame(run_summaries).groupby("task_variant").glm_columns.sum().reindex(per_task.index)
    per_task["n_delay_rows"] = pd.DataFrame(run_summaries).groupby("task_variant").delay_rows.sum().reindex(per_task.index)
    per_task = per_task.reset_index(); per_task.insert(0, "subject", subj)
    per_task.to_csv(out / f"{subj}_counts_per_task.csv", index=False)
    pd.DataFrame(run_summaries).to_csv(out / f"{subj}_counts_per_run.csv", index=False)

    mask = np.load(PROJECT / "data/valid_vertex_mask.npy")
    rec_max = np.abs(betas[:, mask]).max(1)
    validation = dict(
        subject=subj, n_runs=len(run_summaries), n_glm_columns_total=n_glm_columns_total,
        n_delay_rows=int(len(prov)), n_delay_retained=int(len(ret)), n_delay_excluded_incorrect=int((~prov.retained).sum()),
        n_legacy_records=int(ret.legacy_multiplicity.sum()), n_duplicates_removed=int(ret.legacy_multiplicity.sum() - len(ret)),
        n_task_variants=int(ret.task_variant.nunique()), task_variants=sorted(ret.task_variant.unique()),
        n_runs_relabelled=int(prov.drop_duplicates(["session", "run"]).label_action.str.startswith("relabel").sum()),
        n_runs_unresolved=int((~prov.drop_duplicates(["session", "run"]).label_resolved).sum()),
        vector_verification=dict(checked=n_checked, ok=n_ok, wrong_column=n_checked - n_ok),
        n_encoding_in_delay_dataset=int((prov.phase != "delay").sum()),
        n_duplicate_keys=int(key.duplicated().sum()), n_physically_identical_vectors=n_identical_vectors,
        n_nan_inf=int((~np.isfinite(betas)).sum()), cifti_axis_sha1=next(iter(axis_sigs.values())),
        n_masked_vertices_const=int((betas[:, mask].std(0) == 0).sum()),
        masked_abs_beta_percentiles={p: float(np.percentile(np.abs(betas[:, mask]), p)) for p in (50, 99, 99.9, 99.99, 100)},
        n_records_with_abs_beta_gt_50=int((rec_max > 50).sum()), n_records_with_abs_beta_gt_100=int((rec_max > 100).sum()),
        elapsed_s=round(time.time() - t0),
    )
    (out / f"{subj}_validation.json").write_text(json.dumps(validation, indent=2, default=str) + "\n")
    np.savez(out / f"{subj}_vertex_qc.npz", vertex_max_abs=vertex_max_abs, vertex_min_baseline=vertex_min_baseline,
             record_max_abs_masked=rec_max, record_argmax_vertex_masked=np.where(mask)[0][np.abs(betas[:, mask]).argmax(1)])
    print(json.dumps({k: v for k, v in validation.items() if k not in ("task_variants",)}, indent=1, default=str))
    print(per_task.to_string(index=False))
    print(f"[{subj}] wrote {h5_out} ({betas.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
