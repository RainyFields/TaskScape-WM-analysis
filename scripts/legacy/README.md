# scripts/legacy: archived scripts (2026-10-07)

Scripts moved here (with `git mv`, history preserved) because their role is finished, their inputs are legacy-only or
invalid, or a migrated script replaces them. **Nothing was deleted.** They are not maintained, still reference the frozen
legacy project `/project/def-pbellec/xuan/fmri_dataset_project` by absolute path, and must not be run against it.
Per-file review and rationale: `results/restructure/path_migration_review.tsv` (dispositions) and
`results/restructure/path_migration_table.tsv` (every legacy-path line).

Sub-folders: `nilearn/` (volumetric nilearn cross-check GLM and its one-off evaluations, archived as a group because
they import each other), `sanity_check/` (beta-contrast and diagnostic one-offs), `slurm/` (shell wrappers hard-wired
to legacy roots), `logs/` (saved process/status dumps), `figure_making/` (superseded figure one-off).

| archived path | former path | disposition | replacement or role |
|---|---|---|---|
| `scripts/legacy/accuracy_plot.py` | `scripts/accuracy_plot.py` | superseded-by | scripts/figure_making/plot_behavioral_performance.py |
| `scripts/legacy/aggregate_subject_beta_activation_vertex.py` | `scripts/aggregate_subject_beta_activation_vertex.py` | superseded-by | scripts/export_beta_activation_session_maps.py |
| `scripts/legacy/check_beta_consistency.py` | `scripts/check_beta_consistency.py` | obsolete | One-off sub-01/ses-001 interdms run-to-run beta correlation diagnostic with hard-wired paths. |
| `scripts/legacy/fd_distribution.py` | `scripts/fd_distribution.py` | superseded-by | scripts/figure_making/plot_fd_distribution.py |
| `scripts/legacy/figure_making/investigate_r2_one_voxels.py` | `scripts/figure_making/investigate_r2_one_voxels.py` | obsolete | Flat-line (R2=1) voxel investigation that motivated the valid vertex mask; mask is now shipped in the atlas dir. |
| `scripts/legacy/glm_analysis_condition_based.py` | `scripts/glm_analysis_condition_based.py` | superseded-by | scripts/glm_analysis.py |
| `scripts/legacy/glm_analysis_regularized.py` | `scripts/glm_analysis_regularized.py` | superseded-by | scripts/glm_analysis.py |
| `scripts/legacy/glm_per_task_concat.py` | `scripts/glm_per_task_concat.py` | superseded-by | scripts/glm_analysis.py |
| `scripts/legacy/logs/check_beta_consistency_output.txt` | `scripts/check_beta_consistency_output.txt` | obsolete | Saved stdout of the one-off check_beta_consistency.py diagnostic. |
| `scripts/legacy/logs/events_diag_shell_log.txt` | `scripts/events_diag_shell_log.txt` | obsolete | 23k-line Cursor shell environment dump from an events-path debugging session. |
| `scripts/legacy/logs/glm_ps_children.txt` | `scripts/glm_ps_children.txt` | obsolete | Process-status dump from a GLM test run. |
| `scripts/legacy/logs/glm_ps_status.txt` | `scripts/glm_ps_status.txt` | obsolete | Process-status dump. |
| `scripts/legacy/logs/glm_python_ps_status.txt` | `scripts/glm_python_ps_status.txt` | obsolete | Process-status dump. |
| `scripts/legacy/logs/glm_python_ps_status2.txt` | `scripts/glm_python_ps_status2.txt` | obsolete | Process-status dump. |
| `scripts/legacy/logs/ols_pid_check.txt` | `scripts/ols_pid_check.txt` | obsolete | Process-status dump of an OLS test run. |
| `scripts/legacy/logs/ols_test_ps2.txt` | `scripts/ols_test_ps2.txt` | obsolete | Process-status dump of an OLS test run. |
| `scripts/legacy/logs/run1_adjusted_output.txt` | `scripts/run1_adjusted_output.txt` | obsolete | 23k-line Cursor shell environment dump of a single GLM test run. |
| `scripts/legacy/nilearn/comprehensive_glm_assessment.py` | `scripts/comprehensive_glm_assessment.py` | obsolete | Nilearn vs custom GLM one-off assessment; validation closed in results/final_nilearn_assessment.md. |
| `scripts/legacy/nilearn/create_nilearn_result.py` | `scripts/create_nilearn_result.py` | obsolete | Hard-coded sub-01 ses-001 nilearn GLM run used only for the closed nilearn validation. |
| `scripts/legacy/nilearn/evaluate_nilearn_glm.py` | `scripts/evaluate_nilearn_glm.py` | obsolete | Evaluates nilearn_data betas that are not part of the dataset; nilearn validation closed (results/final_nilearn_assessment.md). |
| `scripts/legacy/nilearn/glm_analysis_nilearn.py` | `scripts/glm_analysis_nilearn.py` | superseded-by | scripts/glm_analysis.py |
| `scripts/legacy/nilearn/glm_evaluation_nilearn.py` | `scripts/sanity_check/glm_evaluation_nilearn.py` | obsolete | Voxel-level evaluation of nilearn_data betas; validation closed. |
| `scripts/legacy/nilearn/nilearn_quality_assessment.py` | `scripts/nilearn_quality_assessment.py` | obsolete | Hard-coded nilearn volumetric quality check from the closed nilearn validation. |
| `scripts/legacy/nilearn/quick_glm_evaluation.py` | `scripts/quick_glm_evaluation.py` | obsolete | Quick nilearn vs original H5 comparison for one run; nilearn validation closed. |
| `scripts/legacy/nilearn/quick_nilearn_evaluation.py` | `scripts/quick_nilearn_evaluation.py` | obsolete | Hard-coded single-run nilearn evaluation; validation closed. |
| `scripts/legacy/nilearn/run_glm_nilearn_permutations.sh` | `scripts/run_glm_nilearn_permutations.sh` | obsolete | SLURM wrapper for the nilearn GLM permutations (def-bashivan, legacy roots). |
| `scripts/legacy/nilearn/run_nilearn_analysis.py` | `scripts/run_nilearn_analysis.py` | obsolete | Runner for the nilearn pipeline; validation closed, outputs not in dataset. |
| `scripts/legacy/nilearn/run_single_session.py` | `scripts/run_single_session.py` | obsolete | Hard-coded nilearn single-session test runner. |
| `scripts/legacy/nilearn/test_single_run.py` | `scripts/test_single_run.py` | obsolete | Hard-coded nilearn single-run debug test. |
| `scripts/legacy/noise_ceiling.py` | `scripts/noise_ceiling.py` | superseded-by | scripts/noise_ceiling_clean/noise_ceiling_clean.py (branch taskscape-migration-nc) |
| `scripts/legacy/noise_ceiling_final_version.py` | `scripts/noise_ceiling_final_version.py` | superseded-by | scripts/noise_ceiling_clean/noise_ceiling_clean.py (branch taskscape-migration-nc) |
| `scripts/legacy/noise_ceiling_tradition.py` | `scripts/noise_ceiling_tradition.py` | superseded-by | scripts/noise_ceiling_clean/noise_ceiling_clean.py (branch taskscape-migration-nc) |
| `scripts/legacy/sanity_all_processed.py` | `scripts/sanity_all_processed.py` | obsolete | Existence check of beta files under the dead data/betas root. |
| `scripts/legacy/sanity_check/beta_contrast_vis.py` | `scripts/sanity_check/beta_contrast_vis.py` | obsolete | Early contrast plot over the dead data/betas root. |
| `scripts/legacy/sanity_check/beta_contrast_vis_condition_level.py` | `scripts/sanity_check/beta_contrast_vis_condition_level.py` | obsolete | Stats for condition-level (encoding/delay) GLM outputs that are not produced for the dataset. |
| `scripts/legacy/sanity_check/beta_contrast_vis_per_task_concat.py` | `scripts/sanity_check/beta_contrast_vis_per_task_concat.py` | obsolete | Contrast for the unused per-task concatenated GLM. |
| `scripts/legacy/sanity_check/beta_contrast_vis_roi_condition_level.py` | `scripts/sanity_check/beta_contrast_vis_roi_condition_level.py` | obsolete | ROI stats for condition-level GLM outputs not produced for the dataset. |
| `scripts/legacy/sanity_check/beta_contrast_vis_roi_trial_level.py` | `scripts/sanity_check/beta_contrast_vis_roi_trial_level.py` | obsolete | Needs XtX_inv/sigma2 datasets that the current glm_analysis.py H5 does not store. |
| `scripts/legacy/sanity_check/beta_contrast_vis_task_aggregated_condition_level.py` | `scripts/sanity_check/beta_contrast_vis_task_aggregated_condition_level.py` | obsolete | Aggregates condition-level ROI stats that are not produced for the dataset. |
| `scripts/legacy/sanity_check/beta_contrast_vis_task_aggregated_trial_level.py` | `scripts/sanity_check/beta_contrast_vis_task_aggregated_trial_level.py` | obsolete | Needs XtX_inv/sigma2 datasets absent from the current glm_analysis.py H5. |
| `scripts/legacy/sanity_check/beta_contrast_vis_trial_level.py` | `scripts/sanity_check/beta_contrast_vis_trial_level.py` | obsolete | Needs XtX_inv/sigma2 datasets absent from the current glm_analysis.py H5. |
| `scripts/legacy/sanity_check/debug_parcel_loading.py` | `scripts/sanity_check/debug_parcel_loading.py` | obsolete | One-off parcel loading debug on a single run. |
| `scripts/legacy/sanity_check/debug_synthetic_vs_glm_design.py` | `scripts/sanity_check/debug_synthetic_vs_glm_design.py` | obsolete | One-off debug of synthetic vs GLM design mismatch. |
| `scripts/legacy/sanity_check/glm_evaluation_noise_ceiling.py` | `scripts/sanity_check/glm_evaluation_noise_ceiling.py` | obsolete | Two-line stub pointing at task_condition_identification.py. |
| `scripts/legacy/sanity_check/raw_data_snr_analysis.py` | `scripts/sanity_check/raw_data_snr_analysis.py` | obsolete | MNI volumetric tSNR diagnostic from the nilearn debugging phase; not used by the paper. |
| `scripts/legacy/sanity_check/vertex_level_analysis.py` | `scripts/sanity_check/vertex_level_analysis.py` | obsolete | Zero-variance vertex diagnostic; superseded in substance by the shipped valid vertex mask. |
| `scripts/legacy/sanity_check/visualize_raw_timeseries.py` | `scripts/sanity_check/visualize_raw_timeseries.py` | obsolete | Raw time-series loading diagnostic from the early debugging phase. |
| `scripts/legacy/slurm/backup_data_to_dropbox.sh` | `scripts/backup_data_to_dropbox.sh` | obsolete | rclone backup of the legacy data/ tree; dataset is now the archive. |
| `scripts/legacy/slurm/cifti_resample.sh` | `scripts/cifti_resample.sh` | obsolete | fsLR91k to Glasser64k resampling; output is shipped as derivatives/taskscape-surface by the dataset build. |
| `scripts/legacy/slurm/datalad_sync.sh` | `scripts/datalad_sync.sh` | obsolete | Datalad install/get of the legacy fMRIPrep RIA; dataset inputs are already materialised. |
| `scripts/legacy/slurm/run_glm_permutations.sh` | `scripts/run_glm_permutations.sh` | obsolete | Legacy SLURM wrapper over include_types permutations with pre-encoding_delay output layout. |
| `scripts/legacy/slurm/smooth_glasser_sm4.sh` | `scripts/smooth_glasser_sm4.sh` | obsolete | 4 mm smoothing wrapper; desc-sm4 dtseries are shipped by the dataset build. |
| `scripts/legacy/slurm/smooth_glasser_sm4_array.sh` | `scripts/smooth_glasser_sm4_array.sh` | obsolete | SLURM array for smoothing; output shipped by the dataset build. |
| `scripts/legacy/slurm/submit_aggregate_subject_r2.sh` | `scripts/submit_aggregate_subject_r2.sh` | obsolete | Trivial wrapper with legacy cd/out roots; rewrite against config.paths when aggregate_subject_r2_vertex_maps.py migrates. |
| `scripts/legacy/slurm/submit_build_bids_release.sh` | `scripts/submit_build_bids_release.sh` | obsolete | SLURM array for the legacy BIDS release build (done by migration repo). |
| `scripts/legacy/slurm/submit_glm_clean_regenerate.sh` | `scripts/submit_glm_clean_regenerate.sh` | obsolete | One-time regeneration array after the sm4/PSC cleanup; done. |
| `scripts/legacy/slurm/submit_regenerate_grouped_betas.sh` | `scripts/submit_regenerate_grouped_betas.sh` | obsolete | Runs the known-broken task_condition_identification.py to rebuild invalid pickles. |
| `scripts/legacy/slurm/submit_smooth_sm4_array.sh` | `scripts/submit_smooth_sm4_array.sh` | obsolete | Two-line sbatch wrapper for the obsolete smoothing array. |
| `scripts/legacy/slurm/submit_tar_release.sh` | `scripts/submit_tar_release.sh` | obsolete | Tar-per-subject copy of the scratch release to def-bashivan; release distribution superseded by TaskScape-WM. |
| `scripts/legacy/slurm/sync_betas_to_shared.sh` | `scripts/sync_betas_to_shared.sh` | obsolete | rsync of legacy trial_level_betas to def-bashivan shared. |
| `scripts/legacy/subject_within_vs_cross_similarity.py` | `scripts/subject_within_vs_cross_similarity.py` | superseded-by | scripts/similarity_analysis_clean.py |
| `scripts/legacy/task_condition_repetition_check_raw_data.py` | `scripts/task_condition_repetition_check_raw_data.py` | obsolete | One-off ctxdm condition count over raw sub-03 events. |

## Kept in place although reviewed as superseded/obsolete/historical

| path | disposition | why it stays |
|---|---|---|
| `scripts/accuracy_sanity.py` | `obsolete` | dataset provenance: produced sourcedata/behavior-corrected (interdms ABAB correction) |
| `scripts/build_bids_release.py` | `obsolete` | previous release builder; historical reference for the release export gate |
| `scripts/clean_encoding_delay_betas.py` | `obsolete` | dataset provenance: executed cleanup recorded in derivatives/glm/provenance |
| `scripts/figure_making/plot_fig8_style_matched_v2.py` | `superseded-by` | unsure (reviewer): may be the manuscript styling variant |
| `scripts/figure_making/plot_fig8_uncluttered_v3.py` | `superseded-by` | unsure (reviewer): may be the manuscript styling variant |
| `scripts/figure_making/plot_per_task_within_across_subj_similarity.py` | `superseded-by` | imported by similarity_analysis_clean.py (legacy variant grouping) |
| `scripts/generate_valid_vertex_mask.py` | `obsolete` | dataset provenance: produced atlas/valid_vertex_mask.npy; named in utils.py error text |
| `scripts/glm_output.log` | `historical` | saved log/report; left in place |
| `scripts/nilearn_quality.log` | `historical` | saved log/report; left in place |
| `scripts/reformat_behavior.py` | `obsolete` | dataset provenance: produced sourcedata/behavior-reformatted |
| `scripts/remove_padded_run_duplicates.py` | `obsolete` | dataset provenance: run-label de-duplication of legacy outputs |
| `scripts/run1_surface_diagnosis_report.txt` | `historical` | saved log/report; left in place |
| `scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc/sub-03_all_sessions_runs_r2_full_region_avg_summary.txt` | `historical` | saved log/report; left in place |
| `scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc/sub-03_all_sessions_runs_r2_task_adj_region_avg_summary.txt` | `historical` | saved log/report; left in place |
| `scripts/sanity_check/glm_fitting_check_results/customized_glm/encoding_delay_ssmoothing_psc/sub-03_all_sessions_runs_r2_task_region_avg_summary.txt` | `historical` | saved log/report; left in place |
| `scripts/submit_glm_array.sh` | `obsolete` | unsure (reviewer): only production SLURM array for glm_analysis.py; flags preserved in repro/submit_fidelity.sh |
| `scripts/task_col_uniqueness_checks.txt` | `historical` | saved log/report; left in place |
| `scripts/task_condition_identification.py` | `superseded-by` | library import of build_clean_delay_betas.py and verify_run_labels_from_blockfiles.py |
| `scripts/update_dms_reformatted_mods.py` | `obsolete` | dataset provenance: edited reformatted DMS events |

## Not moved: blocked and active scripts

The 15 `migrate-blocked-invalid-input` scripts (they read the invalid grouped pickles or results derived from them)
stay in `scripts/` unchanged until they are re-pointed to `derivatives/glm-clean-delay`; the `migrate` set is being
migrated to `config/paths.py` (see `results/restructure/repro_report.md` and the path-migration tables).
