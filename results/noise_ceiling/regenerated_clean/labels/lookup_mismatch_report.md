# Run-to-condition label verification (block-file matching)

Runs checked: 564 (all acquired runs with an events file, excluding `__dup` files).
Exact block-file match: 558/564. Lookup label confirmed by block file: 554/564.
Relabelled from authoritative block file: 4. Unresolved: 6.

Authoritative source: `task_stimuli/data/multfs/updated_cond_file/blockfiles/*.csv`, loaded by `task_stimuli/src/sessions/ses-multfs.py`; the runner derives feature and ABAB/ABBA order from the block-file name.

## Runs where sources disagree or matching is not exact

| subject | session | run | bold_task | in_glm_outputs | lookup_label | runner_design_label | blockfile_label | blockfile_exact | blockfile_trial_overlap | behaviour_label | behaviour_agreement | lookup_behaviour_agreement | final_label | final_label_behaviour_agreement | behaviour_inconsistent | action | justification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-02 | ses-000 | run-01 | 1backcat | False | LOOKUP_FAIL:LookupError |  |  | False | 0.0 |  |  |  | LOOKUP_FAIL:LookupError |  | False | unresolved_keep_lookup | no block file matched; lookup label kept, run flagged |
| sub-02 | ses-000 | run-01 | 1backloc | False | LOOKUP_FAIL:LookupError |  |  | False | 0.0 |  |  |  | LOOKUP_FAIL:LookupError |  | False | unresolved_keep_lookup | no block file matched; lookup label kept, run flagged |
| sub-02 | ses-000 | run-01 | ctxlco | False | LOOKUP_FAIL:LookupError |  |  | False | 0.0 |  |  |  | LOOKUP_FAIL:LookupError |  | False | unresolved_keep_lookup | no block file matched; lookup label kept, run flagged |
| sub-02 | ses-000 | run-01 | dmsloc | False | LOOKUP_FAIL:LookupError |  |  | False | 0.0 |  |  |  | LOOKUP_FAIL:LookupError |  | False | unresolved_keep_lookup | no block file matched; lookup label kept, run flagged |
| sub-03 | ses-001 | run-01 | interdms | True | interdms_ctg_ABBA | interdms_ctg_ABBA | interdms_ctg_ABBA | True | 1.0 | interdms_obj_ABBA | 0.733 | 0.6 | interdms_ctg_ABBA | 0.6 | False | keep | block-file match confirms lookup label |
| sub-03 | ses-005 | run-02 | interdms | True | interdms_ctg_ABBA | interdms_ctg_ABBA | interdms_ctg_ABBA | True | 1.0 | interdms_ctg_ABAB | 1.0 | 0.625 | interdms_ctg_ABBA | 0.625 | True | keep | block-file match confirms lookup label |
| sub-03 | ses-006 | run-01 | interdms | True | interdms_loc_ABBA | interdms_loc_ABBA | interdms_loc_ABBA | True | 1.0 | interdms_loc_ABAB | 1.0 | 0.562 | interdms_loc_ABBA | 0.562 | True | keep | block-file match confirms lookup label |
| sub-03 | ses-012 | run-01 | 1back | True | nback_obj | nback_obj | nback_obj | True | 1.0 | nback_loc | 1.0 | 0.581 | nback_obj | 0.581 | True | keep | block-file match confirms lookup label |
| sub-03 | ses-013 | run-01 | interdms | True | interdms_ctg_ABAB | interdms_ctg_ABAB | interdms_ctg_ABAB | True | 1.0 | interdms_ctg_ABBA | 1.0 | 0.625 | interdms_ctg_ABAB | 0.625 | True | keep | block-file match confirms lookup label |
| sub-03 | ses-014 | run-02 | interdms | True | interdms_loc_ABBA | interdms_loc_ABBA | interdms_loc_ABBA | True | 1.0 | interdms_loc_ABAB | 0.938 | 0.5 | interdms_loc_ABBA | 0.5 | True | keep | block-file match confirms lookup label |
| sub-05 | ses-001 | run-01 | interdms | True | interdms_ctg_ABBA | interdms_ctg_ABBA | interdms_ctg_ABBA | True | 1.0 | interdms_obj_ABBA | 0.969 | 0.781 | interdms_ctg_ABBA | 0.781 | False | keep | block-file match confirms lookup label |
| sub-05 | ses-002 | run-03 | interdms | True | interdms_ctg_ABAB | interdms_ctg_ABAB | interdms_ctg_ABAB | True | 1.0 | interdms_loc_ABAB |  |  | interdms_ctg_ABAB |  | False | keep | block-file match confirms lookup label |
| sub-05 | ses-003 | run-01 | interdms | True | interdms_obj_ABBA | interdms_obj_ABBA | interdms_obj_ABBA | True | 1.0 | interdms_ctg_ABBA | 0.781 | 0.719 | interdms_obj_ABBA | 0.719 | False | keep | block-file match confirms lookup label |
| sub-05 | ses-004 | run-01 | interdms | True | interdms_obj_ABAB | interdms_obj_ABAB | interdms_obj_ABAB | True | 1.0 | interdms_loc_ABAB | 0.938 | 0.594 | interdms_obj_ABAB | 0.594 | True | keep | block-file match confirms lookup label |
| sub-05 | ses-005 | run-01 | 1back | True | nback_ctg | nback_ctg | nback_ctg | True | 1.0 | nback_loc | 0.889 | 0.667 | nback_ctg | 0.667 | True | keep | block-file match confirms lookup label |
| sub-05 | ses-005 | run-03 | interdms | True | interdms_loc_ABAB | interdms_loc_ABAB | interdms_loc_ABAB | True | 1.0 | interdms_ctg_ABAB | 0.781 | 0.719 | interdms_loc_ABAB | 0.719 | False | keep | block-file match confirms lookup label |
| sub-05 | ses-009 | run-01 | interdms | True | interdms_obj_ABAB | interdms_obj_ABAB | interdms_obj_ABAB | True | 1.0 | interdms_ctg_ABAB | 1.0 | 0.781 | interdms_obj_ABAB | 0.781 | True | keep | block-file match confirms lookup label |
| sub-05 | ses-011 | run-01 | 1back | True | nback_ctg | nback_ctg | nback_ctg | True | 1.0 | nback_loc | 0.978 | 0.622 | nback_ctg | 0.622 | True | keep | block-file match confirms lookup label |
| sub-05 | ses-012 | run-01 | dms | False | dms_loc | dms_loc |  | False | 0.0 |  |  |  | dms_loc |  | False | unresolved_keep_lookup | no block file matched; lookup label kept, run flagged |
| sub-05 | ses-012 | run-01 | interdms | False | interdms_loc_ABAB | interdms_loc_ABAB |  | False | 0.0 | interdms_loc_ABAB |  |  | interdms_loc_ABAB |  | False | unresolved_keep_lookup | no block file matched; lookup label kept, run flagged |
| sub-05 | ses-014 | run-01 | interdms | True | interdms_loc_ABAB | interdms_loc_ABAB | interdms_loc_ABAB | True | 1.0 | interdms_loc_ABBA | 0.969 | 0.781 | interdms_loc_ABAB | 0.781 | False | keep | block-file match confirms lookup label |
| sub-06 | ses-007 | run-01 | 1back | True | nback_loc | nback_loc | nback_obj | True | 1.0 | nback_obj | 1.0 | 0.6 | nback_obj | 1.0 | False | relabel_verified | exact stimulus-sequence match to a different block file (authoritative) |
| sub-06 | ses-007 | run-01 | interdms | True | interdms_loc_ABAB | interdms_loc_ABAB | interdms_loc_ABBA | True | 1.0 | interdms_loc_ABBA | 1.0 | 0.562 | interdms_loc_ABBA | 1.0 | False | relabel_verified | exact stimulus-sequence match to a different block file (authoritative) |
| sub-06 | ses-007 | run-02 | interdms | True | interdms_ctg_ABBA | interdms_ctg_ABBA | interdms_loc_ABBA | True | 1.0 | interdms_loc_ABBA | 1.0 | 0.531 | interdms_loc_ABBA | 1.0 | False | relabel_verified | exact stimulus-sequence match to a different block file (authoritative) |
| sub-06 | ses-007 | run-03 | interdms | True | interdms_loc_ABBA | interdms_loc_ABBA | interdms_loc_ABAB | True | 1.0 | interdms_loc_ABAB | 1.0 | 0.548 | interdms_loc_ABAB | 1.0 | False | relabel_verified | exact stimulus-sequence match to a different block file (authoritative) |
| sub-06 | ses-014 | run-01 | interdms | True | interdms_loc_ABAB | interdms_loc_ABAB | interdms_loc_ABAB | True | 1.0 | interdms_loc_ABBA | 0.969 | 0.719 | interdms_loc_ABAB | 0.719 | True | keep | block-file match confirms lookup label |

## Action summary

| action | n_runs |
|---|---|
| keep | 554 |
| unresolved_keep_lookup | 6 |
| relabel_verified | 4 |

## Per-subject relabels

| subject | n_relabelled |
|---|---|
| sub-06 | 4 |
