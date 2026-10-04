# Clean delay-beta dataset validation: **PASS**

Atlas CIFTI brain-model axis sha1: `5da42bc458`; all subject datasets must match it.
Glasser parcels with valid vertices: 360/360 (min 22 vertices). Networks: 12 (min 588 vertices).

## Failures

none

## Per subject

| subject | n_runs | n_glm_columns | n_delay_rows | retained | excluded_incorrect | legacy_records | duplicates_removed | runs_relabelled | runs_unresolved | vectors_checked | wrong_column | const_masked_vertices | abs_beta_p99_9 | abs_beta_max | records_absbeta_gt100 | cifti_axis_sha1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 112 | 12736 | 6368 | 6034 | 334 | 10260 | 4226 | 0 | 0 | 6034 | 0 | 0 | 172.6 | 465114.9 | 3641 | 5da42bc458 |
| sub-02 | 112 | 12736 | 6368 | 6227 | 141 | 10646 | 4419 | 0 | 0 | 6227 | 0 | 0 | 117.03 | 255870.2 | 3166 | 5da42bc458 |
| sub-03 | 112 | 12736 | 6368 | 6098 | 270 | 10388 | 4290 | 0 | 0 | 6098 | 0 | 0 | 173.24 | 4797.1 | 3129 | 5da42bc458 |
| sub-05 | 110 | 12544 | 6272 | 5368 | 904 | 8960 | 3592 | 0 | 0 | 5368 | 0 | 0 | 240.65 | 281105.8 | 2840 | 5da42bc458 |
| sub-06 | 112 | 12736 | 6368 | 6036 | 332 | 10264 | 4228 | 2 | 0 | 6036 | 0 | 0 | 158.13 | 2145.0 | 3174 | 5da42bc458 |

## Per subject x task variant

| subject | task_variant | n_runs | n_glm_columns | n_delay_rows | retained_delay_betas | excluded_incorrect | legacy_records_pair_plus_single | duplicates_removed |
|---|---|---|---|---|---|---|---|---|
| sub-01 | ctxdm_col | 16 | 1920 | 960 | 866 | 94 | 1412 | 546 |
| sub-01 | ctxdm_lco | 16 | 1920 | 960 | 896 | 64 | 1472 | 576 |
| sub-01 | dms_loc | 16 | 1024 | 512 | 505 | 7 | 754 | 249 |
| sub-01 | interdms_ctg_ABAB | 5 | 640 | 320 | 303 | 17 | 526 | 223 |
| sub-01 | interdms_ctg_ABBA | 5 | 640 | 320 | 290 | 30 | 500 | 210 |
| sub-01 | interdms_loc_ABAB | 14 | 1792 | 896 | 861 | 35 | 1498 | 637 |
| sub-01 | interdms_loc_ABBA | 14 | 1792 | 896 | 864 | 32 | 1504 | 640 |
| sub-01 | interdms_obj_ABAB | 5 | 640 | 320 | 311 | 9 | 542 | 231 |
| sub-01 | interdms_obj_ABBA | 5 | 640 | 320 | 310 | 10 | 540 | 230 |
| sub-01 | nback_ctg | 2 | 216 | 108 | 97 | 11 | 176 | 79 |
| sub-01 | nback_loc | 12 | 1296 | 648 | 628 | 20 | 1148 | 520 |
| sub-01 | nback_obj | 2 | 216 | 108 | 103 | 5 | 188 | 85 |
| sub-02 | ctxdm_col | 16 | 1920 | 960 | 876 | 84 | 1432 | 556 |
| sub-02 | ctxdm_lco | 16 | 1920 | 960 | 916 | 44 | 1512 | 596 |
| sub-02 | dms_loc | 16 | 1024 | 512 | 512 | 0 | 768 | 256 |
| sub-02 | interdms_ctg_ABAB | 5 | 640 | 320 | 320 | 0 | 560 | 240 |
| sub-02 | interdms_ctg_ABBA | 5 | 640 | 320 | 319 | 1 | 558 | 239 |
| sub-02 | interdms_loc_ABAB | 14 | 1792 | 896 | 894 | 2 | 1564 | 670 |
| sub-02 | interdms_loc_ABBA | 14 | 1792 | 896 | 888 | 8 | 1552 | 664 |
| sub-02 | interdms_obj_ABAB | 5 | 640 | 320 | 319 | 1 | 558 | 239 |
| sub-02 | interdms_obj_ABBA | 5 | 640 | 320 | 320 | 0 | 560 | 240 |
| sub-02 | nback_ctg | 2 | 216 | 108 | 108 | 0 | 198 | 90 |
| sub-02 | nback_loc | 12 | 1296 | 648 | 647 | 1 | 1186 | 539 |
| sub-02 | nback_obj | 2 | 216 | 108 | 108 | 0 | 198 | 90 |
| sub-03 | ctxdm_col | 16 | 1920 | 960 | 868 | 92 | 1416 | 548 |
| sub-03 | ctxdm_lco | 16 | 1920 | 960 | 908 | 52 | 1496 | 588 |
| sub-03 | dms_loc | 16 | 1024 | 512 | 511 | 1 | 766 | 255 |
| sub-03 | interdms_ctg_ABAB | 5 | 640 | 320 | 305 | 15 | 530 | 225 |
| sub-03 | interdms_ctg_ABBA | 5 | 640 | 320 | 289 | 31 | 498 | 209 |
| sub-03 | interdms_loc_ABAB | 14 | 1792 | 896 | 894 | 2 | 1564 | 670 |
| sub-03 | interdms_loc_ABBA | 14 | 1792 | 896 | 863 | 33 | 1502 | 639 |
| sub-03 | interdms_obj_ABAB | 5 | 640 | 320 | 309 | 11 | 538 | 229 |
| sub-03 | interdms_obj_ABBA | 5 | 640 | 320 | 317 | 3 | 554 | 237 |
| sub-03 | nback_ctg | 2 | 216 | 108 | 108 | 0 | 198 | 90 |
| sub-03 | nback_loc | 12 | 1296 | 648 | 644 | 4 | 1180 | 536 |
| sub-03 | nback_obj | 2 | 216 | 108 | 82 | 26 | 146 | 64 |
| sub-05 | ctxdm_col | 16 | 1920 | 960 | 859 | 101 | 1398 | 539 |
| sub-05 | ctxdm_lco | 16 | 1920 | 960 | 888 | 72 | 1456 | 568 |
| sub-05 | dms_loc | 15 | 960 | 480 | 455 | 25 | 670 | 215 |
| sub-05 | interdms_ctg_ABAB | 5 | 640 | 320 | 277 | 43 | 474 | 197 |
| sub-05 | interdms_ctg_ABBA | 5 | 640 | 320 | 293 | 27 | 506 | 213 |
| sub-05 | interdms_loc_ABAB | 13 | 1664 | 832 | 715 | 117 | 1222 | 507 |
| sub-05 | interdms_loc_ABBA | 14 | 1792 | 896 | 837 | 59 | 1450 | 613 |
| sub-05 | interdms_obj_ABAB | 5 | 640 | 320 | 296 | 24 | 512 | 216 |
| sub-05 | interdms_obj_ABBA | 5 | 640 | 320 | 293 | 27 | 506 | 213 |
| sub-05 | nback_ctg | 2 | 216 | 108 | 46 | 62 | 74 | 28 |
| sub-05 | nback_loc | 12 | 1296 | 648 | 341 | 307 | 574 | 233 |
| sub-05 | nback_obj | 2 | 216 | 108 | 68 | 40 | 118 | 50 |
| sub-06 | ctxdm_col | 16 | 1920 | 960 | 873 | 87 | 1426 | 553 |
| sub-06 | ctxdm_lco | 16 | 1920 | 960 | 910 | 50 | 1500 | 590 |
| sub-06 | dms_loc | 16 | 1024 | 512 | 512 | 0 | 768 | 256 |
| sub-06 | interdms_ctg_ABAB | 5 | 640 | 320 | 316 | 4 | 552 | 236 |
| sub-06 | interdms_ctg_ABBA | 4 | 512 | 256 | 249 | 7 | 434 | 185 |
| sub-06 | interdms_loc_ABAB | 14 | 1792 | 896 | 819 | 77 | 1414 | 595 |
| sub-06 | interdms_loc_ABBA | 15 | 1920 | 960 | 907 | 53 | 1574 | 667 |
| sub-06 | interdms_obj_ABAB | 5 | 640 | 320 | 319 | 1 | 558 | 239 |
| sub-06 | interdms_obj_ABBA | 5 | 640 | 320 | 320 | 0 | 560 | 240 |
| sub-06 | nback_ctg | 2 | 216 | 108 | 106 | 2 | 194 | 88 |
| sub-06 | nback_loc | 11 | 1188 | 594 | 543 | 51 | 987 | 444 |
| sub-06 | nback_obj | 3 | 324 | 162 | 162 | 0 | 297 | 135 |
