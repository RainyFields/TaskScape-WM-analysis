# Limitations and unresolved points — spatial-pattern similarity

1. **Task-label shuffle.** Deranging one participant’s 12 task labels barely changed network-level across-subject similarity for 1-Back, InterDMS and CTXDM. Shared network structure in those families should not be described as task-specific. DMS is the exception (across-subject *r* rose from ~0 to 0.25 when that participant’s DMS map was replaced by another task), consistent with DMS having a distinctive high-reliability pattern.

2. **Pair heterogeneity.** Several network-level pair correlations are large and positive while others are near zero or negative (DMS pair range −0.55 to 0.88). A near-zero family mean is not evidence that all pairs are dissimilar, and it is not evidence of idiosyncratic strategies.

3. **CTXDM and some InterDMS / 1-Back variants** have low or even negative within-subject similarity in individual participants (for example vertex-level 1-Back location in one participant). Those families should not be called “highly reliable” as a group.

4. **LCI / CIL naming.** Internal codes `ctxdm_lco` and `ctxdm_col` are mapped to “CTXDM LCI” and “CTXDM CIL” from letter order (location–category–identity vs category–identity–location). Authors should confirm this against the task description before publication.

5. **Identity vs object.** Manuscript-facing outputs use “identity”; on-disk variant keys remain `*_obj_*` / `nback_obj` to match GLM filenames.

6. **Behaviour vs block file.** Fourteen runs had been flagged because behaviour agreed better with an alternative condition. Stimulus-sequence matching to the PsychoPy block files confirmed the lookup label for those runs (and for all other GLM runs except four in sub-06, which were relabelled). Behaviour is a noisy proxy for the feature rule (match/non-match can be scored under the wrong feature). Sensitivity variants that drop or relabel behaviour-inconsistent runs do not change the headline pattern, but they drop entire variants for some participants (sub-03 1-Back identity and sub-05 1-Back category become empty).

7. **Clone control is not a pure within-subject copy.** The clone used split iteration *i*+1 of the same participant, so the two partitions can share trials. Clone-pair *r* is therefore an upper-ish bound, not a clean match to the diagonal.

8. **Session-level scalar control** is not a pure affine transform of the finished maps: different halves contain different session mixes, so Pearson *r* can move slightly. Invariance was confirmed by applying \(3.7x+12\) to one participant’s already-averaged maps (max |Δ*r*| = 7.45 × 10⁻⁷).

9. **Extreme PSC.** Vertices with near-zero (or slightly negative) PSC baselines produce |β| of 10⁴–10⁵. They are concentrated at a few grayordinates (for example vertex 14842 in sub-01). The primary analysis does not drop them. The first robust attempt used a baseline floor of 1000 and emptied Glasser parcel 42; the corrected robust variant uses a floor of 1.0 (truly undefined PSC) plus exclusion of whole-map blow-up trials (cortical |β| p95 > 100).

10. **Family-pooled vs family-of-tasks.** Pooling all InterDMS (or 1-Back / CTXDM) trials before the split-half raises both within- and across-subject *r* (more trials, mixed variants). That is a different estimand from the primary family mean of task-level estimates. February 2026 manuscript text used a family-pooled analysis without DMS; it is not interchangeable with Figure 8.

11. **No inferential test.** With five participants, a hierarchical or permutation test could be added later; it is not part of this descriptive revision.

12. **Unresolved acquisition rows.** Six events files (sub-02 ses-000 pilot tasks; two sub-05 ses-012 files without GLM outputs) had no matching block file. None of them enter the GLM-based analysis.
