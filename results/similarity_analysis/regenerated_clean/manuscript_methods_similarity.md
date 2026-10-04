# Proposed Methods text — spatial-pattern similarity (Section 4.5 / Figure 8)

Do not paste into the manuscript until the corresponding Results and caption are accepted. Values come from the regenerated clean analysis (`clean_verified` variant).

---

Delay-period trial-level beta maps were taken from the OLS GLM (all trials fitted; percent-signal-change scaling). Only correct trials were retained for the similarity analysis, using the original GLM column index (`trial_index` in the unfiltered design matrix) so that filtering never remapped a trial onto a different beta column. Each physical delay beta was stored once (pair-rule and single-frame annotations were kept as metadata and were not used to duplicate vectors).

Run-to-condition labels were verified by matching each acquired run’s stimulus sequence to the PsychoPy block files loaded by the session runner. Four runs in sub-06 were relabelled to the matched block file; all other analysed runs kept the lookup label after the block file confirmed it. Behaviour-inferred labels were not treated as ground truth.

For each participant and each of the 12 task variants (DMS; 1-Back location, category, identity; CTXDM LCI and CIL; InterDMS location/category/identity × ABAB/ABBA), the retained delay maps were randomly divided into two non-overlapping halves, averaged within each half, z-scored across spatial units, and compared with Pearson correlation. Within-subject similarity was the diagonal of the resulting 5 × 5 matrix (each participant’s half A versus that participant’s half B). Across-subject similarity used the 10 unique unordered participant pairs, symmetrised as \(0.5\,[r(i_A,j_B)+r(j_A,i_B)]\). This procedure was repeated for 50 random split-half iterations with a recorded seed sequence. The 50 iterations were used only to stabilise the estimate and to report Monte Carlo variability of the random partition; they are not independent biological samples and were not treated as inferential replicates.

The same trial partitions were analysed at three spatial resolutions: 58,396 valid cortical vertices; 360 Glasser parcels (mean of valid vertices); and 12 Cole–Anticevic networks (unweighted mean of the parcel means assigned to each network). All unique retained delay trials were used at every resolution (no trial cap). Task-level estimates were first averaged across the 50 splits for each participant or pair, then summarised across the five participants or ten pairs. Family-level summaries (DMS, 1-Back, InterDMS, CTXDM) are means of those task-level estimates, not a separate analysis that pools trials across variants. A family-pooled estimand was computed only as a supplementary contrast.

No paired test across split iterations was performed. No inferential *p*-values are reported for this descriptive analysis.
