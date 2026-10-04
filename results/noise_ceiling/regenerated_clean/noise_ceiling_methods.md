# Methods — delay-period noise ceiling

Delay-period reliability was estimated from the validated clean single-trial beta set
(`data/encoding_delay/clean_delay_betas/`; validation status PASS). Each record is one physical
delay-period GLM column. Column identity was checked against the unfiltered design matrix and the
H5 regressor name: wrong-column mappings, encoding columns, duplicated physical betas, and
non-finite values were all zero. Original trial indices were retained. Task-variant labels are the
block-file labels. Four sub-06 runs whose stimulus sequences matched a different block file than
the lookup table were relabelled from that match. No acquired run that entered the GLM remained unresolved.

Twelve task variants were included: DMS; 1-Back location, category, and identity; CTXDM CIL
(`ctxdm_col`) and LCI (`ctxdm_lco`); and InterDMS location, category, and identity, each with
ABAB and ABBA order. A delay beta was retained when that item’s `is_correct` value was not false.
Blank correctness values, which occur on items that did not require a response, were kept. Only
correct, unique delay-period betas were used.

For every subject, task variant, and split iteration, the retained betas were divided at random
into two non-overlapping halves of equal size. When the trial count was odd, the extra trial
alternated between halves across iterations (half A on even iterations). The random generator for
variant *i* and iteration *b* was NumPy’s `default_rng` seeded with `[20260926 + 1000i, b]`.
Every variant contributed data to both halves. The trial identifiers in each half were saved.

Within each half, betas were averaged within task variant. Those means were aggregated to 360
Glasser regions by averaging valid vertices inside each parcel (58,396 of 64,984 vertices; medial
wall and low-variance vertices excluded), and to 12 Cole–Anticevic networks by the unweighted mean
of the region values in each network. At every region or network, separately, the two 12-element
vectors were correlated with Pearson’s *r* across the 12 task variants. This is not a spatial
correlation across regions or networks. The split was repeated 50 times.

Raw correlations were clipped to [−1+10⁻⁶, 1−10⁻⁶] only for the Fisher *z* transform, averaged in
*z* across the 50 iterations, and transformed back to a mean split-half correlation *r̄*. If
*r̄* ≤ 0, the noise ceiling was set to 0. If *r̄* > 0,

NC = 2*r̄* / (1 + *r̄*).

The estimate is therefore bounded in [0, 1]. The Fisher-averaged correlation, the fraction of
iterations with *r* < 0, and the count of units set to zero were saved. Arithmetic-mean and
median aggregation of the raw correlations, and the previous per-iteration Spearman–Brown
correction, were computed only as diagnostics. The per-iteration correction is not the reported
estimate: applied to negative correlations it is unbounded near *r* = −1.

The reported group summary is the unweighted mean of the five subject-level means. A pooled
summary across spatial units is reported separately. Region and network units were never combined.
Vertex-level maps were not used for the primary result.
