# Limitations — noise ceiling

The correlation uses 12 task variants. Pearson’s *r* on 12 points is noisy, which is why many
regions have a negative mean split-half correlation and a noise ceiling of 0. A zero means that
the average split-half correlation was not positive. It is not a demonstration that the region
contains no task-related variance of any kind.

Correctness is item-level, matching the validated delay-beta release: a delay row is removed
only when that row is marked incorrect. Requiring every response in the trial to be correct
removed all 1-Back category trials in sub-05 (46 item-level betas, 0 fully correct trials), so
that sensitivity could not be estimated for sub-05. In the other four subjects the trial-level
rule changed subject means only modestly.

Four sub-06 runs were relabelled because their stimulus sequences matched a different block file
than the study-design lookup. That match is exact and is treated as authoritative. Using the
lookup labels instead changed sub-06’s region mean from 0.53 to 0.54 and the network mean from
0.65 to 0.67. No GLM run was left unresolved. Behaviour-inferred labels were not used as ground
truth; they disagree with the block files on some runs that the block files match exactly.

A minority of delay maps contain implausible PSC values, usually at vertices with a near-zero
baseline (11, 1, 56, 59, and 35 records in sub-01, 02, 03, 05, and 06). The primary estimate
keeps them. Dropping records with a 95th-percentile absolute PSC above 100 increases reliability,
especially for sub-05 networks (from 0 to 0.33). Which of those vertices are acquisition
artefacts versus baseline-normalised spikes is not fully resolved from the beta files alone.

Network means give every Glasser region equal weight, including small parcels. They are not
vertex-count-weighted. Vertex-level noise ceilings were not estimated; a few vertices reach
absolute PSC values of 10⁴–10⁵, so a vertex-wise correlation would need an explicit outlier rule
before it could be reported.

The 50 splits are Monte Carlo partitions of the same trials. Their average is a reliability
estimate. They are not 50 independent experiments, and they were not entered into a paired *t*
test.

Reliability is high enough in sub-02 and sub-06, in a subset of regions, to expect some
shared task-variant structure. It is low or absent in other subjects and in many regions and
networks. A model comparison should use the subject- and region-specific maps, not the former
group means of 0.66 and 0.71.
