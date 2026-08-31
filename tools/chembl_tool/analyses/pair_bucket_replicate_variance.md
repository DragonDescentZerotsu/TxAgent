# Pair-bucket replicate variance

> **Regenerated 2026-08-29:** These statistics use the rebuilt normalized-v7
> Stage-03/04 artifacts under source-value cleaning v5, measurement routing v4,
> and measurement-resolution v2. Final pruning is downstream of this requested
> Stage-04 cohort and does not alter these statistics.

## Question and cohort

For every normalized-v7 Stage-04 pair bucket, retain a SMILES only when it has at
least five eligible records, then retain a bucket only when it has at least five
such SMILES. Calculate within-SMILES and across-SMILES variation using only those
qualifying SMILES. BBB, Bioavailability, and Skin are in scope; ClinTox has no
corresponding promoted normalized-v7 Stage-04 pair-bucket artifact.

The final cohort contains 648 buckets and 332,826 records across 19,610
SMILES-within-bucket units:

| Task | Buckets | Qualifying SMILES-bucket units | Records used |
|---|---:|---:|---:|
| BBB | 68 | 4,903 | 143,057 |
| Bioavailability | 209 | 5,491 | 69,165 |
| Skin | 371 | 9,216 | 120,604 |
| **Total** | **648** | **19,610** | **332,826** |

## Statistics

For qualifying SMILES `i`, let `s_i^2` be the sample variance of its records and
`mean_i` its record mean. Each bucket reports:

- mean within-SMILES variance: `mean_i(s_i^2)`;
- mean within-SMILES SD: `mean_i(s_i)`;
- pooled within-SMILES variance: `sum_i((n_i - 1) s_i^2) / sum_i(n_i - 1)`;
- across-SMILES variance: the sample variance of `mean_i`;
- across-SMILES SD: the square root of the across-SMILES variance.

The mean within-SMILES SD is not the square root of the mean within-SMILES
variance: averaging and taking a square root do not commute.

Both native and geometry-aware statistics are saved. Positive multiplicative
measurements use log10 geometry; already-log measurements remain on their
canonical log scale; binary and ordinal measurements use canonical category
ranks. Native percentage and logit scales remain separate.

Across the 320 continuous dex-geometry buckets, the median bucket statistics are:

| Statistic | Median |
|---|---:|
| Mean within-SMILES variance | 0.184 dex^2 |
| Mean within-SMILES SD | 0.351 dex, approximately 2.24-fold |
| Across-SMILES variance | 0.548 dex^2 |
| Across-SMILES SD | 0.740 dex, approximately 5.50-fold |

Task medians for continuous dex-geometry buckets:

| Task | Within variance | Within SD | Across variance | Across SD |
|---|---:|---:|---:|---:|
| BBB | 0.074 | 0.238 | 0.366 | 0.599 |
| Bioavailability | 0.092 | 0.244 | 0.467 | 0.683 |
| Skin | 0.395 | 0.525 | 0.828 | 0.910 |

For the 13 percentage buckets, the median mean within-SMILES SD is 9.32
percentage points and the median across-SMILES SD is 15.09 percentage points.

## Unit and distribution validation

All 648 selected buckets pass the structural unit checks:

- exactly one `canonical_unit_text` across all eligible records in the bucket;
- exact Stage-03 versus Stage-04 unit agreement for every record;
- exact agreement between the unit and the third element of `pair_bucket_key`;
- finite analysis values and valid categorical ID-to-rank mappings.

This proves canonical unit-string consistency, not that every upstream scale,
endpoint, or reference-scope assignment is scientifically correct.

The distribution audit applied two deliberately high-specificity screens to both
the requested qualifying cohort and all eligible rows in each selected bucket:

- a supported gap of at least two dex, with observations on both sides;
- records beyond ten interquartile ranges from the bucket's central range.

Every triggered bucket was then checked against its raw measurement, raw unit,
support text, and final canonical value. Twelve qualifying cohorts triggered a
screen. Seven were source-supported extremes or genuine between-molecule gaps.
Five contained likely data-contract problems:

| Task | Finding | Status | Affects requested variance? |
|---|---|---|---|
| Bioavailability | Source `109.4` was inferred as a fraction and became canonical `10,940%` | Confirmed scale error | Yes |
| Bioavailability | Literal `602%` systemic availability sits in an absolute-reference bucket | Probable reference-scope error | Yes |
| Bioavailability | A 5 mg-dose AUC cluster uses physically implausible `mg*h/mL` | Probable unit-extraction error | Yes |
| Skin | A `81.33%` epidermal-retention row sits under dermal absorption | Confirmed endpoint mismatch | Yes |
| Skin | A skin-retention row uses physically implausible `44.39 g/cm^2` | Probable unit-extraction error | Yes |

One additional Bioavailability record is malformed but belongs to a SMILES with
fewer than five records, so it does not affect the requested variance. The source
value `1.7 ng*h/mL*10^3` was mapped with scale `0.001` rather than `1000`, moving
its log10 value from the literal 3.23 to -2.77.

No source record or canonical artifact was repaired by the analysis script. All
reported statistics use the rebuilt stored Stage-04 values; the issue table is
an audit annotation, and the user deferred those scientific-data repairs.

## Why the review happened

The requested work had two layers:

1. Calculate replicate and between-molecule variation for the requested cohort.
2. "Validate one more time that all the records are on the same unit and nothing
   is super out of distribution."

The second clause prompted the structural unit checks and numerical tail screens.
The semantic inspection was not an independent broad audit: it was a follow-up on
only the buckets triggered by those screens. Numerical extremeness alone was not
treated as an error. A row was called problematic only when its raw measurement,
unit, support text, and canonical tuple showed a concrete scale, endpoint, or
reference-scope concern.

An active Bioavailability rebuild replaced Stage 03 during the first analysis
pass. The read was discarded rather than mixing versions. After that build
finished, only the deterministic Stage-04 sidecar was regenerated from the new
Stage 03, and the full analysis was rerun. The final report verifies that each
Stage-04 input hash matches its current Stage-03 artifact and that a second run
produces identical output hashes.

## Artifacts and reproduction

- [Per-bucket TSV](../../../outputs/chembl_tool/pair_bucket_replication_variance_audit/pair_bucket_variance.tsv)
- [Per-bucket Parquet](../../../outputs/chembl_tool/pair_bucket_replication_variance_audit/pair_bucket_variance.parquet)
- [Reviewed distribution flags](../../../outputs/chembl_tool/pair_bucket_replication_variance_audit/reviewed_distribution_flags.tsv)
- [Flagged record outside the variance cohort](../../../outputs/chembl_tool/pair_bucket_replication_variance_audit/flagged_records.tsv)
- [Input/output manifest](../../../outputs/chembl_tool/pair_bucket_replication_variance_audit/manifest.json)
- [Reproduction script](../../../outputs/chembl_tool/pair_bucket_replication_variance_audit/run_audit.py)

The regenerated run used
`/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/bin/python` on
`login01`.
