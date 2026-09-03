# Measurement routing and final-tail pruning

## Current active build (2026-08-31)

The active implementation is now `assay_transfer_outlier_review.v1`. It reviews
only assay-transfer-eligible Stage-03 pair buckets with at least 20 finite
records and 16 distinct canonical parent SMILES. A row is flagged when its
leave-one-out distance passes either the 5-SD test or the 2-decade test under
the scientifically selected raw/log10 geometry. A verified log10 axis may use
the retained numerical approval gate only when it has no 5-SD tail.

Each prompt marks only the flagged rows as decision targets and shows up to five
same-bucket comparison rows. The approved Jinja template presents semantic
measurement, unit, support text, and assay context rather than internal routing
metadata. One schema-valid structured binary vote per target row is final.

The review does **not** delete records or rewrite measurements. A reviewed drop
sets `assay_transfer_eligible=false` with a versioned
`llm_review_drop:<reason_code>` reason. The source row and its pair-bucket
assignment remain in Stage 03 for provenance and non-transfer uses.

### Materialized review result

| Task | Flagged buckets | Target rows | Prompt chunks | Keep | Assay-transfer-ineligible | Wrong endpoint/quantity | Wrong unit/scale | Corrupt extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB | 21 | 79 | 21 | 72 | 7 | 0 | 7 | 0 |
| Bioavailability | 70 | 587 | 76 | 549 | 38 | 15 | 22 | 1 |
| Skin | 72 | 3,250 | 135 | 3,060 | 190 | 102 | 84 | 4 |
| **Total** | **163** | **3,916** | **232** | **3,681** | **235** | **117** | **113** | **5** |

The Bioavailability and Skin figures are the post-reference-v3 rebuild. All 228
of their reviewed drops remain physically present in the rebuilt Stage-03
Parquet, all are `assay_transfer_eligible=false`, and all carry an explicit
`llm_review_drop:<reason_code>`. Each Stage-03 manifest binds the exact review
manifest and decisions hashes. BBB was not rebuilt in this pass and retains its
previous seven exclusions.

The earlier estimate of 166 buckets came from a pre-rebuild snapshot. Rebuilding
Stage 02 with the refined scientific transform policy reduced that candidate
set. An intermediate census then found 138 buckets, but 32 reviewed rows were
already transfer-ineligible direct binary-vote records whose legacy
`bucket_eligible` alias was stale. Synchronizing that alias to the authoritative
`assay_transfer_eligible` field produced the final 130-bucket scope. Cached
answers were reused only when the complete rendered prompt hash was unchanged;
four changed prompts required new calls.

### Reference-semantics refresh

Bioavailability and Skin now load
`data_processing/reference_semantics_v3/reference_semantics.parquet`. Stage 02
was rebuilt before pruning, and its manifests hash the corresponding v3 mapping.

| Task | Mapping rows | New supplement rows | Explicitly resolved | Explicit unknown/failure |
|---|---:|---:|---:|---:|
| Bioavailability | 194,936 | 19,895 | 194,936 | 0 |
| Skin | 235,702 | 148,648 | 219,480 | 16,222 |

The Skin unknown/failure rows remain complete mapping assignments but fail
closed for assay transfer. They include 10,987 `api_failure` rows caused by the
second OpenAI credential returning HTTP 401; the remaining rows are structured
semantic validation failures. OpenRouter Luna validly resolved 111,333 Skin
rows. The mapping builder did not resubmit already attempted rows and did not
use the third OpenAI key.

### Runtime and budget receipt

This refresh ran on `epyc-1-5`. The OpenAI key-one remaining-budget epoch
charged 4,863,889 of 4,884,429 tokens and has 20,540 left. Key two returned
HTTP 401 for every attempted request; its conservative fail-closed ledger has
12,279 of 5,753,768 tokens left, but this is accounting rather than confirmed
provider usage.

The authorized OpenRouter Luna epoch covered the remaining Skin reference rows
and the Bioavailability/Skin pruning pass. It reports 10,770,047 input and
3,713,029 output tokens, plus 730,346 conservatively charged unreported tokens:
15,213,422 charged of 20,000,000, leaving 4,786,578. No reservations remain.

Four first-pass Skin pruning chunks returned a binary decision with an
incompatible reason code. Those outputs were rejected, the other 128 valid
Skin chunks remained cached, and only the four unresolved chunks were
resubmitted. Each target ultimately has exactly one accepted vote; there is no
ensemble or majority stage.

Per-task manifests, decisions, exact prompts/responses, and prompt-hash caches
are under
`outputs/chembl_tool/tasks/<task>/evidence_library/starling_normalized_v7/reviews/assay_transfer_outlier_v1/`.

### Retrieval-invariance audit

The transfer-eligibility result is fully materialized, but the frozen
retrieval-projection audit is not green. Relative to the pre-reference-v3
snapshot, the rebuilt unified dedup stage retains three fewer Bioavailability
representatives (`439,428` versus `439,431`) and nine more Skin representatives
(`553,337` versus `553,328`). The source/reference refresh changes the final
assay-transfer numeric tuple used by current v3 row deduplication, which can
change duplicate grouping even though source-visible fields are unchanged.

An attempted switch to the stable pre-transform tuple was tested and rejected
from this build because it produced a broader migration (46 Bioavailability and
28 Skin count differences). The active code remains
`starling_record_deduplication.v3`; the old frozen snapshot was not overwritten.
This is a bounded lineage issue to resolve separately, not evidence that the
228 pruning exclusions were deleted or misapplied.

The sections below document older routing and pruning iterations and are kept
as historical implementation lineage; their candidate counts and deletion
semantics are not the active Stage-03 contract.

## Historical outcome

Two pipeline defects are fixed in code:

1. A source value can use the deterministic `source_exact` route only when
   Python `float(value)` succeeds and the result is finite. Comma-bearing values
   therefore go to reviewed extraction.
2. Final pruning now screens every continuous Stage-05 pair bucket selected by
   `pair_bucket_records.bucket_eligible` when it has at least three finite
   values. It no longer substitutes the broader record-level
   `retrieval_eligible` flag for pair-bucket eligibility.
3. A verified log-normal distribution can now bypass LLM review when the raw
   values reject normality but the exact persisted log10 values do not. The
   gate never overrides a leave-one-out 5-SD tail.

The current pruning code contract is `final_endpoint_pruning.v6`. Earlier v3-v5
artifacts are preserved as development lineage but are not valid inputs to the
current downstream builder.

## Files and data flow

- `common/starling/normalization/source_value_cleaning.py` preserves decimal
  commas instead of repairing them before routing.
- `common/starling/measurement_routing.py` implements the finite `float(...)`
  gate for `source_exact`.
- `common/starling/build_measurement_resolution_mapping.py` reuses the frozen v1
  mapping and sends only new v2 rows to `gpt-5.4-mini`; each materialized row
  records its base-versus-delta provenance and model.
- `common/starling/final_endpoint_pruning.py` detects tails, selects up to five
  value-spread comparison rows, applies the audited log-normal approval gate,
  creates bounded review chunks, and treats one validated review as the final
  keep/drop decision. It cannot rewrite a value, unit, endpoint, SMILES, or
  assay field.
- `common/starling/llm_prompts/endpoint_pruning/v6.jinja` renders
  source measurement, source unit, canonical value/unit, earlier LLM extraction
  when present, support text, and semantic assay context as readable cards.
- `data/processing/evidence_library/versions/v7/pair_bucket_build.py` consumes
  the completed pruning manifest and marks reviewed rows assay-transfer
  ineligible while retaining them in the canonical Stage-3 records.

The former record-collapse and split-specific downstream stages were retired;
heldout-safe catalogs and indices are now built from completed libraries by
`data/processing/evidence_library/views.py`.

“Regeneration” here means deterministic rebuilding of downstream Parquet and
calibration artifacts after selected rows change. It is not LLM rewriting. The
only new extraction calls in this change were the 16 comma-bearing rows below.

## Measurement-routing result

| Task | Extraction candidates | Reused v1 rows | New v2 rows | Batched calls | Result |
|---|---:|---:|---:|---:|---|
| BBB | 109,166 | 109,153 | 13 | 2 | 13 `ok` |
| Bioavailability | 192,728 | 192,728 | 0 | 0 | Base mapping only |
| Skin | 243,377 | 243,374 | 3 | 1 | 3 `ok` |

Stage 00-05 was rebuilt for BBB, Bioavailability, and Skin from these v2
mappings. The rebuilt Stage-05 retained-record counts are 584,809, 467,296, and
807,759 respectively.

## Final-tail rule

For each finite record, v5 compares the value with the leave-one-out mean and
sample SD of the other records in its pair bucket.

- Every bucket with at least three finite values receives a
  `distance >= 5 SD` screen. Positive linear and declared log10/ln values use
  log10-equivalent geometry; other buckets use their native canonical scale.
- Buckets with supported log10-equivalent geometry additionally receive a
  `distance >= 2 decades` screen.
- If a bucket contains nonpositive values or has an ambiguous log base, it still
  receives the native-scale SD screen; only the decade trigger is unavailable.
- Buckets with one or two finite values are reported in the census but are not
  automatically pruned.

A numerical trigger starts review; it is never itself a drop reason. V6 sends
one Jinja-rendered semantic prompt and the first schema-valid response supplies
the final vote. Validation retries correct malformed output but do not add a
second vote. Tail prompts contain up to five distinct non-target rows spanning
the bucket's sorted value range; if fewer exist, all are shown. Complete
supported-gap buckets show every row once without duplicated context. Prompts
larger than 180,000 bytes are split by target rows, with up to five rows outside
the chunk retained as comparison context.

## Log10 policy audit

The transform policy is keyed by the complete axis
`(source, endpoint, canonical unit, reference scope)`, not by unit text alone.
This matters because `ratio` can be a positive multiplicative ratio or a
bounded/additive score depending on its endpoint semantics.

The current frozen policy files contain:

| Task | Exact axes | Log10 axes / rows | Raw axes / rows | Nonpositive rows on log10 axes |
|---|---:|---:|---:|---:|
| BBB | 8,628 | 7,133 / 51,043 | 1,495 / 10,531 | 141 |
| Bioavailability | 3,019 | 2,657 / 161,755 | 362 / 97,758 | 138 |
| Skin | 4,647 | 3,525 / 125,827 | 1,122 / 59,079 | 607 |

All log10 decisions belong to one of three semantic classes: positive physical
quantities, multiplicative ratios/folds, or positive rates/densities. Bounded
percentages and fractions, counts, scores/indices, signed axes, pH/temperature,
and already-log or base-ambiguous log units remain raw. Percent in a rate or
density such as `%ID/g` or `%/h` is a scientific log10 candidate, while a
bounded `%` outcome is not. The numerical normality result is retained only as
an audit flag and does not override this semantic mapping. The nonpositive rows listed above are retained as
evidence but are individually assay-transfer-ineligible and are not passed
through `log10`.

The raw-normality screen evaluated 75 scientifically loggable axes. It flags
49 as more raw-like—11 BBB axes covering 455 rows, 16 Bioavailability axes
covering 437 rows, and 22 Skin axes covering 554 rows—and 26 as more log10-like.
All 75 remain log10 under the scientific mapping. These are projected row
assignments from the current Stage-02 data; the persisted evidence library has
not yet been rebuilt.
Two additional one-row Skin axes (`log min^-1` and `log enhancement ratio`)
were corrected to raw because their units already declare a log scale.

The follow-up [75-axis audit](raw_normality_axis_audit.md) records the advisory
49/26 raw-versus-log10 normality comparison and retains the separate
raw-normal-versus-lognormal AIC/bootstrap analysis as a sensitivity check.

## Log10 statistical approval gate

The final-pruning gate applies only when all of the following hold:

- at least 20 records are present;
- every row declares `assay_transfer_transform_id=log10.v1` and a
  `log10(...)` canonical unit;
- every persisted transformed value equals `log10(pretransform value)` within
  `1e-9` absolute tolerance;
- a D'Agostino-Pearson K-squared test rejects raw-scale normality at
  `alpha=0.05`; logged-scale normality is not required; and
- no row has a leave-one-out distance of at least 5 SD.

Passing buckets remain unchanged and receive no LLM call. The manifest records
every approved bucket, its raw-scale test statistic and p-value, and the number
of review rows avoided. Failure or inapplicability is fail-closed: the existing
tail/gap review proceeds.

On the current preserved pre-three-stage Stage-05 snapshot, the offline census
is:

| Task | Numerically triggered buckets | Statistically approved | Remaining LLM buckets | Remaining target rows | Prompt chunks |
|---|---:|---:|---:|---:|---:|
| BBB | 473 | 25 | 448 | 532 | 448 |
| Bioavailability | 2,153 | 349 | 1,804 | 2,889 | 1,804 |
| Skin | 440 | 137 | 303 | 2,149 | 304 |
| **Total** | **3,066** | **511** | **2,555** | **5,570** | **2,556** |

The gate avoids 8,792 would-be review rows in total. None of the 511 approved
buckets contained a 5-SD tail.

## Pre-gate V6 offline prompt census

The complete v6 candidate set was rendered and validated without loading API
keys or making network calls:

| Task | Candidate buckets | Target rows | Single-pass prompt chunks | Largest prompt | Historical second calls avoided |
|---|---:|---:|---:|---:|---:|
| BBB | 253 | 292 | 253 | 8,891 bytes | 253 |
| Bioavailability | 2,153 | 6,476 | 2,158 | 179,833 bytes | 2,158 |
| Skin | 463 | 7,328 | 474 | 179,909 bytes | 474 |
| **Total** | **2,869** | **14,096** | **2,885** | **179,909 bytes** | **2,885** |

Every target occurs in exactly one prompt chunk, target and context IDs are
disjoint, comparison rows belong to the same bucket, no prompt exposes the
pair-bucket key or canonical record IDs, and all prompts remain below the
180,000-byte limit. No v6 LLM reviews or downstream artifacts have been
materialized because the authorized key-one and key-two budget epochs are
exhausted. Key three is not authorized for v6.

## Historical v5 materialization state

The final v5 census and exact prompt-hash cache coverage are:

| Task | Candidate buckets | Review rows | Review chunks | Reviewer calls | Valid cached calls | Missing calls | Published v5 manifest |
|---|---:|---:|---:|---:|---:|---:|---|
| BBB | 253 | 292 | 253 | 506 | 506 | 0 | Yes |
| Bioavailability | 2,153 | 6,476 | 2,162 | 4,324 | 4,243 | 81 | No |
| Skin | 463 | 7,328 | 482 | 964 | 0 | 964 | No |
| **Total** | **2,869** | **14,096** | **2,897** | **5,794** | **4,749** | **1,045** | **BBB only** |

BBB v5 has complete two-reviewer coverage. All 292 candidate rows were kept: 247
fit the bucket, 10 were reviewed as valid extremes, and 35 were kept because the
reviewers disagreed. The v5-aware Stage 06-07 rebuild therefore excludes zero
records and materializes 291,210 collapsed rows representing 480,325 source
records. The root build cache hashes the v5 manifest, decisions, and reviews.

Bioavailability and Skin remain fail closed under v5 because their two-reviewer
coverage is incomplete. The three physical keys were historically accounted at
9,987,356, 9,994,838, and 9,956,081 tokens, leaving 12,644, 5,162, and 43,919
tokens respectively. V6 is restricted to the first two keys; their exhausted
epochs are insufficient for a new materialization.

The prior paid work remains in versioned v3/v4 caches, but changing the bucket
membership and displayed tail statistics invalidated most prompt hashes. Those
answers were not relabeled as v5 reviews. The former BBB v4 Stage 06-07 build was
replaced only after complete v5 coverage passed the downstream preflight.

## Root causes of the earlier suspicious records

The earlier five variance-cohort findings do not share one failure mode:

| Finding | What happened | Why extraction/pruning did not fix it |
|---|---|---|
| Bioavailability `109.4` became `10,940%` | The support text gave `F = 109.4` without a percent sign. The extraction model labeled it `fraction`; deterministic normalization then correctly multiplied that declared fraction by 100. | This was an LLM unit-semantic error, not a comma-routing error. Earlier pruning used the wrong `retrieval_eligible` selector and never reviewed this direct pair-bucket row. |
| Literal `602%` systemic availability | The model extracted the displayed percent correctly. The questionable part is whether the source quantity is truly absolute systemic availability. | This is a reference-scope/source-semantics issue, not a numeric or unit parse failure. Pruning can only drop it after review; it cannot reclassify it. |
| 5 mg AUC rows reported as `mg*h/mL` | The source support and stored unit both literally say `mg*h/mL`; these rows used `source_exact`. Unit normalization converted the stated unit correctly, producing physically implausible values. | The likely error is upstream source extraction or the source table's metric prefix, not the measurement-resolution LLM. Earlier reviewers treated the displayed unit as a supported extreme and kept the cluster. |
| `81.33%` epidermal retention under dermal absorption | The numeric value and percent unit are correct, but one source row's endpoint is dermal absorption even though its support text says epidermal retention. | This is endpoint classification, not unit extraction. Pruning can drop the row but does not repair the endpoint. |
| `44.39 g/cm^2` skin retention | The stored source value and unit literally say `g/cm2` and therefore took the exact route. The normalized conversion is dimensionally correct but scientifically implausible. | The likely metric-prefix error is already present in source text. The stricter float gate does not apply to units, and Skin v5 review is not materialized. |

The additional out-of-cohort `1.7 ng*h/mL*10^3` issue is different again: its
reviewed exact-unit map used scale `0.001` where the literal suffix implies
`1000`. That is a deterministic map-entry defect, not an LLM extraction error.

## Why old pruning missed problems

Final-pruning v2 required a well-supported two-sided gap in a sufficiently large
bucket. A single bad tail, a small bucket, or a source-supported cluster could
therefore survive. The first v3 implementation added leave-one-out tails but
incorrectly filtered on record-level `retrieval_eligible` and silently omitted
nonpositive values from positive log geometry. V5 fixes both errors by using
pair-bucket eligibility and native-SD fallback.

Pruning remains intentionally conservative: it removes a reviewed bad row; it
does not infer a replacement value or silently change scientific semantics.

## Validation

- The v6 focused pruning suite has 23 passing tests, including deterministic
  value-spread sampling, all-available small-bucket context, complete-gap
  chunking, context/target disjointness, Jinja formatting, direct single-vote
  decisions, and downstream exclusion.
- The broader pruning, collapse, measurement-routing, source-cleaning,
  measurement-resolution, and reference-semantics regression run has 288
  passing tests.
- A prior focused run covering final pruning, token-ledger behavior,
  measurement routing, source cleaning, measurement-resolution generation,
  and reference semantics has 224 passing tests. This includes pair-bucket
  versus retrieval eligibility and native-SD fallback regressions.
- The broader routing, cleaning, mapping, canonicalization, collapse, and task
  policy run has 364 passing tests. Its 11 setup errors all come from one known
  bounded Skin fixture that lacks the exact unit-map key
  `skin_sensitization_grading_score/score`; the full production Skin build has
  zero unresolved extraction rows and remains covered by its exact unit map.
- The pair-bucket variance audit produces identical output hashes on a second
  run, and `git diff --check` passes.
- BBB v5 has 506 validated review results and 292 keep decisions with zero
  drops. Its strict downstream rebuild completes through Stage 07 and the root
  manifest hashes only `final_endpoint_pruning_v5` inputs.
- Bioavailability has 81 exact prompt-hash misses and Skin has 964, so neither
  has a v5 manifest or a v5 downstream rebuild.
