# Record diversity: shared preprocessing and historical molecule-local experiments

The molecule-local experiments below are historical development runs, not
comparison arms for the redesigned paper. Shared extraction/encoding remains
reused by [Direct + Joint](BUDGETED_EVIDENCE.md), whose working annotation/input
contracts are v4/v3. That workflow is paused by user as of 2026-09-17; do not
resume extraction or old experiments from the example commands here.

## DILI native relevance and completeness (2026-09-23)

For the separately requested simple DILI record score, use the shared
`dili_score.score_dili_record` function. The old
`common.record_selection_context.score_dili_record` import re-exports this same
implementation. Install and API details are in `dili_score/README.md`; the package
uses only the Python standard library. This is available
for indirect L2+ records and makes no model calls. It is independent of the
paused Quality/Joint annotation workflow and does not enable a prediction arm.

```python
from tools.chembl_tool.common.record_selection_context import score_dili_record

result = score_dili_record(record, condition="age_group=older_than_65")
score = result["score"]
components = result["components"]  # relevance, completeness
condition = result["condition"]    # separate compatibility, including per-axis status
```

`record` can be a native/canonical source row, including `raw_record_json` and
`reviewed_record_json`, or an existing `prepare_record` output. Prefer the source
row: it preserves typed age/population/regimen fields omitted by historical
prepared caches. The scorer respects reviewed nulls and does not mutate records.
Existing lost fields cannot be recovered from an old prepared payload alone.

The version is `dili_relevance_completeness.v5`, with `score = E * I`.

**Current audit status: bounded repairs verified; coarse relevance limits remain.**
The previous 300-record audit's 11 score defects and five excluded-fragment errors
are repaired. A further 276 unique records across L2-L7 and six sources were read
in rounds of 180, 60 and 36; those rounds exposed another 7, 2 and 1 repaired score
defects. All three rounds informed edits, so none independently certifies the final
version. A serum ET-1 record in a hepatopulmonary-syndrome model remains an explicit
relevance boundary: broad hepatic-model wording can overstate hepatic endpoint
specificity. The 10,596-candidate replay is regression verification, not manual
review of every row. See `outputs/paper/analysis/rule_quality_audit_20260923/dili_v5_review/`
for frozen samples, decisions and reproducible checks. These are risk-enriched
assistant field reviews, not a library error-rate estimate.

- E: explicit liver endpoint or recognized mechanism with hepatic system = 1;
  unknown relevance/system = .5; explicit off-task readout = 0; explicit plant
  system caps E at .25. Mixed liver/off-task readouts remain limited (.5).
  FXR/HMGB1, nucleophile trapping/cysteine adducts and generic mechanisms require
  an identified hepatic context. Bounded phrases describing liver origin as
  implied/inferred/presumed/assumed do not establish that context.
  Source support paragraphs never supply an unrelated readout's relevance.
  v3 adds Hep3B, HMGB-1, hepatic-gated CYP/P450 and explicit stress/metabolic
  categories, and the steatohepatitis endpoint. A statement that hepatic
  competence is not reported does not establish liver identity. An explicit
  source warning of a misidentified HeLa derivative caps E at .5; this is not
  independent cell-line authentication. Cotyledons trigger the plant cap.
  v5 adds hepatic-gated transporter substrate/kinetics and rejects explicit
  unconfirmed hepatocyte identity and bounded liver-disease-context wording as
  evidence of measured hepatic tissue; it does not resolve every disease/model
  versus measurement-site ambiguity.
- I: fraction of structured endpoint, result, assay/system and support present. Field-name
  wrappers and missing placeholders are removed before checking. Qualitative
  outcomes count; units/IDs alone do not. Dose/time/control completeness is not
  claimed by these four checks.
  `completeness_basis` names this field-presence contract and
  `completeness_evidence` lists the source field keys used by each check.
  A missing result field does not imply that the support prose has no result;
  prose-only outcomes are not automatically extracted or counted twice.
  v3 recovers scientific key=value fields from canonical-only cards, respecting
  explicit native/reviewed nulls. Measurement type alone is not a result.
  v4 screens obvious method-only, metric-name-only and figure/table-only result
  descriptions. Explicit qualitative outcomes, including negative/null findings,
  still count, as do numeric readouts. Doses, model names and figure numbers alone
  do not rescue the known description-only cases. `excluded_result_descriptions`
  exposes the removed fragments; if no independent result remains, reasons include
  `result_description_without_reported_outcome`. This bounded screen is not a
  general semantic parser: unknown prose is retained, and numeric attribution or
  whether a statement is true is not verified.
  v5 preserves result prose across semicolons and reference-only parentheses,
  while still separating typed key=value fields. `result_content_status` is
  `reported` (rule-recognized), `description_only`, `missing`, or `unresolved`.
  Unrecognized prose retains field-presence credit and is listed in
  `unresolved_result_descriptions`; a high score with `unresolved` does **not**
  establish that a real outcome was reported. `reported` also does not establish
  source truth, focal attribution, or correct numerical interpretation.
- Condition compatibility is separate. Exact typed fields/qualifiers and bounded
  age rules are used; missing data stays unknown. A different baseline disease
  is not automatically a mismatch because diseases can coexist. Animal age
  does not establish a human age stratum. Raw support is not mined for conditions.
- Object role, benchmark labels, outcome direction and `needs_more_context`
  do not alter E*I. The latter flag remains visible for inspection.

This is a coarse relevance/information score, not study validity, a probability,
or query-molecule transferability. Many complete records tie; keep the existing
retrieval order for ties. The historical `record_quality` API and its assessed
V/E/C/O/I contract remain separate. Freeze this version and input hashes before
using the new score in an experiment.

Preceding result-screen repair/replay:
`outputs/paper/analysis/rule_quality_audit_20260923/dili_result_review/`.
It replays all 10,596 candidates and the 492 expanded samples against frozen v3,
checks source/prepared score agreement, and reviews changed result fields.
Relevance is unchanged; this repair only affects result completeness.

The preceding v3 expanded audit is preserved in
`outputs/paper/analysis/rule_quality_audit_20260923/dili_expanded_review/`:
492 additional unique records across L2-L7: 360 deterministic stratified records
outside the frozen candidate union and 132 targeted risk records inside it.
The 120 follow-up records also informed a repair; they are not an independent
test set. All 10,596 frozen candidates were replayed: 54 simple scores changed;
legacy quality scores were unchanged. The 1,278,020 eligible library membership
rows were inventoried for sampling, not exhaustively reviewed. This is an
assistant source-field/excerpt audit, not full-paper verification or a ranking
accuracy estimate. See the audit README for unresolved semantic limitations.
Preceding v2 pair review (30 pairs / 60 records) remains in `dili_pair_review/`;
the v1 audit remains in `dili_repair/` under the same parent.

## Historical molecule-local status

The six-task **scaffold-valid** experiment uses similarity ≥ 0.3, at most 10
molecules per level, at most 5 newly unlocked records per molecule per level,
L1/L2 scaffold-disjoint and L3+ parent-disjoint retrieval. Earlier cards remain
visible. Gold labels never enter feature extraction or record selection.

| Version | Role | Status |
|---|---|---|
| `assay_diverse.v1` | Shared runner default; assay-spanning quality order | Baseline |
| `feature_based.v1` | Five-field lexical apricot ablation | Optional runner policy, not the semantic experiment |
| `outcome_diverse.v1`, `anchored_complete.v1/v2` | Historical level-wide semantic coverage | Retained to reproduce frozen experiments |
| `molecule_local.v1` + context v1 | Independent semantic coverage per molecule | All six valid tasks complete |
| `molecule_local.v1` + `record_selection_context.v2` + reviewed UGT overlay | Expanded endpoint recognition and bounded condition repair | Selection replay only; known quality issues; no inference scores |
| `molecule_local.v1` + `record_selection_context.v3` | Working species-attribution repair using explicit population/system fields | Code regression checks only; no full selection replay or inference scores |

The authoritative result index is `current_conditioned_results.json`, under
`replicate_suites.record_diversity_six_task_valid_20260916`. Its
`molecule_local_replay` contains completed results and the latest selection-only
release. [RESULTS.md](RESULTS.md#record-diversity-current-results) gives the
comparison; the release does not certify all source semantics.

## Maintained entrypoints

Run commands from the repository root; on node001/node002 use the `vllm` conda
environment and one native BLAS/OpenMP thread per worker.

| Entry | Purpose | Model calls |
|---|---|---|
| `prepare_record_diversity.py` | Inventory source columns, routing and frozen molecule-level scope; optionally write normalized Parquet inputs | None |
| `benchmark_record_diversity.py` | Bounded extraction speed/validation probe using the shared provider pool | Explicitly requested sample |
| `run_record_diversity_extraction.py` | Resumable exact-input-deduplicated offline annotations | Pending eligible requests; none with `--prepare-only` |
| `common.record_diversity.prepare_record / encode_features` | Source adapter, annotation contract, normalized sparse features | None |
| `common.record_selection_context.select_contextual_records` | Current molecule-local selection over cached features and eligible raw cards | None |
| `run_conditioned_assay_progressive_curve.py` | Progressive inference over prepared evidence | Reasoning calls |
| `run_conditioned_assay_family_curve.py` | Matched full-flat or frozen-input replay | Reasoning calls |
| `plot_assay_retrieval_curve.py` | Canonical level comparison plots | None |

All paths above are within `tools/chembl_tool/paper_experiments/` except the
explicit common modules. There is no second semantic inference runner.
Suite-local `prepare_paired.py`, `prepare_full.py`, repair/replay scripts and frozen
runtime copies are experiment provenance, not additional maintained CLI APIs.
The semantic selector is currently integrated through those registered prepared
arms. **`--card-selection-policy feature_based.v1` does not enable it**: that flag
selects the earlier lexical ablation only.

## Data flow

1. **Recover native fields once.** `prepare_record` preserves six groups:
   endpoint, outcome, assay, biological system, exposure and comparison/role.
   Reviewed JSON overrides raw JSON, including explicit null corrections. Ames
   native rows join by source ID and source row number. Ambiguous joins must not
   borrow another record's annotation.
2. **Inventory and scope.** `prepare_record_diversity` scans eligible source rows.
   Exact source calls can supply outcomes directly; nonstandard calls, uncertain
   roles, interventions and contrasting text route to LLM candidates. This is a
   workload estimate, not a semantic-quality certificate.
3. **Extract missing semantics offline.** The v3 annotation binds endpoint/result
   to focal entity, conditions, comparator, statistical support and source quotes.
   Native numerical values and full source text remain authoritative.
4. **Encode/cache once.** `encode_features` returns `(feature_weights, outcomes)`:
   a sparse nonnegative feature dictionary and a set of `(endpoint, outcome)`
   coverage keys. Feature cache manifests pin source inputs, encoder settings and
   any reviewed overlay. Repeated queries reuse these cached vectors.
5. **Select at retrieval time.** Retrieve molecules under the frozen budget and
   disjoint rules, take this level's newly unlocked cards, then select independently
   inside each molecule. No LLM is called for ranking.
6. **Render unchanged raw cards.** Features, inferred outcome keys, relevance ranks
   and selection audits are not reasoning evidence and are not added to prompts.

Fresh index construction uses `--max-record-examples 0` to retain all distinct
cards per assay×molecule. Positive caps reproduce historical indexes. This
upstream setting differs from the five-record query-time budget. Source-native
Skin result labels and Bioavailability study/dose/comparator fields are preserved
by fresh uncapped builds. Index changes require refreshing the candidate scope.
See [CURRENT_STARLING_RETRIEVAL.md](CURRENT_STARLING_RETRIEVAL.md).

### Inventory and frozen scope

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m \
  tools.chembl_tool.paper_experiments.prepare_record_diversity \
  --output-dir outputs/paper/analysis/record_diversity_inventory --workers 6
```

Default output is counts plus bounded hash-selected samples. `--write-features`
writes normalized input payloads to `*.features.parquet`, **not** encoded vectors.
`--molecule-filter FILE --molecule-filter-field molecule_id` accepts
`task -> subset -> level -> [source molecule IDs]`; canonical SMILES is also
supported. Expand selected analogs through the index's exact identity mapping,
not just their displayed representative SMILES. Counts cover the union once and
include eligible source records omitted by historical index/card caps.

Changing budgets, thresholds, splits, sources or indexes requires a new scope
census and incremental annotation completion. Matched full-flat can reuse the
same evidence scope. Historical workload/speed estimates are artifacts, not
current live throughput promises.

### Offline annotations and resume

Both extraction CLIs support `--annotation-version record_diversity_annotation.v3`;
pass it explicitly for bound outcomes. Their v2 default remains for historical
compatibility. Version/prompt/input/model changes require a separate database.
Optional v3 `review_notes` are source-bound and participate in cache keys.

`run_record_diversity_extraction --inputs DIR --output-dir DIR --provider-config
FILE` reads normalized Parquet plus its inventory summary. `--parallelism` bounds
the global job pool; each provider's `max_inflight` and shared admission socket
bound actual requests including retries. Probes and bulk jobs share the same
thinking-parameter adapter. Bulk reasoning is disabled; a benchmark can explicitly
request `--enable-thinking`. Endpoints and credentials come from provider config
and environment; do not copy time-sensitive addresses into new code.

SQLite stores unique inputs, source-to-job mappings, all attempts and validated
annotations. `progress.json` reports pending/done/failed jobs and provider load;
`completion.json` distinguishes complete from complete-with-failures. Exhausted
validation failures return exit code 1; fatal provider errors return 2. Successful
annotations and exhausted attempts are preserved on resume. `pause.json` holds
survive restarts. SIGTERM drains requests before checkpointing.

`--database-path` supports node-local storage with an atomic shared-disk checkpoint
every ten minutes and on graceful exit. Restore that checkpoint after node loss;
work since the last snapshot may need repeating. Input/model contracts must still
match. Provider aliases can change only with compatible model identity and
semantic generation settings. Provider-reported costs do not include unreported
institutional compute costs.

`--quote-catalog` offers deterministic source-span IDs; `--repair-quotes` restores
uniquely matched literal spans. Both retain raw attempts and repair receipts and
still require validation. Schema/substring validation does not prove the result
belongs to the focal molecule or has the right scientific interpretation.

## Selection semantics

The current adapter groups cards by molecule. If its eligible count is ≤ budget,
all records retain their input order without ranking. Otherwise:

1. Order by bounded query-condition compatibility, task relevance, measurement
   scope and the existing measured-value/support quality order.
2. Protect one anchor within the best condition rank; it counts toward the quota.
3. Fill remaining slots using condition/completeness priority, new relevant
   endpoint×outcome coverage, then diminishing square-root feature gain.

Both outcome and feature coverage are **molecule-local**. Outcome coverage resets
per molecule/level; only that molecule's prior context features carry forward.
Global unkeyed history is rejected. Earlier unselected records do not become
newly unlocked again. There is no forced positive/negative ratio.

The objective is `sum_j sqrt(sum_{r in selected} X[r,j])`. Each populated feature
block is normalized so its isolated gain is bounded regardless of verbosity.
The semantic selector uses one cached Numba sparse gain kernel for its constraints;
it does not run unconstrained apricot `fit` across the level. Historical policies
retain their old coverage semantics for reproducibility, not as current defaults.

Native fields preserve assay, species/model, dose, time, route, treatment and
comparator distinctions. Semantic tags supplement rather than replace them.
Missing values remain missing. Administrative IDs, gold labels, votes, final-task
inferred directions and level numbers are not features. Unresolved annotations,
unclear/non-focal results, numeric-only results and nonsignificant trends do not
receive hard outcome priority (`require_resolved_outcomes=True`).

`record_selection_context.v2` uses shared endpoint vocabularies and a bounded
assay-field fallback for generic readouts. It never searches support text to make
an unrelated outcome eligible. These relevance hints do not change family
assignment or infer final labels. Explicit Ames strain variants and transformed
specimens are treated separately from the strict gold parser.

The working `record_selection_context.v3` additionally reads native biological
system/model fields for species matching. Human-equivalent doses or platform
mentions in qualifiers cannot turn an explicit rat specimen into a human study.
Missing tested species stays unknown; conflicting explicit groups are not exact
matches. The reviewed Carcinogens #367 rat-sperm example motivated this repair.
114 relevant regression tests passed. Frozen v1/v2 results and selected lists
remain unchanged; v3 still needs a full selection replay before any inference.
This does not resolve all experimental-system attribution or task-relevance issues.

### Continuous values: absolute oral F

`encode_features(..., absolute_oral_f=True)` adds versioned `absolute_oral_f.v2`
features; other values remain field-scoped context, not raw numerical weights.
Explicit absolute oral F with percent/fraction units uses fixed percentage bins
`[0,20)`, `[20,40)`, `[40,60)`, `[60,80)`, `[80,100]`. No labels fit the bins and no
high/low task call is inferred. Context and measured/dosed entity bindings keep
incompatible observations separate.

Scalars and estimate±uncertainty use the estimate's bin; uncertainty stays in the
source. Genuine ranges/bounds share unit total initial gain across intersected
bins, not an invented midpoint. Unitless, relative, ambiguous or multiple readouts
receive no absolute-F feature and remain eligible raw evidence. Prodrug/metabolite,
unbound and labelled measurements retain explicit scope; no AUC-to-F calculation
or guessed chemical alias equivalence is performed.

The UGT reviewed overlay corrects one Bio L6 card: low catalytic capacity relative
to named isoforms is not absence/negative activity. Relative increase/decrease
retains those cross-isoform comparators, not treatment-induced inhibition/induction.
Its full Bio replay changed zero additional selected lists. The latest release
pins the overlay separately; historical cache/prepared inputs remain frozen.

## Lexical apricot ablation and dependencies

`common.record_feature_selection` implements `feature_based.v1` using only endpoint,
assay_context, species, qualifying_conditions and reported_value. Field-scoped
words/bigrams include local negation text but do not solve scientific negation,
focal attribution or synonymy; support text remains raw but is not encoded.

`FeatureBasedSelection.fit(X)` takes a nonnegative record×feature matrix; CSR
stores only nonzeros. `ranking` gives selected row indices and `gains` their
objective increments, not probabilities. `fit_transform` returns matrix rows,
not records. The adapter pins optional `apricot-select==0.6.1`, reuses its compiled
sparse kernel and keeps legacy query-wide history. Normal assay selection and the
semantic cached Numba selector do not require apricot. An absent/mismatched
optional dependency fails explicitly when the lexical policy is requested.

## Artifacts, known issues and reuse

The six-task suite's latest `selector_repair/final_review/selection_release.json`
pins the feature manifest directory and selected lists. The UGT replay consumes
its reviewed overlay explicitly. Historical `prepare_full.py` reads the original
suite manifests; it is not a new-release preparation command. A future arm must
use a separate root and pin its effective manifest, including overlays, before
inference. Never overwrite completed prepared inputs or infer that changed source
code updated an old experiment.
A changed selected surface invalidates its progressive suffix; verified unchanged
prefixes may be reused only with the existing dependency/hash receipt.

The latest selection audit still finds native species fields missed or incidental
human references treated as study species, insufficient DILI hepatic-system
priority, formulation/relative-exposure tradeoffs, and near-duplicate records
occupying multiple slots. Expanded outcome counts do not prove better evidence.
These are open scientific selection issues, not fixed by this maintenance review.
Bio's opt-in `measurement_scope.v1` prompt adaptation remains separate and was
**disabled** in the completed molecule-local comparison.

Retain source/annotation/feature hash receipts, reviewed overlays, selected lists,
prepared/request/output traces and figures for registered comparisons. Suite-local
scripts and frozen runtimes are provenance, not new active entrypoints. See
[TRACE_RETENTION.md](TRACE_RETENTION.md). The standalone `record_observations.v2`
pilot has been retired from maintained code; its exact source/test snapshot and
historical location are recorded in
`receipts/record_diversity_maintenance_20260916.json`. The active v2/v3 annotation
contracts remain in `record_diversity.py`; no new extraction schema is introduced.
