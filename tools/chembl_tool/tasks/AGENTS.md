# Molecular Evidence Task Extension Conventions

This file applies to all current and future tasks under `tools/chembl_tool/tasks/`. A task directory may add its own data paths, endpoint semantics, and run commands, but it must not violate the shared layering defined here.

## Two-layer evidence classification

Every task must explicitly distinguish the following two layers. They must not both be implemented as LLM reasoning branches.

### Mechanism family

A mechanism family is a task-level, source-independent semantic category and the basic unit of retrieval and parallel reasoning. For example, Bioavailability_Ma uses Observed direct F, Observed oral exposure, Fa, Fg, and Fh; BBB uses direct exposure, passive permeability, efflux, and influx.

Requirements:

- Define a small, stable set of mechanism families from the task mechanism before integrating ChEMBL, Starling, or another data source.
- In `full_mechanism`, each mechanism family may correspond to at most one parallel group-reasoning branch.
- `direct` normally enables only the direct-outcome family. `full_flat` uses the same evidence union as `full_mechanism`, but combines it into one reasoning branch.
- There is no fixed family count shared across tasks, but the count must remain restrained. It should normally be a small set of interpretable categories rather than dozens of endpoint branches.
- Family definitions must not depend on whichever assays happen to exist in a current data source, and must not contain benchmark label policy, voting rules, or deterministic overrides.

### Source-local endpoint group

Source-local endpoint groups are fine-grained normalization and audit labels stored in evidence and
provenance. Paper-facing retrieval should map them into task-level mechanism families before ranking and
reasoning. The current ChEMBL behavior and its limitations are documented below.

## Configuration locations for a new task

Use the following division of responsibilities when adding a task:

```text
tools/chembl_tool/tasks/<task>/experiment_config.py
  Declares source-independent mechanism families and mappings from each source group to a family.

tools/chembl_tool/tasks/<task>/endpoint_groups.py
  Handles only fine-grained endpoint normalization and audit labels for heterogeneous raw data such as ChEMBL.

tools/chembl_tool/common/evidence_contract.py
  Converts different sources to minimal_evidence.v1; does not predict labels.

tools/chembl_tool/common/experiment_retrieval.py
  Combines candidates by mechanism family, retrieves neighbors, and constructs direct, full_flat, and full_mechanism views.

tools/chembl_tool/common/molecule_identity.py
  Uses versioned RDKit FragmentParent normalization for whole records, fragment/molecular parents, and mixture components.

tools/chembl_tool/common/retrieval_policy.py
  Defines operational and parent_disjoint candidate-exclusion policies centrally; tasks and source adapters must not duplicate this logic.

tools/chembl_tool/common/neighbor_selection.py
  Performs pluggable top-k set selection over candidates that have already passed the similarity threshold,
  identity policy, and evidence-availability checks. `similarity` preserves the historical pointwise Tanimoto
  ranking. Without lowering the existing `min_similarity`, `query_feature_coverage` greedily maximizes marginal
  union coverage of query Morgan bits and uses Tanimoto only to break ties. A selector must not modify evidence
  rows, mechanism-family mappings, or the downstream retrieval JSON payload schema.

tools/chembl_tool/common/retrieval_ablation.py
  Computes stable hashes of LLM-visible sample/family inputs, materializes reuse of a complete run or independent branch, and records provenance.

tools/chembl_tool/common/retrieval_replay.py
  Reads frozen retrieval artifacts and validates query identity for strict matched-prefetch replay; it must not retrieve again from the current index.

tools/chembl_tool/common/identity_blind.py
  Implements identity redaction, harness prefetch, and visible prefetched-tool replay centrally; task runners must not fork this logic.

tools/chembl_tool/common/units.py
  Owns context-free unit cleaning, structural parsing, and dimensional folding. Parsing and arithmetic
  live here and are never scoped per task; task context enters only through the two policy files below.

tools/chembl_tool/common/qualifier_vocabulary_policy.json
  Declares, per task, which unresolved tokens count as dimensionless qualifiers. A token in one task's
  vocabulary must never change another task's parsing; tokens shared by every task go in the `shared` entry.

tools/chembl_tool/common/contextual_unit_policy.json
  Declares reviewed, exact-match assay-context rules that pick a canonical metric prefix within one
  assay stratum. Fail-closed: every field a rule declares must be present and exactly equal.

tools/chembl_tool/common/PROMPT_PROFILE_CONTRACT.md
  Freezes the shared reasoning payload, structured-output validation, task prompt profile,
  final-decision profile, manifest provenance, and branch-reuse gate. A task-local profile owns only
  that task's semantics, schema, and cross-field rules.

tools/chembl_tool/tasks/<task>/run_reasoning_pipeline.py
  Exposes `build_group_prompt_payload()` to external workflows that reuse a task-specific full-flat
  prompt. RL/data materializers must not call the private `_group_prompt_payload()` implementation;
  the historical private alias exists only for compatibility.
```

Task code must not duplicate shared retrieval, source aggregation, LLM client, validation, or batch orchestration.

## Starling cleaning, canonicalization, and pair buckets

New task builds use `starling_normalized_v7`. Frozen v6 artifacts remain readable and must not be
rewritten or relabeled as v7.

Every source field first receives only basic, meaning-preserving cleaning. Only four universal roles may
be renamed at this boundary: the source endpoint, measurement/value, unit, and structure fields become
`endpoint_name`, `measurement_text`, `unit_text`, and `smiles`. All other fields keep their real source
names. Do not create cleaned aliases such as `species_context` when the source actually supplied only an
`assay_system` string.

Reviewed integration happens in Stage 02 and produces one final `canonical_*` field per semantic
dimension. A canonical field may use several cleaned inputs, and one cleaned input may support several
canonical outputs. For example, `assay_system` may independently support `canonical_assay_context` and
`canonical_species_context`; both outputs must record `assay_system` as their input. Missing canonical
values do not fall back to a differently named source field.

Scalar reference semantics are also a Stage-02 canonical dimension. Each task freezes its own bounded
`gpt-5.4-mini` request shape while the shared offline classifier assigns each record at most once; invalid
responses and failed requests become `unknown` without retry. Its frozen row mapping is joined as
`canonical_reference_scope` and, where required, `canonical_reference_basis`. Context fields used only as
classification evidence may remain residual-heterogeneity candidates because the classifier does not
normalize or consume their values. Stage 02 publishes `reference_semantics_manifest.json` with mapping
hash and coverage; task prompts and mappings are cache-fingerprinted.
The validator may normalize exact provenance formatting, but it must never relabel GPT scope or basis.
Contradictory semantic assignments fail closed to `unknown`; revisit the prompt or taxonomy only when one
failure class exceeds 10% of classified rows.

Every field used in pair-bucket identity must be canonicalized, whether by a deterministic rule, a
reviewed frozen mapping, or a controlled encoder. A cleaned field touched by ordinary canonicalization must
not also be a variance candidate. The narrow exception is an input used exclusively by a conditional
controlled encoder: because real scalars and encoded outcomes are mutually exclusive, that field may remain
a residual-heterogeneity candidate for the continuous rows. The selected categorical scale's inputs are
excluded from its categorical residual audit. Other untouched cleaned source fields may be evaluated for
residual heterogeneity, and they do not silently create child buckets. Measurement and unit canonicalization is an
atomic, task-reviewed decision: never change a numeric value without changing its unit provenance in the
same rule. A parser is not applied blindly to every source value. Each controlled measurement has a frozen
source ID, real input fields, parser ID, measurement kind, and definition. Binary and ordinal declarations
also freeze category IDs, ranks, and encoded values.

Stage 04 is the only layer that decides pair-bucket membership. Its key uses canonical fields only and its
sidecar persists `measurement_kind`, `canonical_measurement_scale_id`, `canonical_category_id`, and
`canonical_category_rank`. Current supported kinds are `continuous`, `binary`, and `ordinal`; nominal
unordered outcomes remain evidence-only. A source with a controlled scale must include
`canonical_measurement_scale_id` in its bucket identity, so incompatible scales cannot mix.
Task schemas also declare eligible reference scopes. Unknown and comparator-relative measurements remain
valid evidence records but are excluded from assay-transfer buckets; accepted reference scope, and basis
for tasks that use it, are part of the bucket key so different denominators cannot mix.

Stage 05 validates and calibrates the bucket observed in Stage 04; it never creates child buckets or changes
membership. Every bucket needs at least 25 records. Binary buckets must observe both declared levels;
ordinal buckets must observe at least three declared levels; every observed categorical level needs at least
three records. Continuous residual heterogeneity uses the existing omega-squared audit. Binary and ordinal
residual heterogeneity uses bias-corrected Cramér's V-squared, requires candidate levels with at least three
records and at least 50% coverage, and flags values at or above 0.20. Valid buckets store a first-class
record-weighted sample SD with its ddof and source field. Valid continuous buckets additionally store the
exact empirical value CDF as sorted support values, counts, and midranks. Valid ordinal buckets store a
category-rank CDF over the complete declared domain; binary buckets retain explicit same/different semantics
and do not publish a CDF. V7 Stage 05 does not store a raw-
distance CDF, pair samples, transfer cutoff, Boolean label, or soft probability. Downstream code may use SD
for standardized raw-distance calculations or use same-bucket empirical-CDF separation for
location-sensitive geometry.

The shared structure is:

```text
tools/chembl_tool/common/starling/build_normalized_evidence_library.py
  Shared, resumable clean -> canonicalize -> organize driver for Stages 01-03.

tools/chembl_tool/common/starling/canonicalization_v7.py
  Strict source schemas, canonical-dimension lineage, pair-bucket declarations, and v6 compatibility
  aliases that exist only in memory.

tools/chembl_tool/common/starling/split_downstream.py
  Shared transactional Stages 04-09: complete-data pair buckets and distance calibration, then
  random/scaffold label-source filtering, molecule evidence, neighbor indices, and audits.

tools/chembl_tool/common/starling/build_pair_bucket_distance_calibration.py
  V7 bucket validation, continuous/categorical residual-heterogeneity audits, first-class sample SD, exact
  continuous value-CDF geometry, and exact ordinal category-rank CDF geometry. The historical module/artifact name is retained for compatibility;
  v2 emits no raw-distance CDF, pair labels, transfer thresholds, or probabilities.

tools/chembl_tool/common/starling/normalization/task_policy.py
  StarlingTaskPolicy: the sole entry point for all task-specific inputs, including source profiles,
  canonical record contract, scientific hooks, column contracts, and manifest versions.

tools/chembl_tool/common/starling/{compact_artifacts,auxiliary_metadata,policy_distance}.py
  Compact artifacts/indexes, globally reconciled auxiliary attachment, and endpoint-policy distance mathematics.
  The task supplies profiles, version numbers, and labels to all three; task-specific strings must not be inlined.

tools/chembl_tool/tasks/<task>/starling_policy.py
  The task plug-in, which must export POLICY. Builders and directory-index loaders resolve it through
  importlib under this convention and do not maintain an additional registry.

tools/chembl_tool/tasks/<task>/build_normalized_starling_evidence_library.py
  Thin policy/downstream binding that preserves the task command line.

tools/chembl_tool/tasks/<task>/starling_schema.py
  The task's complete source-visible fields, canonical dimensions and their real inputs, pair identity,
  and cleaned variance candidates.
```

The persisted v7 stages are `01_cleaned`, `02_canonicalized`, `03_records`, `04_pair_buckets`,
`05_distance_calibration`, `06_remove_heldout_overlap`, `07_molecule_evidence`, `08_neighbor_index`, and
`09_audits`. When integrating a new task,
write only the task policy and a small set of declarative/scientific modules. Do not copy staged
construction, invalidation, resume validation, or manifest assembly logic.

`compact_persisted_records` removes `assay_tier`, `endpoint_group`, `evidence_role`, and `target_pref_name` because they are derivable. The evidence catalog must therefore accept a `family_resolver` and rederive these fields. Otherwise, a resumed `--from-stage index` build produces a different catalog from a complete build.

`05_distance_calibration` is fit on the complete unfiltered record set in every task, held-out gold
included. This aggregate, label-free SD/empirical-CDF fit is the sole permitted use of benchmark-held-out source
measurements before evaluation. Bucket geometry describes the assay landscape rather than any particular
molecule set, so excluding gold molecules would bias the statistics without preventing prediction-time
exposure. `06_remove_heldout_overlap` is the mandatory boundary: it drops held-out label-source records
before molecule evidence and neighbor indices are built, which prevents those records from reaching a
prediction. Every task must follow this ordering and must not declare `heldout_sources` or a held-out key
loader on its Stage-05 calibration.

## Normalization regression corpus

When a new class of normalization or parsing defect is found, the fix must add the real
records that exposed it, together with their hand-validated expected output, to
`tests/chembl_tool/common/fixtures/normalization_regressions.jsonl`. Take the record
verbatim from a built `01_cleaned` stage; never invent a string that merely resembles the
data. Each entry must record the reasoning that establishes its expected output as correct,
so a later reader can re-derive the decision instead of trusting it.

This corpus is hand-maintained and must never be machine-regenerated. It is deliberately
unlike `units_golden.json`, which is generated from the top real forms and then reviewed.
A failing case is either a regression or a decision that must be re-validated by hand and
its reasoning updated. Do not edit an expected value to make a test pass.

Place a fix at the most general layer that is correct, and descend only when the correct
behavior genuinely differs at that scope:

1. `common/units.py` -- context-free parsing and arithmetic. A wrong dimension is wrong in
   every assay, so fix it here and never per task.
2. `common/qualifier_vocabulary_policy.json` -- which tokens a task treats as dimensionless
   qualifiers. Always task-scoped, never global: a declared qualifier resolves, and so
   promotes records out of the unrecognized-unit quarantine in every task that shares the
   declaration.
3. `common/contextual_unit_policy.json` -- reviewed exact-match assay-context rules.
4. The `source_measurement_resolver` hook, per source and endpoint. Skin's
   `measurement_semantics.json` is the model: a declarative reviewed registry, not ad hoc
   code. This is the granular floor; there is no record-level override.

Descending is allowed -- some data is genuinely too messy to generalize -- but the narrower
layers all carry a `review` block, so say there why a general rule was not possible.

**Do not chase every residual form.** A unit that cannot be parsed correctly should stay
unrecognized, which already makes the record non-scalar and therefore ineligible for pair
buckets and assay transfer. That is the intended outcome, not a gap to close: dropping a
handful of unparseable records costs far less than a bespoke rule that encodes one dataset's
mess, or an approximation that yields a confidently wrong number. Prefer failing closed.

The same precision rule applies to contextual numeric text. A leading mean or point is not
an atomic scalar when directional or comparative wording remains in the measurement after
measurement/unit separation (for example, ``62 ± 3% decrease`` or ``2-fold higher than``).
Keep the source-facing display text, set ``finite_scalar_value`` to null, and exclude the row
from pair bucketing. A direction remains scalar-eligible only when the source supplies it as
an explicit controlled unit, such as ``% increase``; in that case the direction is retained
in ``canonical_unit_text`` and therefore in the pair-bucket identity.

Any change to the shared normalizer must be measured per task against real records before
a library is rebuilt on it. A change motivated by one task is not evidence about another.

## Starling direct gold benchmark

Starling evidence ingestion and Starling gold-label construction are separate modules and must not share one interpretation:

```text
tools/chembl_tool/common/starling/evidence_library.py
  Builds inference-time molecule evidence/indexes and does not produce benchmark labels.

tools/chembl_tool/common/starling/benchmark_dataset.py
  Centrally handles parent identity, binary/ambiguous decisions, 70% record-weighted majority,
  historical random/scaffold splits, and audit output.

tools/chembl_tool/common/starling/build_benchmark_datasets.py
  Historical `record_agreement70_split811_v1` random/scaffold build CLI.

tools/chembl_tool/common/starling/build_record_supported_benchmark.py
  Current scaffold-only `record_supported_v2` quality-split builder for Bioavailability and Skin.

tools/chembl_tool/common/starling/build_bbb_experimental_meaningful_cns_access.py
  Current `experimental_meaningful_cns_access_v2` BBB build/audit orchestration.

tools/chembl_tool/tasks/<task>/starling_benchmark.py
  Declares only the task's sources, endpoint/scope/population rules, units/thresholds,
  and conservative free-text-to-label mapping.
```

A task adapter must first map every source record to `0`, `1`, or a rejected/ambiguous decision with a
reason. It must not treat a supporting passage as an unconditional keyword vote or duplicate parent
aggregation or split algorithms. The shared layer aggregates by `rdkit_fragment_parent.v1`; it accepts a
parent when the majority label reaches 70% and is not an exact tie, otherwise recording it in the reject
audit. Multiple accepted records from the same PMID still vote separately.

The current paper-facing roots are:

```text
data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold/
data/processed_starling_record_supported_v2/{Bioavailability_Ma,Skin_Reaction}/scaffold/
```

The current builder first keeps Bemis-Murcko scaffolds disjoint, then constructs train/valid/test using the
frozen lexicographic quality objective. `heldout_molecule_labels.jsonl` is the valid+test exclusion contract
for the train-only retrieval index. Formal evaluation must wait for zero parent and scaffold overlap audits.
`data/processed_starling/<Task>/{random,scaffold}` is the first historical lineage and must not be mixed with
current results. See `tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md` for the full rules,
frozen counts, and commands.

## Molecule identity and parent-disjoint retrieval

Molecule identity is retrieval ranking/filtering metadata and is not part of the evidence semantics in `minimal_evidence.v1`. ChEMBL, Starling, and future sources must all call the shared normalizer. They must not infer same-parent status from source IDs, names, or task labels.

```text
operational:
  Exclude exact whole-record matches and whole-record connectivity variants.
  Retain same-molecular-parent evidence such as salts, solvates, and protonation forms.

parent_disjoint:
  In addition to operational exclusions, exclude the same molecular parent.
  Continue taking candidates from lower ranks, but backfill only above the original min_similarity.
  Fewer than top-k candidates are allowed; the threshold must never be lowered to fill the quota.
```

`same_parent` is allowed only when the primary parents are identical or one primary parent explicitly appears among the other record's mixture components. Arbitrary component-key intersection must not be used, because two unrelated salts could otherwise be classified as the same parent merely because both contain a counterion such as chloride or sodium. Here, parent is an RDKit structure-standardization concept rather than a pharmacological active moiety. Covalent prodrugs, metabolites, and active-moiety relationships are not inferred from the parent key. They must remain structural analogs or be expressed through an independent PK scope annotation.

Historical operational/parent-disjoint sensitivity reruns use stable hashes of the LLM-visible retrieval
contract. A complete run may be reused for an identical sample input; if only some `full_mechanism` families
change, other independent group outputs may be reused before rerunning changed branches and final synthesis.
Every reuse records `reused_from`, `reuse_reason`, and the input hash, and metrics still cover the complete
evaluation subset.

The current paper-facing structural-analog policy is `parent_disjoint`, fresh from a held-out-filtered index.
`operational` is an explicit historical/deployment-sensitivity reference, not a staging dependency for new v4
conditions. Multiple launchers must not be used to bypass the global concurrency budget.

Same-parent exposure audits must distinguish the query-condition, group, neighbor slot, and record deduplicated within a query-condition. A slot represents one LLM-visible appearance of a neighbor in one group. When the same record appears in multiple mechanism groups, count each appearance separately, while also reporting the query-condition-level unique count and rank-1 slot count. This prevents repeated branch exposure from being described as independent molecules.

## Starling system background

In this repository, Starling specifically means the system described in the paper *Self-Driving Datasets: From 20 Million Papers to Nuanced Biomedical Knowledge at Scale*, not other software with the same name. The official description presents Starling as a multi-agent deep-research system for large-scale biomedical literature. Given a natural-language extraction task, it designs corpus-retrieval probes that balance precision and recall, derives a unified extraction schema from sample papers, and then generates structured records with supporting passages and experimental conditions from the retrieved subcorpus.

Official resources:

- Paper: https://arxiv.org/abs/2605.07022
- Code: https://github.com/starling-labs/starling
- Oral Bioavailability example data: https://huggingface.co/datasets/starling-labs/Oral_Bioavailability

The paper reports an underlying corpus of approximately 22.5 million PubMed papers and emphasizes that experimental conditions and supporting passages provide important information beyond traditional tabular databases. Even if the paper reports a low per-extraction cost, complete corpus retrieval, schema induction, extraction, and validation are repeated as the number of tasks grows. Starling acquisition must therefore be treated as an expensive operation in this project.

## Starling acquisition granularity

By default, a Starling acquisition task, prompt, and schema operate at mechanism-family granularity rather than fine-grained endpoint-group granularity.

Specific requirements:

1. Freeze the mechanism families for each task before submitting a Starling acquisition.
2. In principle, run only one Starling task per mechanism family. Do not start a separate Starling task for every endpoint subtype, unit, assay system, species, or formulation within a family.
3. Express within-family differences through extraction-schema fields such as `endpoint_type`, `value`, `unit`, `species`, `assay_system`, `dose`, `formulation`, `comparator`, `support_text`, and `confidence`.
4. After integrating Starling output into TxAgent, fine-grained audit groups may be derived from these structured fields. Those groups do not create new acquisition prompts or independent LLM reasoning branches.
5. A family may be divided into multiple Starling tasks only when it cannot be represented with consistent retrieval semantics and a unified schema, and a sample audit demonstrates material precision/recall degradation. Record the reason for the split and the additional cost in the task-local `AGENTS.md`.
6. Do not create one Starling prompt for every legacy `Tier.endpoint_group` in ChEMBL. The ChEMBL ontology organizes existing heterogeneous assays; the Starling schema should directly acquire condition-rich literature evidence around a mechanism family.

Therefore, Starling data does not need the large number of small acquisition groups used for ChEMBL. It must still retain endpoint subtypes and experimental conditions for auditing, scope decisions, and provenance, but these are within-family fields rather than additional agent branches.

## New-task checklist

Before adding a task or data source, confirm all of the following:

1. The number of mechanism families is small and each family has clear task semantics.
2. The boundary between direct evidence and mechanistic/surrogate evidence is explicit.
3. Every source-local group maps to at most one default mechanism family. Any reuse requires an explanation.
4. `full_flat` and `full_mechanism` use exactly the same evidence union.
5. The number of parallel group-reasoning branches equals the number of enabled mechanism families, not the number of underlying endpoint groups.
6. By default, the number of Starling tasks equals the number of mechanism families that require acquisition and does not grow with the number of endpoint subtypes.
7. The schema preserves values, units, experimental conditions, scope, supporting passages, quality/uncertainty, and provenance.
8. Missing SMILES, structure-standardization failures, duplicate source molecules, and unclassifiable records all have auditable statistics.

The experimental distance-expansion workflow has an additional same-molecule relevance publication gate.
Its scope, statuses, and integration contract are documented in
`tools/chembl_tool/common/DISTANCE_EXPANSION_SELF_RELEVANCE.md`.

## Current ChEMBL-specific implementation

ChEMBL keeps fine-grained `Tier.endpoint_group` labels in its evidence index. In `full_mechanism` and
`full_flat` retrieval, task configuration maps those labels into a smaller number of mechanism families
before candidate molecules are combined, ranked, and limited to top-k. In `native` mode, this mapping is
bypassed and retrieval remains one group per `Tier.endpoint_group`; treat that mode as legacy/debug behavior.

`exclude_source_groups` is an exact-name retrieval filter. It does not delete records from the evidence
library. It is currently used only to remove `Tier 1.context_dependent` from the BBB and Skin Reaction
ChEMBL `direct` views. It is not a global default and does not currently exclude context-dependent groups
from their `full_mechanism` views. Bioavailability uses explicit source-group allow-lists, while ClinTox has
no context-dependent exclusion. Any new paper-facing ChEMBL configuration must explicitly exclude such a
group or map it to a predeclared background/context family.
