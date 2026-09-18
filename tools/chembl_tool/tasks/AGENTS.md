# Molecular Evidence Task Extension Conventions

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

This file applies to all existing and future tasks under `tools/chembl_tool/tasks/`. Task directories may add their own data paths, endpoint semantics, and run commands, but must not break the common layering defined here.

## Two-layer evidence classification

Each task must clearly distinguish the following two layers, and must not implement both as LLM reasoning branches.

### Mechanism family

Mechanism family is a task-level, data-source-independent semantic category and the basic unit of retrieval and parallel reasoning. For example, Bioavailability_Ma uses Observed direct F, Observed oral exposure, Fa, Fg, Fh; BBB uses direct exposure, passive permeability, efflux, influx.

Requirements:

- Before integrating ChEMBL, Starling, or other data sources, define a small number of stable mechanism families based on task mechanism.
- `full_mechanism` Each mechanism family corresponds to at most one parallel group reasoning branch.
- `direct` Usually only the direct-result family is enabled; `full_flat` uses the same evidence union as `full_mechanism`, but merges into one reasoning branch.
- There is no fixed cross-task number of families, but restraint is required; typically these should be a small number of interpretable categories, not dozens of endpoint branches.
- Family definitions must not depend on which assays a data source currently happens to have, nor include benchmark label policy, voting rules, or deterministic overrides.

### Source-local endpoint group

Source-local endpoint group is the data integration, semantic normalization, filtering, and audit label. For example, in ChEMBL, endpoints such as AUC, Cmax, Papp, efflux ratio, hERG, and Ames must be distinguished because they cannot be directly mixed.

Requirements:

- Fine-grained groups may be retained in evidence rows, indexes, and provenance for detecting misclassification, checking coverage, and explaining source differences.
- Multiple fine-grained groups must first map to one mechanism family; at query time, candidate molecules are merged within the family before top-k retrieval.
- When the same molecule appears in multiple fine-grained groups within one family, it occupies only one neighbor slot and carries the relevant records under that family.
- Fine-grained groups must not each start parallel LLM reasoning. Otherwise, this causes evidence fragmentation, duplicate neighbors, token cost inflation, and final synthesis overload.
- Records that cannot be reliably classified into `context_dependent` or an equivalent bucket do not enter the paper-facing retrieval view by default. Only semantically clear, pre-declared background/context families may enter reasoning.

Therefore, the old fine-grained ontology may continue to exist, but it belongs to the source adapter and audit layer, not the expert reasoning policy in the paper method. The paper and user documentation should describe it as `source-local endpoint normalization`, and describe mechanism families as the agent's reasoning organization.

## Configuration location for new tasks

When adding a new task, follow the following responsibility division:

```text
tools/chembl_tool/tasks/<task>/experiment_config.py
  Declares data-source-independent mechanism families and the mapping from each source group to a family.

tools/chembl_tool/tasks/<task>/endpoint_groups.py
  Responsible only for fine-grained endpoint normalization and audit labels of heterogeneous raw data such as ChEMBL.

tools/chembl_tool/common/evidence_contract.py
  Converts different sources into minimal_evidence.v1; does not do label prediction.

tools/chembl_tool/common/experiment_retrieval.py
  Merges candidates by mechanism family, retrieves neighbors, and constructs direct, full_flat, and full_mechanism views.

tools/chembl_tool/common/molecule_identity.py
  Uses versioned RDKit FragmentParent to normalize whole records, fragment/molecular parents, and mixture components.

tools/chembl_tool/common/retrieval_policy.py
  Uniformly defines operational and parent_disjoint candidate exclusion policies; tasks and source adapters must not duplicate this logic.

tools/chembl_tool/common/neighbor_selection.py
  Performs pluggable top-k set selection on candidates that have already passed similarity threshold, identity policy, and evidence-availability checks. `similarity` retains historical pointwise Tanimoto ranking; `query_feature_coverage` greedily maximizes marginal union coverage of query Morgan bits without degrading existing `min_similarity`, and uses Tanimoto only as a tie-break for tied candidates. The selector must not modify evidence rows, mechanism-family mappings, or the downstream retrieval JSON payload schema.

tools/chembl_tool/common/retrieval_ablation.py
  Computes a stable hash of LLM-visible sample/family input, materializes reuse of the entire run or independent branches, and writes provenance.

tools/chembl_tool/common/retrieval_replay.py
  Reads frozen retrieval artifacts and validates query identity for strict matched-prefetch replay; must not re-retrieve against the current index.

tools/chembl_tool/common/identity_blind.py
  Uniformly implements identity redaction, harness prefetch, and visible prefetched-tool replay; task runners must not fork this logic themselves.

tools/chembl_tool/common/PROMPT_PROFILE_CONTRACT.md
  Uniformly freezes shared reasoning payload, structured-output validation, task prompt profile, final-decision profile, manifest provenance, and branch-reuse gate; task-local profiles maintain only the task's semantics, schema, and cross-field rules.

tools/chembl_tool/tasks/<task>/run_reasoning_pipeline.py
  For external workflows that need to reuse task-specific full-flat prompts, exposes `build_group_prompt_payload()`; RL/data materializers must not call the task-internal `_group_prompt_payload()` private implementation. Historical private aliases are used only for backward compatibility with old callers.
```

Task code must not duplicate common retrieval, source aggregation, LLM client, validation, or batch orchestration.

## Starling direct gold benchmark

Starling evidence ingestion and Starling gold-label construction are two independent modules and must not share a single meaning:

```text
data/processing/evidence_library/evidence_library.py
  Builds inference-time molecule evidence/index; does not produce benchmark labels.

data/processing/gold_labels/benchmark_dataset.py
  Uniformly handles parent identity, binary/ambiguous decision aggregation, 70% record-weighted majority, historical random/scaffold splits, and audit outputs.

data/processing/gold_labels/build_conditioned_benchmark.py
  First-version `record_agreement70_split811_v1` random/scaffold historical build CLI.

data/processing/gold_labels/build_record_supported_benchmark.py
  Bioavailability/Skin molecule-only source-lineage builder; not an active evaluation entry point.

data/processing/evidence_library/versions/v7/tasks/bbb_martins/experimental_meaningful_cns_access_benchmark.py
  BBB source voting/build audit orchestration; not an active evaluation entry point.

data/processing/gold_labels/publish_conditioned_benchmark.py
  Publishes the four-task source builds as the single Conditioned Benchmark and fixes migration/hash audit.

tools/chembl_tool/tasks/<task>/starling_benchmark.py
  Declares only the task's source, endpoint/scope/population, units/thresholds, and conservative free-text-to-label mapping.
```

Task adapters must first map each source record to `0`, `1`, or a rejected/ambiguous decision with a reason; must not treat supporting passages as unconditional keyword votes, nor duplicate parent aggregation or split algorithms within the adapter. The common layer aggregates accepted records according to `rdkit_fragment_parent.v1`; a majority label is accepted when it reaches 70% and is not an exact tie, otherwise it is written to the reject audit. Multiple accepted records from the same PMID still count separately.

Current paper-facing roots are:

```text
data/gold_labels/{BBB_Martins,Bioavailability_Ma,Skin_Reaction}/v1/scaffold/
data/legacy/clintox/gold_labels/conditioned_benchmark/scaffold/
```

The current builder first ensures Bemis–Murcko scaffolds do not cross splits, then builds train/valid/test according to a frozen lexicographic quality objective. `heldout_molecule_labels.jsonl` is the train-only retrieval-index exclusion contract for the valid+test union; formal evaluation must not start until both parent overlap and scaffold overlap audits are zero. Old molecule-only, selected-vN, and `data/gold_labels/legacy/processed_starling/<Task>/{random,scaffold}` serve only as source provenance and must not be used as runner inputs or mixed into tables with current results. Full rules, current frozen counts, and run commands are in `tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`.

## Molecule identity and parent-disjoint

Molecule identity is retrieval ranking/filtering metadata, not part of the evidence semantics of `minimal_evidence.v1`. ChEMBL, Starling, and future data sources must all call the common normalizer and must not determine same-parent on their own based on source IDs, names, or task labels.

```text
operational:
  Excludes exact full-record matches and whole-record connectivity variants;
  retains same molecular parent evidence such as salts, solvates, and protonation forms.

parent_disjoint:
  Additionally excludes same molecular parent on top of operational;
  continues to take candidates from further down the ranking, but only backfills above the original min_similarity;
  if candidates are insufficient, allows fewer than top-k, and must never lower the threshold to fill.
```

`same_parent` only allows the main parent to be the same, or one party's main parent to explicitly appear in the other party's mixture components. Arbitrary component key intersection is not allowed; otherwise two unrelated salts would be incorrectly grouped as the same parent because they share counterions such as chloride or sodium. Here, parent is an RDKit structural normalization concept, not a pharmacological active moiety. Covalent prodrugs, metabolites, and active-moiety relations are not inferred from parent keys; they must be retained as structural analogs or expressed by separate PK scope annotations.

Selective reruns of historical operational/parent-disjoint sensitivity ablations are based on the stable hash of the LLM-visible retrieval contract. When sample inputs are exactly identical, the entire run may be reused; when only some families change in `full_mechanism`, other independent group outputs may be reused, and only the changed branches and final are rerun. All reuse must record `reused_from`, `reuse_reason`, and input hashes; final metrics are still computed on the full evaluation subset.

The current main policy for paper-facing structural-analog retrieval is `parent_disjoint`, and it is freshly run directly from the held-out-filtered index. `operational` is only an explicit opt-in historical/deployment-sensitivity reference; it is no longer a staging dependency for new v4 conditions, and must not bypass the global concurrency budget through multiple launchers.

The same-parent exposure audit must distinguish query-condition, group, neighbor slot, and deduplicated records within a query-condition. A slot represents one LLM-visible occurrence of a neighbor in a group; when the same record appears in multiple mechanism groups, it is counted separately, while also reporting the unique count within the query-condition and the number of rank-1 slots, to avoid misreporting branch duplicate exposure as independent molecule counts.

## Starling system background

Here, Starling specifically refers to the Starling in the paper *Self-Driving Datasets: From 20 Million Papers to Nuanced Biomedical Knowledge at Scale*, not other software with the same name. In the official description, Starling is a multi-agent deep research system for large-scale biomedical literature: given a natural language extraction task, it designs corpus retrieval probes balancing precision/recall, induces a unified extraction schema from sample literature, and then generates structured records with supporting passages and experimental conditions on the retrieved sub-corpus.

Official resources:

- Paper: https://arxiv.org/abs/2605.07022
- Code: https://github.com/starling-labs/starling
- Oral Bioavailability example data: https://huggingface.co/datasets/starling-labs/Oral_Bioavailability

The paper reports its underlying corpus contains approximately 22.5M PubMed papers and emphasizes that experimental conditions and supporting passages are a significant increment over traditional tabular databases. Even if the paper reports a low per-extraction cost, the full corpus retrieval, schema induction, extraction, and validation are repeated with the number of tasks; in this project, Starling acquisition should be treated as an expensive operation.

## Starling acquisition granularity

Starling defaults to mechanism family as the granularity of acquisition task/prompt/schema, not fine-grained endpoint groups.

Specific requirements:

1. Each task freezes mechanism families before submitting Starling acquisition.
2. In principle, only one Starling task is run per mechanism family. Do not start separate Starling tasks for each endpoint subtype, unit, assay system, species, or formulation under a family.
3. Differences within the same family are expressed through extraction schema fields, such as `endpoint_type`, `value`, `unit`, `species`, `assay_system`, `dose`, `formulation`, `comparator`, `support_text`, and `confidence`.
4. After Starling output is integrated into TxAgent, fine-grained audit groups may be derived from these structured fields, but these groups do not generate new acquisition prompts or independent LLM reasoning branches.
5. Only when a family cannot be expressed with consistent retrieval semantics and a unified schema, and sampling audits prove that precision/recall are significantly impaired, is it allowed to split into multiple Starling tasks; the reason for the split and additional costs must be recorded in the task-local `AGENTS.md`.
6. Do not copy every old Tier.endpoint_group from ChEMBL into Starling prompts one by one. The ChEMBL ontology is used to organize existing heterogeneous assays; the Starling schema should directly acquire conditioned literature evidence around mechanism families.

Therefore, Starling data does not need to maintain a large number of fine-grained groups as acquisition units like ChEMBL. It must still retain endpoint subtypes and experimental conditions for audit, scope judgment, and provenance, but this information is fields within a family, not additional agent branches.

## Checklist for new tasks

Before adding a new task or new data source, confirm:

1. The number of mechanism families is small and has clear task semantics.
2. The boundary between direct evidence and mechanistic/surrogate evidence is clear.
3. Each source-local group maps to at most one default mechanism family; if reuse is needed, the reason must be explained.
4. `full_flat` and `full_mechanism` use exactly the same evidence union.
5. The number of parallel group reasoning branches equals the number of enabled mechanism families, not the number of underlying endpoint groups.
6. The number of Starling tasks defaults to the number of mechanism families to acquire, and does not grow with the number of endpoint subtypes.
7. The schema retains numeric values, units, experimental conditions, scope, supporting passages, quality/uncertainty, and provenance.
8. Missing SMILES, structural normalization failures, duplicate source molecules, and unclassifiable records all have auditable statistics.
9. Mechanistic/surrogate evidence passes the same-molecule causal continuity gate: trace the causal subject from the assay molecule along the reasoning path; do not automatically write "this molecule changes the system state" as "another molecule subsequently transported/metabolized/harmed by that system is itself."
10. If the conclusion also requires the query molecule to have additional roles such as transporter substrate, enzyme substrate, metabolic precursor, target engagement, or sensitizer, that role must be explicitly supported by retrieval evidence for the same molecule; the LLM must not guess based on pathway common sense. If not satisfied, mark as `requires_query_role`; if it only affects other molecules/systems, mark as `context_only`; neither may enter the main H1/H2 retrieval.

## Same-molecule causal continuity

This check is orthogonal to graph hops, `scope_match`, and `quality_status`. An edge can have the correct direction, reliable literature, and high-quality assay, but still be unsuitable for predicting the assay molecule's own task label. For example:

```text
molecule A activates NRF2/AhR
  -> barrier P-gp abundance increases
  -> known probe substrate B has lower brain accumulation
```

If the evidence does not prove `A is a P-gp substrate`, the last step cannot be used to predict A's own BBB disposition. Similarly, `HIF-1 -> GLUT1 abundance -> glucose uptake` cannot be used for any HIF perturbagen's own BBB influx unless the same molecule has separate GLUT1 substrate evidence.

All future distance-expansion tasks must generate a machine-readable `FamilySelfRelevanceAudit` for each measurement family and call before publishing the graph:

```python
validate_self_relevance_audit(config, audits, require_publishable=True)
```

`requires_query_role`, `context_only`, and `unresolved` may be retained in candidate/audit artifacts, but must not pass the publication gate. Prompt disclaimers cannot replace missing molecule-role evidence.
