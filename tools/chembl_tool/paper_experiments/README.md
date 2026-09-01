# Molecular-evidence agent paper experiments

This directory contains the maintained paper-facing experiment layer. Task
science stays in `tools/chembl_tool/tasks/<task>/`; retrieval, visibility,
reasoning, validation, checkpointing, analysis, and plotting are shared.

## Paper scope

The retained paper matrix has six result families and two visibility controls:

| Question | Retrieval source | Evidence organization | Maintained runner |
|---|---|---|---|
| Does literature-scale Starling retrieval outperform ChEMBL retrieval? | ChEMBL versus Starling | matched direct, full-flat, and full-mechanism views | `starling_benchmark_matrix.py` |
| What happens as progressively more distant evidence becomes available all at once? | Starling | one-shot cumulative full-flat levels | `run_conditioned_assay_family_curve.py` |
| Can the model revise an earlier decision when new indirect evidence is appended? | Starling | append-only progressive state and evidence | `run_conditioned_assay_progressive_curve.py` |
| Does mechanism grouping help relative to one flat evidence branch? | ChEMBL or Starling | full-flat versus full-mechanism over the same evidence | `starling_benchmark_matrix.py` |
| Does evaluation difficulty depend on structural generalization? | matched source and reasoning setup | scaffold versus parent-grouped random split | the same runner with the split flag |
| Does molecular identity visibility change behavior? | matched retrieval payload | `identity_blind` versus `deployment_visible_prefetched` | shared visibility contract and replay |

`deployment_visible` remains available for end-to-end agentic deployment
experiments. The paper-facing matched visibility comparison uses
`deployment_visible_prefetched`, which restores identities/structures while
replaying the blind retrieval and tool payload.

The one-shot family curve is currently identity-blind. The current append-only
progressive curve is deployment-visible-prefetched. Do not describe these two
independent protocol families as a matched visibility ablation. Blind and
visible settings are both retained and supported by the source/reasoning matrix.

## Canonical benchmark

All active evaluation data lives at:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

The two split schemes contain the same molecule-condition rows and labels:

- `scaffold`: parent- and Bemis-Murcko-scaffold-disjoint;
- `random`: exact parent-grouped 80/10/10 allocation; parent-disjoint but not
  scaffold-disjoint, with all conditions represented in all three splits.

The benchmark creation code, task-specific voting boundaries, condition and
parent construction, multi-vote held-out priority, complete lexicographic split
objectives, deletion-repair behavior, leakage policy, and required audit files
are frozen in:

```text
tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md
data/conditioned_benchmark/manifest.json
data/conditioned_benchmark/migration_receipt.json
data/conditioned_benchmark/random_split_receipt.json
```

Historical molecule-only, `selected_vN`, task gold-vN, and source-build names
are provenance only. Runners and plots must use the canonical roots above.

## Maintained entrypoints

### Benchmark and evidence construction

```text
tools/chembl_tool/common/starling/publish_conditioned_benchmark.py
tools/chembl_tool/common/starling/build_conditioned_random_split.py
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/build_assay_family_catalog.py
tools/chembl_tool/paper_experiments/build_conditioned_source_family_purity.py
tools/chembl_tool/paper_experiments/build_bbb_source_family_purity.py
tools/chembl_tool/paper_experiments/build_bioavailability_vote_pure_source.py
```

Task-specific voter and source-review entrypoints are indexed in
`CONDITIONED_BENCHMARK.md`; they are intentionally not duplicated here.

### Model experiments

```text
tools/chembl_tool/paper_experiments/molecular_evidence_agent.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_progressive_curve.py
```

### Analysis and figures

```text
tools/chembl_tool/paper_experiments/summarize_starling_benchmark.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
tools/chembl_tool/paper_experiments/analyze_starling_best_agent_baselines.py
tools/chembl_tool/paper_experiments/analyze_progressive_trace_adoption.py
tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
```

`plot_starling_model_comparison.py` is the single model/source/visibility
comparison figure entrypoint. `plot_assay_retrieval_curve.py` is the single
level-curve/resource figure entrypoint. Do not add a one-off plot module for a
new model row or a new progressive rerun.

### Baselines

```text
baselines/minimol/run_train_cv.py
baselines/minimol/run_embedding_knn.py
baselines/conditioned_knn.py
baselines/structure_knn/run.py
```

Condition-aware MiniMol is the paper-facing trained baseline whenever a task
has accepted condition groups. The two KNN families retain both
same-condition-then-null and unrestricted-train variants.

## Shared contracts

- Retrieval source, evidence organization, split, visibility, model, prompt,
  input, index, and family-catalog identities are independent manifest axes.
- A stored prediction may be reused only when all LLM-visible semantic hashes
  match. Directory names or matching molecule counts are insufficient.
- `none` may reuse a compatible frozen single/final state. Evidence-equivalent
  levels may carry forward without a model call.
- The current cumulative-family/progressive evaluation uses
  `scaffold_disjoint` for scaffold and `parent_disjoint` for random. The
  source/reasoning matrix exposes its identity policy explicitly in the
  manifest; a split name alone never silently changes an old matrix contract.
  Every policy prohibits the query parent itself.
- Evaluation indices remove valid+test benchmark-defining direct rows while
  preserving eligible non-direct mechanism evidence from held-out parents.
- Morgan retrieval uses the configured `0.30` minimum similarity. The current
  progressive protocol also applies its monatomic-element collision guard.
- Identity-blind prompts remove query/neighbor structures, identifiers, and
  source names. Visible-prefetched prompts restore them while holding retrieval
  and tool outputs fixed.
- LLM outputs pass shared JSON validation. Validation retries and provider
  failover are recorded in the trace; they do not alter label policy.

Detailed group-level data flow is in [PIPELINE.md](PIPELINE.md). The cumulative
level and progressive protocols are in
[ASSAY_LEVEL_RETRIEVAL.md](ASSAY_LEVEL_RETRIEVAL.md).

## Current artifact registry

The only machine-readable index for paper-facing result roots and freshness is:

```text
tools/chembl_tool/paper_experiments/current_conditioned_results.json
```

It distinguishes:

- `current`: benchmark and retrieval hashes match;
- `last_complete_reference_pending_*`: a complete historical reference is
  retained because the corresponding current-condition matrix is incomplete;
- `stale_retrieval_index_requires_targeted_replay`: labels may still match, but
  changed retrieval inputs prohibit publishing the old score as current.

At the 2026-09-01 snapshot, BBB and Skin progressive scaffold/random artifacts
match their current indices. Bioavailability requires targeted progressive
replay after the reviewed nitrendipine source-identity correction. Its stored
scores remain a clearly marked pre-fix reference, not a current result.

## Documentation map

- [Conditioned benchmark contract](../common/starling/CONDITIONED_BENCHMARK.md)
- [Pipeline and visibility](PIPELINE.md)
- [Assay/family-level and progressive retrieval](ASSAY_LEVEL_RETRIEVAL.md)
- [Current result status](RESULTS.md)
- [Trace retention](TRACE_RETENTION.md)
- [MiniMol and KNN baselines](../../../baselines/minimol/README.md)
- [ClinTox source contract](../tasks/clintox/CLINTOX_BENCHMARK.md)

Historical experiments that are not part of the paper scope must not be linked
as active entrypoints from this page. Git history and explicitly retained
receipts provide provenance for removed no-go launchers and transient smoke
runs.
