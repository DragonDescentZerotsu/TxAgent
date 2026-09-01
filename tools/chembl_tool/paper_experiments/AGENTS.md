# Paper experiment operating contract

This directory owns reusable paper orchestration. Task-specific endpoint
mapping, scientific label policy, prompt schema, and prediction fields remain in
`tools/chembl_tool/tasks/<task>/`. Do not create a task-local copy of a shared
runner, retrieval engine, provider client, validator, analysis, or plotter.

## Active paper scope

Maintain only these result families:

1. ChEMBL retrieval;
2. Starling retrieval;
3. one-shot cumulative full-flat levels;
4. append-only progressive levels that see prior state and evidence;
5. full-mechanism organization over the same evidence as full-flat;
6. scaffold and parent-grouped random split evaluation.

Both `identity_blind` and `deployment_visible_prefetched` must remain supported.
`deployment_visible` may remain as the end-to-end deployment mode. Router and
RL are retained only as isolated archived/stopped research; they are not active
paper entrypoints. Removed no-go method branches must not be reintroduced into
the maintained matrix.

## Isolated historical research

The following packages remain reproducible but are outside the active paper
dependency graph:

```text
router_oof/
rl_lora/
```

Active runners, result registries, and default documentation must not import or
launch them. Their own README/plan files are the only operational documentation
for those stopped branches.

## Canonical data

All active runners and baselines read:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

Never hard-code a historical molecule-only, `selected_vN`, task gold-vN, or
source-build path as an evaluation input. Import public roots from
`tools.chembl_tool.common.starling.conditioned_benchmark`.

Benchmark creation is part of the paper method and must remain reproducible.
The authoritative voting, parent/condition, split, vote-support priority,
leakage, repair, and audit requirements are in:

```text
tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md
data/conditioned_benchmark/manifest.json
data/conditioned_benchmark/migration_receipt.json
data/conditioned_benchmark/random_split_receipt.json
```

Do not simplify the random split to independent row shuffling. It assigns whole
parent groups, has exact 80/10/10 row counts and three-way condition coverage,
and for BBB/Bioavailability/Skin lexicographically prioritizes multi-vote rows
for valid/test before balancing labels and conditions. ClinTox uses source-role
labels and therefore skips the vote-count objective.

## Maintained entrypoints

```text
molecular_evidence_agent.py
starling_benchmark_matrix.py
run_conditioned_assay_family_curve.py
run_conditioned_assay_progressive_curve.py
build_starling_benchmark_indices.py
build_assay_family_catalog.py
build_conditioned_source_family_purity.py
build_bbb_source_family_purity.py
build_bioavailability_vote_pure_source.py
summarize_starling_benchmark.py
analyze_starling_direct_significance.py
analyze_starling_best_agent_baselines.py
analyze_progressive_trace_adoption.py
plot_starling_model_comparison.py
plot_assay_retrieval_curve.py
```

The source/reasoning matrix owns ChEMBL versus Starling and direct/flat/
mechanism conditions. The family runner reasons over each cumulative level from
scratch. The progressive runner appends evidence and previous state. Their
manifests and output roots must remain separate even when they share retrieval
components.

## Experiment axes and reuse

Treat these as independent manifest fields:

- task and benchmark split;
- evaluation subset;
- retrieval source;
- direct/full-flat/full-mechanism/progressive organization;
- visibility mode and tool-execution contract;
- neighbor identity policy;
- benchmark input, held-out index, and family-catalog hashes;
- prompt profile, model identity, reasoning parameters, and output cap.

Never reuse a prediction based only on directory name, query index, molecule
count, or valid-input hash. Reuse requires semantic hash equivalence for every
LLM-visible dependency. A changed prerequisite invalidates downstream final and
trace artifacts. `none` single/final states and evidence-equivalent carry-forward
levels may be reused only with an explicit receipt.

The current family/progressive split policy is `scaffold_disjoint` for scaffold
and `parent_disjoint` for random. Other historical/source matrix experiments
must preserve and report their explicit policy; never infer it from a path.

## Visibility

- `identity_blind`: remove query/neighbor structures, identifiers, and source
  names from LLM messages; harness-prefetched tool evidence remains visible.
- `deployment_visible_prefetched`: replay the blind retrieval and tool payload,
  then restore permitted query/neighbor identity and structure surfaces. This is
  the matched visibility control.
- `deployment_visible`: fresh visible retrieval with model-driven tool use; it is
  an end-to-end deployment experiment, not a pure visibility comparison.

Raw retrieval may retain identities for provenance. Audit the actual messages,
not only the retrieval JSON.

## Execution and validation

- Use one global ready queue; `--parallelism` is the total outstanding prompt
  budget, not a per-task or per-level multiplier.
- Single/group/final or progressive-level checkpoints must be atomic and
  resumable. Do not restart completed compatible work.
- Provider failover may change transport but not model identity or semantic
  prompt settings. Record every provider attempt and served model.
- Shared JSON validation may retry malformed output. It must never repair a
  scientifically valid prediction into a task-preferred label.
- Run valid first. Formal test starts only after the method/prompt/retrieval
  contract is frozen; never tune on test.

## Analysis and documentation

`plot_starling_model_comparison.py` is the only source/model/visibility overview
plotter. `plot_assay_retrieval_curve.py` is the only level-curve and resource
plotter. Extend these interfaces instead of adding a one-off figure module.

After a completed run:

1. verify expected sample/level counts and zero failures;
2. verify benchmark, retrieval, visibility, and model hashes;
3. update `current_conditioned_results.json` first;
4. update `RESULTS.md` and `ASSAY_LEVEL_RETRIEVAL.md` from that registry;
5. update `TRACE_RETENTION.md` if the canonical allowlist changes.

Never call a stale or partial artifact complete. Historical roots may be kept as
one last-complete reference, but their lineage must be explicit and they must
not be paired across incompatible datasets.

## Code and filesystem hygiene

- Prefer shared modules in `tools/chembl_tool/common/`.
- Keep task directories limited to scientific configuration and source review.
- Do not create a new launcher or plotter for one experiment row.
- Keep only current inputs/traces plus one explicitly indexed last-complete
  reference when a current cell is missing.
- Follow `TRACE_RETENTION.md` before deleting artifacts. Use exact paths; never
  recursively clean `outputs/paper`, `data`, or the repository root.
- Historical details belong in Git history or a compact receipt, not thousands
  of lines in active operating documentation.
