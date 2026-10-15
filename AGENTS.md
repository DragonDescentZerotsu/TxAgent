# TxAgent working context

This checkout is authoritative under `/vast/projects/myatskar/design-documents/joseph/TxAgent`.
Check `hostname` and `pwd` before using another checkout. Preserve unrelated
working-tree changes and immutable experiment artifacts.

## Code and filesystem map

TxAgent predicts BBB, Oral, Skin, Ames, DILI and Carcinogens from frozen molecular evidence.
Flow: benchmark queries → retrieval cache → frozen context selection → evidence hydration → query prior + flat prompt → shared model client → checkpoints, predictions, metrics and traces.
Paths below are relative to this checkout; exact inputs come from the selected manifests.

| Piece | Location and responsibility |
| --- | --- |
| Prediction harness | `predict/harnesses/branches/`: `__main__.py` routes the CLI; `flat.py` assembles evidence and renders prompts; `runner.py`, `runtime.py` and `scheduler.py` execute/resume requests; `matrix.py` runs matrices. |
| Prediction prompts | `predict/prompts/`: six `tasks/<task>/profiles.yaml`, shared `flat/final_full_flat/` and `query_prior/` templates, and explicit `variants/`. |
| Task behavior and priors | `predict/tasks/`, `predict/harnesses/branches/tasks/`, and `six_task_query_priors.py` / `tdc_query_priors.py` in the harness directory. |
| Retrieval implementation | `predict/retrieval/assay_reranking/`: builders, scoring, readers, YAML cache configurations and `flat_v5_manifests/`; its `prompts/` are assay-transfer scoring templates. |
| Retrieval cache payloads | `data/caches/assay_reranking/active/flat_v5/<benchmark>/<task>/<role>/<method>/<release>/<variant>/`: release indexes and per-level databases; verification receipts live in `data/caches/assay_reranking/active/flat_v5/receipts/`. |
| Disjoint retrieval | `predict/retrieval/policies.py` owns parent/scaffold identity policies; named cache releases implement them. Example below the cache root: `tdc/dili/l2plus/`. |
| Benchmark splits | Gold: `data/gold_labels/<Task>/<release>/scaffold/`; TDC: `data/gold_labels/TDC/<Task>/`, configured by `data/gold_labels/TDC/manifest.json`; resolvers live in `data/processing/gold_labels/conditioned_benchmark.py`. |
| Experimental evidence | `data/evidence_libraries/<task>/<release>/` owns canonical records, level maps and retrieval projections; `data/processing/` owns construction code. |
| Context selection | `optimization/selection/`: selection algorithms and objectives; `optimization/semantic_buckets/`: canonical construction, scoring, reviewed grouping, weights and ledgers; `optimization/dili_score/`: offline record scoring. |
| Model transport | `predict/api_client/transport.py` creates SDK clients; `client.py` handles model requests; `pool.py` and `providers/` own provider selection, routing, capacity and reasoning settings. |
| Molecular tools | `tools/service/`: deterministic molecule and pair tools; `predict/tools/` consumes their schemas. |
| Run artifacts | `outputs/predictions/<harness>/<prompt>/<run-name>/<benchmark>/<task>/<split>/<profile>/`: shared batch manifests, per-query checkpoints, retrieval snapshots, predictions, metrics and traces. Historical raw locators resolve through `outputs/predictions/relocations.json`; published FINAL remains under the Joseph root. Honor explicit batch/trace roots. |
| Supporting work | `predict/baselines/`: baselines; `predict/traces/`: trace utilities/viewer; `analysis/`: independent studies; `outputs/analysis/`: derived reports; `docs/paper/`: manuscripts/tables. |

Semantic code and artifacts live only under `optimization/semantic_buckets/`, imported as `optimization.semantic_buckets`. Use canonical paths in code, metadata and documentation; do not recreate a root alias.

## Coding philosophy

Before modifying or reviewing code, read [.agents/skills/simplicity-first/SKILL.md](.agents/skills/simplicity-first/SKILL.md).

- Trace the actual callers and data flow before editing; make CLI → records → selection/prompt → metrics understandable.
- Favor code a human can read in one pass. Use the fewest lines, concepts and files that preserve that clarity.
- Reuse the existing shared implementation first, then the standard library or an established scientific library.
- Inline simple single-use logic; avoid unnecessary wrappers, classes, configuration and abstractions for hypothetical future needs.
- Fix the cause where callers share it; preserve useful capabilities, performance controls and observability.
- Keep a small shared Python implementation for each distinct capability. Expose meaningful experiment variations through CLI arguments; instantiate specific experiments as Bash recipes under `experiments/<capability>/`.
- Share code when the algorithm and execution behavior are the same. Full and flat harnesses, query-prior generation, selection, scoring, evidence construction and specialized analyses may keep separate commands. Reuse their common mechanisms without forcing different scientific operations into one dispatcher.
- Bash defines what to run: inputs, parameter combinations, provider configurations, dependencies and output identities. Python defines how it works: algorithms, validation, scheduling, aggregation, checkpointing and artifact handling. Do not move scientific logic into Bash to reduce Python line counts.
- Conditioned baselines use the shared Morgan KNN, MiniMol embedding KNN and MiniMol head commands, instantiated by `experiments/baselines/conditioned.sh`. Preserve parent deduplication, condition-first/null fallback, train-derived head categories, benchmark identity and condition/coverage diagnostics. Recipes set policies explicitly; generic defaults and separate train-only CV contracts stay intact.
- Expose the smallest useful interface. Prefer existing implementations, direct arguments and task policies over a configuration framework, plugin system, global mutation or forwarding wrappers. Generalize from actual experiments, not hypothetical future callers; specialized analyses can remain separate when that is clearer.
- Simplification preserves performance controls, concurrency, resumability, provenance and diagnostics. Preserve distinct scientific contracts unless explicitly retired; Bradley–Terry ranking is retired. Fewer files do not justify tangled conditionals, weakened validation or changed scientific meaning.
- Judge consolidation by removed executable duplication, understandable Bash recipes, explicit scientific methods and focused behavioral equivalence checks. Sometimes parameterize one function; sometimes keep two capabilities separate. Minimize experiment machinery rather than the concepts that genuinely exist in the science.
- Semantic scoring has one canonical batched-first-pass, repeated sorted sliding-window implementation with locked anchors, manual/automatic chains, configurable overlap and replica averaging, and shared local/OpenRouter transport.
- Direct/L1 uses normalized parents only for filtering; rank and score full molecules, count distinct canonical context cards, and preserve authoritative binary card labels. Indirect selection remains joint across L2+ and uses ledger-bound semantic weights. Exact assay score identity is canonical.
- Direct/L1 candidates are always scaffold-disjoint. Only indirect L2+ offers parent-disjoint or scaffold-disjoint cache sets; scaffold-disjoint is the default. Assay-transfer contrastive selection is retired; keep ordinary assay-transfer ranking and frozen optimizer selections.
- Every context optimizer returns ordered physical `source_row_uid` selections, including L1 context members. Every evidence-consuming predictor accepts manual UIDs or the same selection file. Both optimizer and predictor reject candidate-universe, membership and disjointness violations before model calls.

## Runtime and scientific safeguards

- Resolve Gold/evidence through `CURRENT` and release manifests; TDC uses its single canonical manifest. `data/artifacts/` is audit-only.
- Use `predict/api_client/pool.py` and provider JSONs; never create another OpenRouter client or copy secrets into code or artifacts.
- Joseph defaults to fixed `deepseek/deepseek-v4-flash-0731`, automatic provider routing, `deepseek_v4_flash_0731_openrouter_auto_128_high.json`, and high reasoning effort. Keep specialized pinned contracts, including V10 measurement-resolution low effort.
- Inspect provider JSON before launch; launcher-local capacity comes from `max_inflight`. Record actual served model/provider. Changed model or route requires a new run identity.
- Launch an approved full batch directly in throughput mode; a separate pilot must never gate it.
- Prediction contracts live once in `prediction_batch.v2` or `joseph_final_arm.v3` batch manifests. Store named benchmark/evidence/map/weight versions, effective scientific parameters, model/prompt versions, ordered query UIDs and sparse genuine query overrides. Do not write per-query manifests or use paths/content fingerprints as experiment identity. Keep query stage state in `stage_resume.json`; use the shared manifest readers and existing runtime resolvers.
- Unknown historical versions and UIDs stay empty and block unsafe reuse; never fill them from today's `CURRENT`. A new scientific configuration needs a new run identity. Metadata format migrations may update historical headers/indexes when explicitly authorized; predictions, evidence, requests/responses, traces, scores and selected releases remain frozen. Compressed FINAL bundles change only for an explicitly requested push.
- Use manifests, receipts and indexed reads for development. Large hashes and SQLite integrity checks belong to publication, restore or explicit verification. Fail on incompatible inputs before scoring or model calls.
- Keep checks proportional; use focused behavioral tests and reviewed renderings/evaluations for prompt prose.
- New raw runs use `outputs/predictions/` with an explicit `--run-name`; harness and prompt namespaces come from the selected configuration. Run names label experiments; manifests remain the scientific authority. Derived comparisons use `outputs/analysis/<domain>/<study-id>/`. Never rewrite completed scientific payloads; approved relocation preserves bytes and run UIDs, with an operational locator map and manual classification for unknown identities. Transient work goes in `/tmp`; new reports use English.
- Resume compares the requested query prior with the saved validated content, even when source run UIDs match. A changed or unverifiable reusable prior raises before model calls or checkpoint writes; never replace it or silently rerun under the same identity.
- FINAL publication freezes parameters, exact prompts, results, provider provenance and per-query traces before updating indexes; verify new arms and preserve existing entries. Rebuild `final/git_bundle/` once only for an explicitly requested push.

Scientific contracts: [inference](predict/README.md), [TDC mixed selection](optimization/TDC_CONTRACT.md),
[reranking caches](predict/retrieval/assay_reranking/CACHE_LAYOUT.md), and [semantic buckets](optimization/semantic_buckets/README.md).
FINAL publication: [.agents/skills/finalizing/SKILL.md](.agents/skills/finalizing/SKILL.md) and [finalize.md](docs/finalize.md).
Selected FINAL scores and caveats: [docs/results.md](docs/results.md); do not replace selected cells with the highest observed score.

Prediction and query-prior prose/schema belong in `predict/prompts/`, with all six task profiles present. Python owns rendering and execution. TDC has no version switch: Oral/Skin use the frozen scaffold cohorts and the other four tasks use frozen original-source cohorts. Read cache and selection inputs from its canonical manifest. Historical labels and relocation metadata live in `data/legacy/tdc/`; never reinterpret their identities. Full-molecule optimizer L1 cards are distinct from frozen parent-grouped retrieval caches; parent scores cannot be reused as card scores.
