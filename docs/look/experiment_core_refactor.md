# Experiment core refactor review

The implementation keeps a shared Python core for each distinct scientific
capability and uses Bash recipes for concrete experiments. The full philosophy,
commands and data contracts are in [experiment_code.md](../experiment_code.md).

## Measured reduction

Counts compare this implementation with the source snapshot taken immediately
before it, including the user's existing working-tree edits. They do not compare
against Git HEAD or count earlier cleanup as part of this refactor. Counts are
physical Python source lines, including comments and blank lines; frozen data,
outputs, prompt assets and historical provenance are excluded.

- Production Python: -9,017 lines net; 19 files retired and 6 modules added (13 fewer files).
- Including tests: -10,238 lines net; 24 files retired and 8 added.
- The canonical scorer replaces two older general scoring engines. The
  MiniMol grid exists once in Bash instead of in both Python and Bash.
- API callers now use one SDK transport owner, with shared provider routing.

## Maintained additions and their purpose

| File | Purpose |
| --- | --- |
| `predict/api_client/transport.py` | Existing SDK/credential behavior moved to the API owner; local and remote transports share it. |
| `predict/retrieval/selection.py` | One physical-UID reader, candidate/disjointness validator and optimizer exporter. |
| `predict/baselines/manual_evidence.py` | Shared manual L1 voting path used by three existing baseline CLIs. |
| `predict/baselines/minimol/summarize_sweep.py` | Validation-based summary; training and scheduling remain in their own owners. |
| `optimization/semantic_buckets/bucket_cards.py` | Pure bucket-card helpers still needed by grouping and eligibility after removing ranking. |
| `optimization/semantic_buckets/legacy_weight_io.py` | Read frozen V5/V6 weight ledgers without retaining their inference engines. |

`official_two_pass_weights.py` owns batched Pass 1 and sorted, locked-anchor
Pass 2. Its manual/automatic chain controls, averaging, provider mixes, receipts
and resume are shared. New optimizer manifests include physical L1 voters and
L2+ UIDs while retaining the original scientific context tables and objective
provenance. Prediction and evidence baselines use the same validation.

The L1-context and pair-bucket analyses expose their study settings as CLI
arguments. Existing specialized diagnostics and V7–V10 scientific construction
stages retain their distinct contracts. Root and all eight owner `AGENTS.md`
files carry the corresponding philosophy and scope rules.

## Retired source and tests

These files provided superseded scoring/ranking execution, duplicate transport
or sweep launching, or tests dedicated to those removed implementations.
Their frozen artifacts and prompt assets are preserved. Useful card and ledger
helpers remain with the maintained owners above.

- `analysis/evidence_library/bioavailability_relevance_bucket_levels.py`
- `analysis/evidence_library/bioavailability_relevance_bucket_levels_v3.py`
- `analysis/evidence_library/bioavailability_relevance_bucket_tournament.py`
- `analysis/evidence_library/relevance_bucket_diagnostic.py`
- `analysis/evidence_library/relevance_bucket_levels.py`
- `analysis/evidence_library/relevance_bucket_luna.py`
- `analysis/evidence_library/relevance_bucket_pilot.py`
- `analysis/evidence_library/relevance_bucket_rounds.py`
- `analysis/evidence_library/relevance_bucket_tournament.py`
- `analysis/evidence_library/skin_reaction_relevance_bucket_tournament.py`
- `data/processing/llm_api.py`
- `optimization/semantic_buckets/bbb_semantic_degree25.py`
- `optimization/semantic_buckets/bbb_semantic_degree25_v2.py`
- `optimization/semantic_buckets/bioavailability_semantic_degree25.py`
- `optimization/semantic_buckets/bioavailability_semantic_degree25_v2.py`
- `optimization/semantic_buckets/calibrate_weights_v6.py`
- `optimization/semantic_buckets/publish_gold_v1_incremental.py`
- `optimization/semantic_buckets/sliding_weight_assignment.py`
- `optimization/semantic_buckets/tests/test_bioavailability_semantic_degree25.py`
- `optimization/semantic_buckets/tests/test_calibrate_weights_v6.py`
- `optimization/semantic_buckets/tests/test_sliding_weight_assignment.py`
- `predict/baselines/minimol/run_hparam_sweep.py`
- `tests/analysis/evidence_library/test_relevance_bucket_levels.py`
- `tests/analysis/evidence_library/test_relevance_bucket_tournament.py`

## Validation

- Shared API/transport and canonical scoring: 43 passed, including offline
  preparation, multi-task chain allocation, locked anchors, fixed provider
  mixes, exact score coverage, resume and drift rejection.
- Flat matrix, prompt, preselection and TDC inference contracts: 117 passed.
- Native UID retrieval checks: 71 passed in the tokenizer environment (one
  Apricot-dependent integration skipped there).
  The added real optimizer-build/export/hydration integration passed separately
  in the environment with Apricot.
- Optimizer objectives and TDC ledger checks: 45 passed.
- Full semantic suite: 135 passed; one existing function-length check failed.
  Its three unchanged violations are `artifacts.py`'s
  `load_skin_candidate_snapshot` (63 lines), `load_dili_candidate_snapshot`
  (63), and `_cached_record_weights` (61). The pre-implementation snapshot has
  the same violations.
- MiniMol Bash sweep: fixture check passed for per-GPU serialization, all eleven
  configurations, cache skips, argument quoting, summaries and failed jobs.
- Both analysis fixtures, manual UID/export fixtures, recipe Bash syntax, CLI help,
  changed-source compilation and new unresolved-global checks passed.

No paid requests, live inference, GPU training or active-release publication
were launched. Completed scientific runs and published artifacts were not
rewritten. Git's partial-clone diff check could not fetch a required remote
object; source checks and reduction measurements use the local pre-change
snapshot instead.
