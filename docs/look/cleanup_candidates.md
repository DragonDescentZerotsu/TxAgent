# Cleanup candidates

Review related groups individually. Delete only after the user approves the
specific group. Preserve completed scientific results and selected releases.

## Group 1: repository-root temporary progressive work

Status: approved and deleted on 2026-10-13 on `login01`; `.tmp/` verified absent.

Deleted target: the entire repository-root `.tmp/` directory. It contained
4,369 regular files totaling 540,485,265 bytes (515.45 MiB) and 58 symlinks.
The final pre-deletion check found directory symlinks missed by the initial
file-only link inventory. Removal deleted the links without following their
targets. The regular-file count and byte total matched the reviewed inventory.

| Folder | What it served | Why deletion is proposed |
| --- | --- | --- |
| `.tmp/failed-v3-cache-builds/` | Staging directories for failed cache construction. | Contains two empty directories and no files. |
| `.tmp/sqlite-v3/`, `.tmp/sqlite-v4/` | SQLite temporary workspace. | Both are empty. |
| `.tmp/indirect-only-l2-l4-20260914/` | Six files: three launch/transition scripts, a provider configuration, a log and a progress note for a September progressive ablation. | Scripts invoke the retired progressive harness and hardcode old process IDs, endpoint addresses and staging locations. Their historical diagnostic information would be discarded. |
| `.tmp/morgan_semantic_l2_20260914/` | Failed/intermediate BBB and Oral SQLite caches, preparation inputs, test-generated files, logs and trace initialization records. | The progressive executable is retired; these files support its old preparation/debugging work. They contain no separate prediction, metric or model-response files. |

The largest component is `full_prepare/`: 3,960 files, including 2,636
`prepared.json` inputs and 1,318 `prepared_manifest.json` records, totaling
334,429,742 bytes. These are historical prepared inputs, not completed model
predictions. Four failed/intermediate cache directories total 203,373,089 bytes.
Two pytest output directories total 374 files and 1,680,557 bytes.

Checks performed:

- Listed every file and counted sizes without following symlinks or opening
  SQLite databases.
- Searched maintained prediction, optimization, experiment, data-processing,
  test and documentation files, excluding frozen semantic history/release/
  provenance trees and the historical cleanup inventory. Only `.gitignore`
  referenced `.tmp/`.
- Checked shared manifest, arm-manifest and matrix files at 10,025 entries in
  the existing prediction-container inventory; none referenced `.tmp/`.
- Confirmed `predict/harnesses/progressive/__main__.py` is absent. The old
  launchers also use a former analysis module location.
- Found no current-user launcher referencing `.tmp/` on `login01`. This is a
  current-host check, not a cluster-wide process audit.

Deletion discarded old debugging logs, the progress note, failed/cache
snapshots and exact prepared inputs for these abandoned runs. It did not
delete the canonical caches, evidence libraries, semantic releases or results
under `outputs/`. Old trace initialization records still say `running` or
`prepared`, but their last recorded updates are from September 14; those labels
alone were not evidence of a live process. Before deletion, a fresh current-host
process check found no current-user command argument or working directory using
the target, and no nested mount was found. Removal used `shutil.rmtree` on the
explicit approved path and checked `os.path.lexists` afterward.

## Group 2: bytecode for deleted Python sources

Status: approved and deleted on 2026-10-13 on `login01`.

Deleted targets: 704 `.pyc` files across 37 `__pycache__/` directories,
totaling 10,171,422 bytes (9.70 MiB). Each file's corresponding `.py` source
is absent. The exact reviewed paths and expected source paths are recorded in
`/tmp/txagent-stale-bytecode-candidates.json`.

Python writes these compiled copies automatically to speed up subsequent
imports. They remain after a source file is deleted or moved. Examples include
the removed MiniMol `run_hparam_sweep.py`, former prediction modules such as
`branches/batch.py` and `branches/flat_matrix.py`, and retired scoring/analysis
modules and their tests. Normal Python imports do not load orphaned bytecode
from `__pycache__/` without its source file.

The scan covered prediction, optimization, analysis, molecular tools, tests,
data-processing code and the root bytecode directory. It skipped symlinked
directories and frozen semantic history, releases, provenance and ledgers.
No maintained code was found explicitly loading `.pyc` files or using
`SourcelessFileLoader`. Cache files whose source still exists are outside this
proposal. Deletion can remove a cache directory only if it becomes empty;
source directories and scientific payloads stay in place.

Before deletion, all 704 files retained their reviewed sizes and all corresponding
source paths were still absent. Verified every reviewed bytecode path was absent
after deletion. Two cache directories became empty and were also removed:
`predict/harnesses/branches/reasoning/__pycache__/` and
`predict/harnesses/branches/tasks/bioavailability_ma/reranking/__pycache__/`.
Other cache directories remained. The deletion receipt is
`/tmp/txagent-stale-bytecode-deletion.json`.

Other inspected items are not proposed for deletion: `traces/` contains actual
model trace files; archived instructions contain scientific notes; the root
Ames/DILI/Carcinogens tarballs contain original evidence deliveries whose
Parquet sizes differ from the current raw files. They are not verified
duplicates of today's data.

## Group 3: archived LoRA/RL training configurations

Status: approved and deleted on 2026-10-13 on `login01`; folder verified absent.

Deleted target:
`archive/chembl_tool/paper_experiments/rl_lora/configs/`.
It contained seven regular files totaling 11,685 bytes:

| File | Original purpose |
| --- | --- |
| `grpo_gpt_oss_20b_one_pass_bio.yaml` | Full local GPT-OSS-20B Bioavailability GRPO/LoRA training recipe. |
| `grpo_gpt_oss_20b_one_pass_bio_smoke.yaml` | Twelve-row, one-step training check inheriting the full recipe. |
| `grpo_gpt_oss_20b_one_pass_bio_2step.yaml` | Two-step check on the same small dataset. |
| `grpo_gpt_oss_20b_one_pass_bio_full_2step.yaml` | Two-step check using the full dataset. |
| `grpo_gpt_oss_120b_lora_smoke.yaml` | Historical local GPT-OSS-120B NeMo training check. |
| `grpo_gpt_oss_120b_lora_feasibility.yaml` | Historical local GPT-OSS-120B feasibility recipe. |
| `README.md` | Describes the recipes, inheritance and their old validation rules. |

The training executables were already retired. No maintained prediction,
optimization, analysis, experiment, molecular-tool, data-processing or test code
references these configurations or the former NeMo/Tinker training entry points.
The short recipes still hardcode a former checkout and old local scratch training
locations. They have no role in the current prediction, retrieval or
context-optimization pipeline.

Their remaining purpose was historical documentation of exact training settings.
Deletion gave up these original configuration files, while preserving the
neighboring historical result reports and all scientific output directories.
Archived documentation does reference them, and `archive/RELOCATION.json`
inventoried all seven. Removed those seven inventory entries and updated the
parent archived README to identify the training commands/configurations as
retired. Updated only that README's existing inventory checksum. Six neighboring
reports retained their sizes, modification times and inodes. Saved training/
inference outputs and compressed archives were outside the deletion target.
The before-state and verification receipts are
`/tmp/txagent-rl-config-cleanup-before.json` and
`/tmp/txagent-rl-config-cleanup-verification.json`.

The BBB/Oral versioned readout commands were also inspected. Several remain
imported by source-local and incremental grouping code, so deleting them directly
would break maintained capabilities. Any future consolidation needs a separate
behavior-preserving refactor.

## Group 4: guidance for the retired RL training subsystem

Status: approved and deleted on 2026-10-13 on `login01`.

Completed deletions: five Markdown files totaling 36,299 bytes, plus three
empty directories. Four files are under
`archive/chembl_tool/paper_experiments/rl_lora/`:

| File | Original purpose |
| --- | --- |
| `LOCAL_NEMO_RUNBOOK.md` | Node/software setup, runtime patches, GPU topology, launch commands, debugging receipts and service restoration for the removed NeMo training stack. |
| `ONE_PASS_IMPLEMENTATION_PLAN.md` | Design of the historical one-pass RL experiment, including its information contract, reward formula and evaluation rules. |
| `ONE_PASS_REASONING.md` | Ownership and execution lifecycle of the removed one-pass materializers, audits, evaluators and training commands. |
| `BACKEND_PARITY.md` | Shared training settings and implementation rules for the removed Tinker/NeMo comparison. |

The fifth is the former subsystem instruction snapshot:
`archive/instructions/tools/chembl_tool/paper_experiments/rl_lora/INSTRUCTIONS.md`.
It repeats the old training boundaries, backend rules and experiment contracts.
It is ordinary archived documentation, not an active AGENTS.md.

The empty directories are `patches/`, `runtime/` and `scripts/` beneath the
archived RL folder. Their implementations were removed previously.

No maintained code or active instructions reference the five documents.
References found are within the proposed documents, the archived RL README,
the parent archived experiment README and the archive relocation inventory.
The remaining value is historical training-design and operational detail;
deletion explicitly gives up those documents, including reward/contract details
and bring-up receipts they contain.

Preserved `ONE_PASS_BASE_VALID_RESULTS.md` and
`VISIBLE_ONE_PASS_VALID_RESULTS.md` and verified both are byte-for-byte identical
to their pre-deletion contents. Replaced the archived RL README with a short
index to these two reports and updated the parent experiment README's retired
guidance link. Removed five relocation-inventory entries and refreshed only the
two changed README entries; all other entries are unchanged. Verified all eight
deletion targets are absent and the archived RL folder contains only its README
and two reports. Saved model outputs, checkpoints and compressed archives were
outside this group.

Exact reviewed paths/sizes are in
`/tmp/txagent-retired-rl-doc-candidates.json`.
The verification receipt is `/tmp/txagent-retired-rl-doc-deletion.json`.

## Group 5: unused prediction utility code

Status: approved and removed on 2026-10-13 on `login01`.

| Target | Original purpose | Reference evidence |
| --- | --- | --- |
| Entire `predict/utils/sqlite.py` (41 lines) | Open ChEMBL SQLite databases read-only, inspect tables/columns, and execute queries. | No maintained module imports it or calls its `connect_sqlite`, `get_table_columns` or `read_sql` functions. `table_exists` is used only inside this otherwise unused module. |
| `write_csv`, `write_jsonl`, `_serialize_cell` in `predict/utils/files.py` | Generic output writers and CSV cell formatting. | No callers of these functions; the separate evidence-library-local `write_jsonl` is a different implementation and remains. |
| `canonical_json_bytes` in `predict/utils/json.py` | Deterministic JSON serialization for hashing. | Only its own utility test calls it; no production callers. |

Keep `ensure_dir` in `predict/utils/files.py`: five maintained evidence-library
builders/index modules import it. Keep JSON parsing, atomic output, JSON/JSONL
atomic writers, JSONL reading and streaming hashing in `predict/utils/json.py`:
they have maintained callers throughout the pipeline. No wildcard imports of
these utility modules were found.

Deleted the SQLite module and its two compiled cache files. Trimmed the unused
functions and imports from the other two utility files. Updated the combined
JSON/hash test to exercise streaming hashing directly across multiple chunks;
all six JSON utility tests passed. A focused directory-helper check passed for
new nested-directory creation and reuse through both Path and string inputs.
Retained utility functions have identical parsed function definitions, and a
fresh reference search found no remaining callers of the removed helpers.

Production Python fell by 85 lines; including the adjusted test, the reduction
is 88 lines. Counts compare against this group's immediately preceding working
tree, not Git HEAD. Scientific algorithms, saved data, experiment recipes and
model transport were outside this change. The verification receipt is
`/tmp/txagent-unused-utility-cleanup-verification.json`.

## Group 6: unused tool-prefetch and text modules

Status: implemented with user approval on 2026-10-13 on `login01`. Preserved
failed-tool retry behavior as a function in the shared client and deleted the
three old source/test files.

| File | Original purpose | Reference evidence |
| --- | --- | --- |
| `predict/tools/prefetch.py` | Batch molecular-tool calls and retry failed positions once. | No caller or maintained module reference to its `invoke_with_retry` function. The tool package exports only the separate `ToolServiceClient`. |
| `predict/utils/text.py` | Normalize separators/case/whitespace and match token-bounded phrases, such as BBB terminology. | No production imports; its only callers are two tests in the following file. |
| `tests/predict/test_text.py` | Check the unused normalizer and phrase matcher. | Dedicated entirely to `predict/utils/text.py`. |

Deleted `predict/tools/prefetch.py` (25 lines), `predict/utils/text.py` (56 lines)
and `tests/predict/test_text.py` (12 lines), plus five corresponding compiled
cache files. The user identified failed-tool retries as useful and requested
that behavior remain a function rather than a separate module.

Reference checks covered maintained source, tests, recipes, active instructions,
documentation, packaging configuration and the asset-relocation inventory.
Dynamic task/module loaders were also identified: they load registered task
pipelines and construction modules, not these utilities. The actual molecular
tool client, tool-service implementation, live batching/fallback behavior and
scientific text-processing code stay with their current owners.

`predict/tools/client.py` now owns `retry_failed_tool_calls`. It checks result
counts, keeps successes in order and retries each failed result once. The
client's `invoke_many` uses it after a valid batch response or after individual
fallback calls for a failed/malformed batch. Six-task query-prior preparation
uses the same function after its direct molecular-property call. TDC priors
reach it through the existing harness prefetch and shared client. Persistent
errors remain visible; six-task preparation still fails before model requests
if molecular properties remain unavailable. Single-call and batching transports
retain their existing routes.

All 65 focused tests passed: identity redaction/prefetch, reasoning calls and
runtime, TDC prior contracts and query payloads. Eight added cases check valid
batches, partial failures, persistent failures, batch transport failure,
malformed result counts, direct property retries and fail-before-call count
validation. HTTP transport was mocked; no service or model requests were made.
Updated the prediction-owner AGENTS.md to name the shared retry owner. Verified
deleted files are absent, changed Python compiles and no imports of the retired
modules remain. Production Python decreased by 63 lines against the immediately
preceding working tree. The verification receipt is
`/tmp/txagent-tool-retry-cleanup-verification.json`.

## Group 7: unused API setup wrappers

Status: simplified and removed with user approval on 2026-10-14 on `login01`.

| Target | Original purpose | Reference evidence |
| --- | --- | --- |
| Entire `predict/api_client/endpoints.py` (43 lines) | Three convenience constructors for LiteLLM, OpenRouter and vLLM provider settings. Each returns the existing `ProviderSpec`; the OpenRouter shortcut adds its URL and reasoning-enabled request field. | Imported only by `test_endpoint_profiles_share_transport_contract` in `tests/predict/test_predict_package.py`. No maintained production callers; not exported by the API package. |
| `load_env_file` in `predict/api_client/pool.py` (17-line function) | Read KEY=VALUE entries into process-wide environment variables. | Called only by the environment section of `test_jsonl_and_env_helpers_preserve_historical_parsing` in `tests/predict/test_reasoning_payload.py`. |

Provider selection and transport remain in `pool.py` and `transport.py`.
Production callers read provider JSONs into the existing provider types, and
`resolve_api_key` uses `transport.read_env_file` plus the process environment.
That active credential reader has distinct behavior and is preserved. Deletion
removes unused constructor shortcuts and the old environment-mutating loader;
local/DGX, OpenRouter, LiteLLM and vLLM transport support remains.

Verified the active path: prediction CLI `--provider-pool-config` selects a
provider JSON; `pool.load_provider_pool_config` reads it and
`pool.build_provider_pool` calls `transport.openai_compatible_client` for each
provider. The checked DGX profile is `providers/flat_dgx.json`; the OpenRouter
profile is `providers/deepseek_v4_flash_0731_openrouter_auto_128_high.json`.
Both resolve through the same API implementation. Profiles also carry model,
concurrency and routing settings. No endpoint availability was probed.
`endpoints.py` is not the connection implementation; it contains unused
ProviderSpec-construction shortcuts. Preserve the shared connection and backend
selection capability throughout this cleanup. The user approved removing the
unused wrappers after this distinction was clarified.

A broader follow-up scanned 876 maintained production Python modules across
prediction, optimization, analysis, data construction and molecular tools. The
only direct OpenAI/AsyncOpenAI SDK constructor calls found are in
`predict/api_client/transport.py`. Main prediction, priors, semantic scoring
and the checked SDK-based builders use that shared transport directly or through
the common client/pool. They do not all select providers through the same JSON
interface: some construction commands pass a base URL and provider directly.
Older Skin embedding/extraction paths also retain an external distillation-API
fallback. Local tokenizer/health and route-discovery requests use their own HTTP
endpoints. Therefore do not claim every request or every legacy path uses one
uniform interface. These exceptions are separate from the unused endpoint
constructors and environment-mutating loader, whose callers remain test-only.

Deleted `predict/api_client/endpoints.py` and its two compiled caches. Removed
`pool.load_env_file`, the endpoint-constructor-only test and the unused
environment-loader section of the combined payload test; kept the JSONL
checks. Every other parsed definition/import in `pool.py` is unchanged.
Clarified the API transport/client/pool responsibilities in the root and
prediction AGENTS.md files. No new API wrapper or configuration layer was added.

All 41 focused package, payload, provider-pool and shared-transport checks
passed. An initial legacy trace-viewer fixture used the old unorganized output
layout; updated only its setup to use an indexed run's `samples/` layout and
mocked publication. Viewer production code and assertions are unchanged.
Verification used temporary fake credentials and mocked endpoints, with no
provider probes or inference. No maintained references to the removed API
helpers remain; changed files compile. Production Python decreased by 62 lines
against the immediately preceding working tree. The verification receipt is
`/tmp/txagent-api-helper-cleanup-verification.json`.

## Group 8: unused compatibility wrappers and mode checks

Status: approved and removed on 2026-10-14 on `login01`.

Five function definitions across three existing files total 21 lines before
surrounding blank lines:

| Target | Original purpose | Why it can be removed |
| --- | --- | --- |
| `visibility.prepare_identity_blind_retrieval` | Call the generic prefetch/redaction function with `identity_blind=True`. | Only tests use this shortcut; production already calls the generic function with explicit arguments. |
| `visibility.prepare_identity_blind_final_retrieval` | Call the generic saved-property redaction function with `identity_blind=True`. | Only its test uses this shortcut; retain the underlying function and test its behavior directly. |
| `retrieve.similarity_bucket` | Call `similarity_bucket_for_index` with an empty Morgan index. | No callers. Active retrieval already uses the shared index-aware function. |
| `runtime._versioned_flat` and `runtime._versioned_flat_args` | Check whether a prepared run specifies a full-flat prompt. | Neither is called by production or tests; the second is referenced only by the first. Active `_flat_one_call` and runner mode checks remain. |

Removed all five functions and updated four identity-hiding test call sites to
use the retained generic functions with `identity_blind=True`. Every retained
module definition/import and every identity-test assertion is unchanged.
No maintained references to the removed names remain. Production Python
decreased by 31 lines including surrounding blank lines.

All 47 identity-redaction, reasoning-runtime and retrieval checks passed. An
initial retrieval test still expected the old `rdkit_fragment_parent.v1` tag;
updated that fixture expectation to the existing declared `v2` contract.
Production molecule normalization was unchanged. No inference or service
requests were made. Exact reviewed definitions are listed in
`/tmp/txagent-unused-wrapper-candidates.json`; the verification receipt is
`/tmp/txagent-unused-wrapper-cleanup-verification.json`.

## Group 9: unused checkpoint and retrieval bookkeeping helpers

Status: approved and removed on 2026-10-14 on `login01`.

Six unused functions across three existing modules:

| Target | Original purpose | Why deletion is proposed |
| --- | --- | --- |
| `runtime._canonical_group_outputs` in `predict/harnesses/branches/runtime.py` | Read saved branch results and order them by expected group ID. | No callers; a test imports it but never uses it. |
| `runtime._merge_group_output` in the same module | Replace one branch result in a saved JSONL file and sort the rows. | Called only by its dedicated test, never by production. Remove that test and both unused test imports with these functions. |
| `retrieval._source_name` in `predict/harnesses/branches/retrieval.py` | Choose a source label from task configuration or index metadata. | No callers. Active retrieval receives the configured source name directly. |
| `runtime.canonical_cache_profile` in `predict/retrieval/assay_reranking/runtime.py` | Forward an old cache-profile name to the canonical alias resolver. | No callers. Active cache-path resolution already calls `artifact_bundle.canonical_cache_alias` directly. Keep that resolver and its mappings. |
| `runtime.prompt_task_to_dict` and `runtime.prompt_task_from_dict` in the same module | Convert a scoring request between its dataclass and a dictionary. | No callers. Builders already use `dataclasses.asdict` and `PromptTask(**row)` directly. Keep the request dataclass and existing serialization; remove this module's now-unused `asdict` import. |

Searched maintained prediction, optimization, analysis, evidence construction,
molecular tools, tests and experiment Python/Bash files for references. The
only uses found outside these definitions are the two test imports and the
dedicated merge test. Checkpoint readers/writers, stage resume, trace export,
source labels, cache aliases and scoring-request serialization remain in use
and are outside the deletion scope. No whole module or saved result is proposed
for deletion.

Removed all six functions, their two unused test imports, the dedicated merge
test and the unused `asdict` import. Production Python decreased by 37 lines;
the test file decreased by 26 lines. Every remaining parsed statement in all
four edited Python files is unchanged, and each file compiles. No maintained
Python/Bash references to the removed functions remain. No saved result or
whole module was deleted.

Focused reasoning-runtime, experiment-retrieval and assay-reranking tests
returned 49 passes and two failures. Both failures require absent historical
V10.4 model manifests in the neighboring `starling_assay_transfer` checkout,
for BBB and Oral; these manifest readers were unchanged by this cleanup.
Both failures were reproduced with the pre-cleanup assay runtime restored
only inside a separate test process. No inference or service requests were made.
The pre-change snapshot is `/tmp/txagent-bookkeeping-cleanup-before.json`;
the structural verification receipt is
`/tmp/txagent-bookkeeping-cleanup-verification.json`.

## Group 10: unused assay-transfer score prompt helpers

Status: approved and removed on 2026-10-14 on `login01`.

Target: eight functions in `predict/harnesses/branches/prompt.py`, totaling
179 function lines before blank separators, plus their three unused constants
and `math` import. Keep the module's query/condition and evidence-cleaning helpers.

| Functions | Original purpose |
| --- | --- |
| `validate_scored_neighbors_configuration` | Check experiment mode, source and reranker arguments before exposing scores. |
| `prepare_assay_transfer_selected_neighbors`, `enable_scored_neighbors_prompt_policy`, `_validate_public_score_bundle` | Validate selected score bundles, drop unscoreable neighbors, update ranks/counts and mark score visibility in retrieval metadata. |
| `scored_neighbors_prompt_enabled`, `public_assay_transfer_score` | Inspect that visibility marker and round one score to two decimal places. |
| `public_assay_transfer_records`, `public_assay_transfer_families` | Build model-facing record/family score bundles while omitting internal audit IDs. |

These functions form an unused block: references are confined to the block
itself. No external callers were found in maintained Python/Bash code or tests.
A broader reference search also checked tracked-style documentation, templates,
YAML and JSON outside generated run/FINAL/cache/audit trees. Task pipelines
import specific query/evidence helpers from this module; they do not load these
score functions dynamically.

Preserve assay-transfer retrieval, ranking, cached scores and model-visible
scores. Current `flat.py` renders selected cached scores directly and applies
Gold/TDC score visibility policies. `runner._validate_assay_transfer_scores`
is called during preparation and validates current CLI combinations. Those
implementations have their own contracts and remain untouched; the unused
helpers are not assumed to be equivalent replacements for them.

The initial proposal called this block "older"; that described its different
group/family prompt representation, not a verified historical date or prompt
release. The established deletion rationale is absence of callers today.
The additional check found no callers in archived code or prompt templates
either. No Git history was inspected to establish when it became unused.

Removed the eight definitions, three constants and `math` import: 201
production lines. Updated only the module docstring to describe the retained
query/evidence helpers. Every remaining parsed statement is unchanged and the
module compiles. No prompt template, saved result, retrieval cache or whole
module was deleted. No maintained references to the removed names remain.

Focused payload, flat-prompt, assay-transfer-selection and retrieval-ablation
checks returned 85 passes and three manifest-related failures. The failures
concern a selection fixture missing named artifact metadata, a reuse fixture
expecting a copied per-query manifest, and a branch-reuse fixture missing its
manifest. These paths were unchanged by the deletion. All three failures were
reproduced with the pre-cleanup prompt module restored only in a separate test
process. No inference or service requests were made.
Snapshots and structural verification are recorded at
`/tmp/txagent-score-prompt-cleanup-before.json` and
`/tmp/txagent-score-prompt-cleanup-verification.json`.

## Group 11: unused artifact convenience functions

Status: approved and removed on 2026-10-14 on `login01`.

Three functions in `predict/harnesses/branches/artifacts.py`, totaling 30
function lines before blank separators. Their only executable callers are tests.

| Target | Purpose | Reference evidence |
| --- | --- | --- |
| `read_jsonl_record` | Stream a JSONL file until one requested row is found. | Called only by its dedicated parsing test. Batch preparation reads inputs through the existing runner JSONL reader. |
| `retrieval_prompt_hash` | Serialize the prompt-visible retrieval fields and compute a SHA-256 digest to compare them. | Used only by three tests. Retain `retrieval_prompt_contract` and change those tests to compare the projected dictionaries directly, preserving checks for neighbor, condition and visible-record changes. |
| `materialize_reused_run` | Clone a completed run directory using hard links or copies, write `reuse.json`, and discard the copied per-query manifest. | Called only by its dedicated test. No current CLI or production path calls this whole-folder cloner. |

Remove the exclusively used `hashlib`, `os` and `shutil` imports with these
functions. Remove the isolated row-reader and folder-cloner tests; keep the
prompt-field comparison tests and the other artifact/reuse tests. The current
folder-cloner test already fails because it expects a copied manifest that the
function intentionally unlinks; this failure is not the deletion rationale.

Preserve trace writing, checkpoint resume, retrieval replay and current reuse
paths. `runtime.prepare_stage_item` separately copies frozen `retrieval.json`
when replaying a batch. `changed_group_ids`, `load_reusable_group_outputs` and
their prompt-field projection remain in scope for preservation. Removing the
unused folder cloner does not replace it with another implementation.

Searched maintained prediction, optimization, analysis, evidence construction,
tools, tests and experiments. Archived executable code has no callers either;
an archived instructions snapshot mentions `retrieval_prompt_hash` as part of a
historical reuse policy. Preserve that historical document. No age or release
date is inferred from this reference. No saved run or whole module was deleted.

Removed all three functions and their exclusively used imports. Removed the
two dedicated tests and the now-unused test `pytest` import. The three evidence
comparison tests now compare `retrieval_prompt_contract` dictionaries directly;
their fixtures, comparison operators and checked scientific differences remain
unchanged. Updated the artifact module docstring to describe its remaining
responsibilities. Production Python decreased by 44 lines, and test files
decreased by 34 lines in total.

Every retained production statement is unchanged apart from the module
docstring. All edited Python files compile, and no maintained executable
references to the removed functions remain. An initial branch-reuse test
failed because its existing fixture omitted the manifest required by the
reader; added the explicit standard neighbor-context manifest to that fixture.
Production reuse code and all assertions remain unchanged.

All 33 focused payload, artifact comparison, retrieval replay and reasoning
runtime checks passed after the fixture correction. No inference or service
requests were made.

Pre-change files and structural verification are recorded in
`/tmp/txagent-artifact-helper-cleanup-before.json` and
`/tmp/txagent-artifact-helper-cleanup-verification.json`.

## Group 12: discarded scratch recovery archive

Status: approved and removed on 2026-10-14 on `login01`.

Deleted `archive/workspace_history/`, including its sole dated snapshot
`20261012T043138Z/`: six regular files totaling 1,581,894,738 bytes. The user
said they did not care about the removed `.scratch/` work and identified this
recovery archive as disposable. This decision does not assume Git preserves
the ignored tarball.

The six files were the compressed scratch backup, its member inventory, a
README, pinned-input notes, checksums and a prior verification receipt. The
backup held intermediate semantic queues, evidence-build stages, analysis
outputs, smoke metrics and environment packages. No current prediction reader
used it. Counts, sizes and modification times matched the reviewed inventory
immediately before deletion; no symlinks inside the deletion target, nested
mounts or local process cwd/argv references were found.

Verified the directory is absent and the underlying Skin endpoint profile is
unchanged. Root `artifacts/`, all other archive contents, active prompts and
published results remain unchanged. No tests or model calls were needed for
removing this unused backup. Receipt:
`/tmp/txagent-workspace-history-deletion.json`. The remaining archive contents
and relevance are described in [archive_artifacts_review.md](archive_artifacts_review.md).

## Group 13: Git-preserved ChEMBL experiment archive

Status: approved and removed on 2026-10-14 on `login01`.

The user approved deleting this folder if its experiment results were saved in
Git. Verified all 53 regular result/report/template/configuration assets
against exact committed blobs under their former paths in HEAD
`a209fb4422b5fe62927746f92ac6d04ccb242f11`. The external ChEMBL dataset link
also matched its committed version. Ensured these committed blobs are locally
available from the current file bytes and verified the historical Starling
report can be read through `git show`; no new commit or staging was performed.

Three READMEs had uncommitted updates. Preserved their exact bytes under
`docs/look/chembl_tool_readmes/` before deletion, with an index explaining that
their old commands/links are historical. These files are not claimed to have
been committed. Two generated links to live unit/qualifier policy files held
no additional data; the target policy files are preserved.

Removed `archive/chembl_tool/`: 56 regular files totaling 1,801,784 bytes and
three internal symlinks. Also removed its `tools/chembl_tool` compatibility
alias and 57 relocation-index entries; every retained index entry is unchanged.
Counts, sizes and modification times were rechecked immediately before
deletion. No nested mounts or local process cwd/argv references were found.

Updated the optional precomputed ChEMBL fingerprint default to the exact
external path formerly reached through the alias, preserving its existing
SMILES-fingerprint fallback and CLI override. Updated the layout guard to
require the deleted paths to stay absent. Two Skin fixtures now use their
existing canonical V7 data paths, and the normalization-corpus documentation
points to the retained instruction snapshot. Removed a dead MiniMol launcher
description, replaced its report link with the verified Git retrieval command,
and clarified the root README. Active prompt bundles, scientific data and
completed prediction runs remain unchanged.

All 20 selected data-layout, retrieval and import-boundary checks passed.
The two additionally checked legacy Skin fixture modules returned five
failures and one skip. The same five tests fail with the previous fixture
location: four require an already-missing reconciliation policy, while the
out-of-denominator test previously referenced a nonexistent path. Pointing it
to the canonical V7 files exposes a mapping/review hash mismatch. No mapping,
review receipt, scientific assertion or production normalization behavior was
changed to make these tests pass. These historical fixture/data problems remain
outside the archive deletion. Edited Python files compile, the equivalent
fingerprint target is preserved, and Git restoration of the report succeeded.

Inspection, preservation and deletion receipts:
`/tmp/txagent-chembl-archive-deletion-review.json`,
`/tmp/txagent-chembl-archive-git-preservation.json`,
`/tmp/txagent-chembl-archive-readme-preservation.json`, and
`/tmp/txagent-chembl-archive-deletion.json`.

## Group 14: consolidate the conditioned Oral baselines

Status: approved, consolidated and removed on 2026-10-14 on `login01`.

Removed `predict/baselines/context_conditioned_bioavailability/`, its dedicated
test module and their generated bytecode: 11 files totaling 131,874 bytes.
The package duplicated Morgan KNN, MiniMol cosine KNN, embedding preparation
and a fixed-epoch MiniMol head. Those capabilities now use the existing shared
commands, instantiated by `experiments/baselines/conditioned.sh`; there is no
compatibility wrapper or separate combined-summary format.

The shared selector takes the highest-ranked eligible row for each molecular
parent, prioritizes the query's condition, and fills from previously unselected
null-condition parents. Null queries never retrieve external-condition rows.
Insufficient pools remain unsupported and are included in coverage diagnostics;
no pool is broadened to make predictions possible. The generic row-agnostic
default and the existing null-preferred all-train comparison remain unchanged.

The recipe explicitly selects condition matching and a train-derived one-hot
head with all train rows, 25 epochs, five members, seeds 1–5 and threshold 0.5.
The architecture, optimizer, schedule and batch defaults match the retired
runner. Train-only CV remains a separate scientific contract. Shared outputs
preserve benchmark/parent IDs and add condition cohort metrics, including
single-class and unsupported cohorts. Cached embeddings are prepared once,
reused by exact SMILES, checked for shape/finite values and conflicting vectors,
and saved with the current labels. Unknown head categories fail before embedding;
molecule-only cache preparation permits unknown KNN query conditions.

Updated root/prediction READMEs, the MiniMol README and applicable root/prediction
AGENTS.md instructions. No maintained executable or README references to the
retired package remain; historical relocation documentation is retained.
Removed 684 lines of duplicate Python, added 65 net lines to shared Python and
70 lines of Bash: production code decreased by 549 lines.

All 45 focused shared baseline tests passed on CPU. They cover conditioned
selection, metrics/coverage, cache rejection and label refresh, category
validation, the three-method Bash recipe and checkpoint score reproduction.
A separate old/new comparison on synthetic records produced identical labels
and neighbors, scores within 1e-7, and identical legacy condition metrics for
all three methods. Bash syntax and the one-time force-embedding argument flow
also passed. No pretrained downloads, GPU jobs, API calls, real benchmark
evaluations or completed-run modifications were performed.

Temporary source snapshot, equivalence and removal receipts:
`/tmp/txagent-baseline-consolidation-before.json`,
`/tmp/txagent-baseline-consolidation-parity.json`,
`/tmp/txagent-baseline-consolidation-deletion.json`, and
`/tmp/txagent-baseline-consolidation-verification.json`.

## Group 15: root historical evidence snapshot

Status: approved and removed on 2026-10-14 on `login01`.

Deleted target: the remaining root `artifacts/` tree, comprising 11 regular files
totaling 172,117,189 bytes and four symlinks. The nine Parquets are a frozen
collaborator export for BBB, Oral and Skin: three tables assigning source
records to evidence levels, and six tables identifying representative evidence
cards in the random/scaffold retrieval indexes. The README and manifest explain
the export and record its counts and inputs. These are historical evidence
tables, not saved model predictions or new-query context selections. The export
command named by the README has already been retired.

The four links expose ClinTox and three V7 evidence stores under old artifact
paths. Removing these links does not delete their `data/legacy/` targets.

No maintained reader of these working-tree paths was found. The source-UID
builder references the level-export path but reads committed data with
`git show 26d44f121141f3738764c0b00dc09e185445341b:<path>`; it does not read
the local folder. Preserve that Git-addressed source contract. Current
prediction resolves published evidence libraries and caches elsewhere.

Refreshed the earlier Git review without rehashing the large tables: all 11
files retain their previously verified sizes and modification times, and all
15 previously verified blob identities remain at the same paths in current
HEAD `0a4f11ef4259f48e8659fd76734c53233249bdf9`. Also checked all four link
targets against their committed identities. Deletion discarded these local
copies while preserving their Git versions and linked datasets. Historical
documentation retains its descriptions. The final maintained-reference check
found only the intentional Git-addressed import paths; those remain unchanged.

Reviewed candidate list: `/tmp/txagent-root-artifacts-cleanup-candidates.json`.

V10 dependency follow-up: checked the active release's source and level-mapping
manifests for all six tasks. All 40 named input paths exist; none is under root
`artifacts/`. V10 source records come from `data/raw/starling/main_universe_v1/`
and, for Ames source fields, the original `data/raw/starling/ames/` acquisitions.
Active UID-to-level assignments live in published gold/evidence-library mapping
tables. Historical mappings under `data/legacy/artifacts/evidence_libraries/`
also remain inputs to mapping publication and must be preserved independently.
The root export is related scientific history, but its local files are not
required inputs for the active V10 releases. The bootstrap builder still reads
its pinned Git snapshot, not these working-tree copies. These inputs were
preserved during deletion.
Receipt: `/tmp/txagent-root-artifacts-v10-dependency-review.json`.

Immediately before removal, verified the exact file/link inventory, regular-file
sizes and modification times, and current committed identities. All 15 Git
blobs are already available locally, so restoration does not depend on a new
fetch. No large tables were rehashed. No nested mounts or current-user processes
on `login01` referenced the deletion target. Removed the root with
`shutil.rmtree`, which does not follow the four internal links, and verified it
is absent. All 48 protected source/mapping files, directories and former link
targets retain their filesystem metadata. No production code, scientific input
or completed-run payload was edited; no model calls or tests were needed.
Deletion receipt: `/tmp/txagent-root-artifacts-deletion.json`.


## Group 16 — canonical prompts and TDC configuration (implemented)

Approved consolidation replaces three Python prompt-profile definitions and the
old flat prompt directory with `predict/prompts/`, including all six task YAMLs.
Frozen archives remain readable directly. One unversioned TDC manifest selects
Oral/Skin scaffold cohorts and the other four original-source cohorts. Active
labels and selected cache indexes were relocated without changing scientific
payloads; historical editions, original headers and 19 retired YAMLs live under
`data/legacy/tdc/` with an explicit relocation map.

The concurrently refactored optimizer uses separate full-molecule L1 card caches;
parent-grouped scores remain distinct. See [implementation, validation and the
Carcinogens disjointness limitation](canonical_prompts_tdc.md). Completed runs,
FINAL artifacts and frozen score values were not rewritten. No model calls ran.

## Group 17: retired assay contrastive execution and obsolete Skin scoring cache

Status: approved and deleted on 2026-10-15 on `login01`.

Deleted cache: `predict/retrieval/cache/assay_reranking/active/v9_skin_gold_v1_scaffold_morgan100_v1/`:
four valid/test VERSION and ranking Parquet files, totaling 3,225,573 bytes.
These held the pre-correction Skin direct scores and depended on old Gold vote
summaries. The corrected source passed the shared six-entry direct-release
validator before deletion. Maintained builder defaults, the scoring recipe and
release receipts now use the corrected source; no lookup aliases the deleted
scores to the corrected scores.

The corrected cache and scoring source were relocated to
`data/caches/assay_reranking/active/flat_v5/gold_v1/skin_reaction/l1/assay_transfer/v9_0_2/scaffold_disjoint_parent100_v1/`.
All eleven files retain their exact bytes and content IDs. Old corrected-cache
locators resolve through `data/caches/assay_reranking/relocations.json`; the
pre-correction deleted cache has no replacement alias. The deletion receipt is
`/tmp/txagent-skin-cache-retirement.json`.

Removed executable assay-transfer contrastive selection, its matrix sweep
branches/arguments and dedicated builder report. Kept the general matrix
launcher, ordinary Morgan/assay-transfer ranking, frozen optimizer selections,
standalone cache construction, historical prompt rendering, and the separate
legacy Morgan contrastive capability. Completed results and provenance remain
unchanged. Direct L1 always uses scaffold-disjoint candidates. Both named
indirect-policy catalogs share those L1 bindings and retain published score
availability; all 24 task/benchmark bundle receipts were verified.

Validation: the broad CPU suite finished with 859 passed and four pre-existing
failures (two missing external V10.4 provenance manifests and two obsolete TDC
path/configuration expectations). The scoring checks passed five tests and
skipped four GPU-only cases. All 24 published task/benchmark bundle receipts
passed verification. Eight actual Skin retrieval/manual-UID round trips passed
across Gold/TDC, valid/test and both indirect policies. All eleven relocated
files remain byte-identical. Retained standalone cache-builder functions are
AST-identical, and `git diff --check` passes. No new assay scores or model API
calls were made; no commit or push was performed.
