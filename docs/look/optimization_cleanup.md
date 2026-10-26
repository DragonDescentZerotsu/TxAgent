# Optimization cleanup

The semantic owner is `optimization/semantic_buckets/`, imported as
`optimization.semantic_buckets`. The root symlink has been removed.

## Small shared implementations, explicit experiments

Keep one Python implementation for each distinct scientific capability. Expose
real experiment variations as CLI arguments, and instantiate specific runs with
Bash recipes. Share algorithms and execution mechanisms where they actually
match; keep prediction, query-prior generation, selection, scoring, evidence
construction and independent analyses separate when their scientific contracts
or execution differ. Scientific logic, validation, scheduling, concurrency,
checkpointing and aggregation stay in Python. Bash chooses inputs, settings,
providers, dependencies and output identities.

Simplification should remove executable duplication and experiment machinery,
while preserving meaningful methods, performance controls, resumability,
provenance and diagnostics. It should not turn distinct methods into a large
conditional dispatcher, move algorithms into shell scripts, or weaken scientific
validation to reduce file or line counts. The root and relevant owner AGENTS.md
files document this policy.

## Shared recovery command

Recovery resumes unfinished model requests from an existing prepared SQLite
ledger. It does not restore deleted files. The two duplicated recovery commands
were deleted and replaced by
`optimization/semantic_buckets/recover_task_bucket_refinement.py`.

Choose `--variant flex` or `--variant standard-medium`. Their distinct study
contracts retain prepared-input and response validation. The shared runner
preserves model/provider settings, reasoning effort, eight retries, completed
rows, attempt history, checkpoints and capacity controls. A completed ledger
makes no API requests and keeps its snapshot bytes unchanged.

The recipes are `experiments/semantic_buckets/recover_flex.sh` and
`experiments/semantic_buckets/recover_standard_medium.sh`. Both require
`RUN_ROOT`, accept `PYTHON_BIN`, and forward CLI arguments. Python recovery code
fell from 294 lines in two commands to 156 lines in one shared command.

## Prediction manifests and migration

Prediction settings live once per batch (`prediction_batch.v2`) or FINAL arm
(`evidence_final_arm.v3`). Shared records retain named benchmark/evidence/map/weight
versions, effective selection parameters, model and prompt settings, ordered
query UIDs, and sparse genuine scientific overrides. Query events remain in
`stage_resume.json`; repeated checkpoint configuration is stored once in the
batch. Readers inherit the shared contract and reconstruct runtime locations
from explicit roots, existing resolvers and local frozen input snapshots.

Experiment identity uses versions and UIDs. It does not depend on file paths or
content fingerprints. Actual served model/provider remains in responses and
traces. Frozen data receipts and payload audit hashes retain their separate
purpose; this cleanup did not propagate changed metadata checksums through a
historical dependency graph.

The metadata-only migration converted **5,248 batch/selection/FINAL records**,
including **137 readable FINAL arms**, and retired **347,406 per-query
manifest files**. It updated operational matrix/collection indexes and readable
FINAL metadata inventories. It preserved 22 separate containers without query
manifests, including historical prompt catalogs and incomplete preparation
folders. Compressed `final/git_bundle/` files were left untouched.

**8,895 historical query UID entries remain empty** because their original saved
inputs did not provide a recoverable UID. They were not filled from today's
`CURRENT` or invented from molecule strings. Historical results remain readable;
unknown identities block unsafe reuse. The migration report records the affected
batches. No predictions, evidence, requests/responses, prompts, scores or selected
releases were rewritten.

The shared helper is `predict/harnesses/branches/manifests.py`; the restartable
migration command is `python -m predict.harnesses.branches.migrate_manifests`.
It processes containers independently and publishes each shared header before
removing obsolete query metadata. Checkpoint writes and query removal can resume
after interruption. The completed per-batch receipt for this checkout is
`/tmp/txagent-prediction-manifest-migration.jsonl`.

## Verification

- The six selected CURRENT releases and 28 selected semantic payloads retained
  their release identities, sizes, timestamps and inodes.
- The 183 checked FINAL prediction/metric/combined-trace files retained their
  sizes, timestamps and inodes. All 61 indexed arms retained their identities,
  query counts and scores; selected numerical table cells are unchanged.
- 103 focused checks passed, covering migration, relocation, sparse overrides, checkpoint state,
  incompatible query/version/settings rejection, prior reuse, render/replay,
  matrices, analysis, semantic artifacts, weighted policy and recovery.
- No inference, recovery or training jobs were launched.
