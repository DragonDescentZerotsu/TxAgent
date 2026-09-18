# Progressive prediction

> Historical runtime guide retained for reference. The current harness contract
> in the repository `AGENTS.md` overrides the versions and defaults below.

The active progressive path has three runtime boundaries:

```text
immutable SQLite cache -> exact prompt messages -> validated model state
retrieval_cache.py         prompt.py                inference.py
```

- `retrieval_cache.py` opens the task/split cache and performs indexed reads of
  rows that were selected in advance. It does not read source Parquet, rebuild
  mappings, normalize molecules, recompute Morgan similarity, or run database
  publication checks.
- `prompt.py` turns prepared rows plus the preceding level state into the exact
  system and user messages. `_records.py` is its private record-to-card assembly
  implementation; it is not another cache or execution path.
- `grammar.py` renders the optional version-owned GBNF template without moving
  grammar assets out of their prompt bundles.
- `inference.py` sends those messages through `predict.api_client`, validates the
  returned JSON, checkpoints the state, and then unlocks the next level.

`runner.py` is the command and run-artifact coordinator around those boundaries.
It selects queries, prepares directories and manifests, schedules independent
queries, and supports resume. It does not implement an alternative retrieval or
model client. The optional matrix coordinator lives in `matrix.py`; scientific
interpretation of completed runs belongs under `analysis/`.

## Active contract

```bash
python -m predict.harnesses.progressive \
  --harness-version reranked-progressive-v2 \
  --reranking assay-transfer
```

`--reranking` is one of `assay-transfer`, `joint`, or `morgan`. All modes use the
active `prompts/reranked_progressive_v8/` assets and the same cache reader.
Assay-transfer and Morgan are native cached rankings. Joint reconstructs L1 from
the top five of each native ranking, merges overlaps, and does not refill.

The versioned asymmetric joint experiment uses
`--harness-version reranked-progressive-v4 --reranking joint`. It reconstructs
L1 from the top three assay-transfer molecules followed by the top seven Morgan
molecules, merges overlaps without refill, and reads only the same immutable v3
cache. This does not change the historical v2 5+5 contract.

Prompt bundle `reranked_progressive_l1_simple_v13` supports the composable
`_no_scores` modifier. Combining
`_no_scores_no_smiles_no_query_smiles` produces the records-only ablation while
leaving experimental measurements intact; `_no_query_prior` remains independently
composable.

The cache location is supplied by `--assay-transfer-cache`; it cannot change the
version's stage rules. The default is validation-only. Historical prompt bundles
were removed from the active checkout after the run-receipt audit recorded in
`prompts/migration_receipt.json`. Use their pinned historical checkout for exact
reproduction.

## Live review and throughput

Live is the default. It writes traces immediately under
`outputs/paper/live/<study>/<method>/<run>/<task>/<condition>/`, publishes the first
three samples, and pauses at `awaiting_review`. The three pilot prompts are
prepared first; their inference runs while the remaining prompts are prepared
in the background. Non-pilot inference stays behind the review gate:

```bash
python -m predict.live list
python -m predict.live show RUN_ID
python -m predict.live continue RUN_ID
python -m predict.live cancel RUN_ID
```

For an uninterrupted private run, add `--execution-mode throughput`. It skips the
pilot gate, executes the complete prompt pool, and keeps viewer data private until:

```bash
python -m predict.live promote RUN_ID
```

To edit a prompt without mutating a versioned bundle, clone and relaunch it:

```bash
python -m predict.live clone-prompt RUN_ID --to reranked_progressive_v9_candidate
# edit the cloned assets
python -m predict.live relaunch RUN_ID --prompt-version reranked_progressive_v9_candidate
```

The clone records `reranked_progressive_v8` as its runtime parent, so it reuses
the active assembly behavior while receiving independent asset hashes and output
paths.

## Experiment matrices

The progressive sweep is explicitly launched with:

```bash
python -m predict.harnesses.progressive.matrix --help
```

Its detailed resume and receipt contract is in
`predict/harnesses/progressive/MATRIX.md`. The flat comparison
launcher is `python -m predict.harnesses.branches.matrix`.

Run leaves are organized under
`outputs/paper/assay_transfer_harness/joseph/<study>/<method>/`. For example,
contrastive Morgan K=10, M=3 is
`contrastive/morgan/k10_m3_<date>/`; assay-transfer is a sibling method, not a
condition buried inside the Morgan directory. Shared matrix calls live only in
`_batches/<batch-id>/`.

`diagnostics.py` is the post-run measurement boundary. It indexes the exact
visible `Molecule N`, `Evidence group N`, and `Cxx` identifiers, measures their
explicit occurrence in `llm.reasoning_content`, and calculates L1 label mix from
the selected condition-context labels. These labels are internal provenance and
are never rendered into the prompt.

New organized runs use a `*_references_v1` successor bundle. Its immutable
`progressive_reasoning_references.v1` contract makes the renderer save the exact
case-insensitive identifier index used by `progressive_run_diagnostics.v2`.
Molecule numbers are unique across the whole prompt, evidence-group numbers use
their own sequence, and cards keep stable `Cxx` aliases. This only measures what
the model mentions: no mention is required, 0/K is valid, and diagnostics do not
change validation, retry, grammar, or prediction behavior.
