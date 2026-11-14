# MolRAG reproduction

The selected reproduction, confirmed on **2026-11-08**, is **direct evidence,
4-shot Sim-CoT, low reasoning effort, with corrected answer parsing**.
[Selection record](../../experiments/baselines/molrag_selected.json) pins the
report and verification hashes. Results (internal document; not distributed in Git)
show the selected scores. This is the only maintained MolRAG setting.
This adapts [MolRAG](https://aclanthology.org/2025.acl-long.755/) to our frozen
benchmarks, reference sources and three backbones; it is not the original paper's
reported model/score or an exact label-only replication.

## Scientific contract

- All models share 4,273 frozen **test** queries: TDC 2,225 and conditioned 2,048,
  six tasks each. These are the frozen paper cohorts, not current Joseph defaults.
- Source indexes match the historical naive direct baseline. TDC pools contain
  TDC train labels plus direct source records; conditioned pools contain direct
  source records, not merely conditioned benchmark train labels.
- Full-molecule Morgan bit fingerprints (radius 2, 2,048 bits, no chirality), Dice
  ranking, and shared scaffold-disjoint filtering. The fingerprint parameters
  are reconstruction assumptions. Parents are used for leakage filtering only.
- Four distinct canonical molecules, one reference per molecule, no source quota.
  Frozen source-record order chooses the representative; source evidence precedes
  an appended TDC label for the same molecule. Dice ties use canonical SMILES.
- Each reference shows SMILES, Dice and either its TDC label or original direct
  evidence, including reported values, units and conditions. Query and original
  conditions are visible; no benchmark condition-group matching is imposed.
  Source records can be experimental, clinical or literature statements.
- No query prior, indirect evidence, tools or fine-tuning. All queries have four
  references. Legacy cards without physical UIDs retain file/line/card provenance;
  no source-row UIDs are fabricated.
- DeepSeek V4 Flash 0731 / GPT-6 Luna / Gemini 3.8 Flash all use low effort.
  Completion limits are 16,384 / 8,192 / 16,384. Temperature is 0 / omitted / 0.
  Fixed models, routes, exact prompts and generation settings live in the batch
  manifests. This is not a strict generation-settings match to historical naive.
- Explicit final YES/NO answers are parsed from text or terminal JSON wrappers.
  Conflicting markers and ambiguous/unfinished answers are rejected. The selected
  report recovers 61 saved DeepSeek answers offline; 51 remaining failures count
  as wrong only in scoring. Raw null predictions and responses remain unchanged.
  Macro-F1 and accuracy means weight six tasks equally; no AUROC is inferred.

## Code and entry points

| Entry | Responsibility |
| --- | --- |
| `python -m predict.baselines.molrag_evidence` | Stream frozen source sidecars into a direct reference library; reuse shared TDC label merging. |
| `python -m predict.baselines.molrag prepare` | Freeze references, query order, prompts and provider contract under a new run name. |
| `python -m predict.baselines.molrag repeat` | Validate and atomically create an independent run from frozen requests, with new run UIDs and no inherited predictions. |
| `python -m predict.baselines.molrag run` | Shared-provider inference, bounded attempts, successful-answer reuse and cost ledgers. |
| `python -m predict.baselines.molrag verify` | Read-only frozen-input/response validation and recomputed coverage diagnostics; `--require-complete` rejects missing answers. |
| `python -m predict.baselines.molrag score` | Offline full-cohort metrics: recover explicit saved answers and count remaining failures as wrong, without changing raw runs. |

Current prompt prose is in `predict/prompts/variants/molrag_evidence_v1/`.
`molrag_evidence.py` is a separate streaming source-export operation;
it does not duplicate inference, model transport or metrics.

Recompute the selected three-model scores into a **new** directory, with no API calls:

```bash
bash experiments/baselines/molrag.sh /tmp/molrag-offline-check
python -m pytest -q tests/predict/test_molrag.py tests/predict/test_openai_provider_pool.py
```

The selected report is
`outputs/analysis/baselines/molrag_evidence_low_reparsed_performance_20261107/`.
`results.json` lists source run UIDs/locators; `scored_predictions.jsonl` retains
original prediction/status and records recovered answers with attempt hashes.
`verification.json` and `source_preservation.json` bind the independent audit.
Raw runs stay under `outputs/predictions/molrag/molrag_evidence_v1/`.

Independent repetitions use:

```bash
python -m predict.baselines.molrag repeat --run-root "$FROZEN_RUN" \
  --run-name "$NEW_RUN_NAME" --provider-config "$PROVIDER_CONFIG"
```

This validates the source and preserves exact requests, references and cohorts,
creates new run UIDs, and excludes all predictions, attempts and resume state.
An interrupted copy leaves no partially published target directory.
The repeat path accepts the shared OpenAI-compatible chat transport with the
same model, low effort and no automatic failovers, including local endpoints.
Execute the result with `run`; record server/provider differences separately.
The [repetition recipe and offline report](../../experiments/baselines/README.md#frozen-llm-repetitions)
own matrix scheduling and five-run/new-four statistics.

The shared credential resolver uses process credentials first and only reads
`--env-file` when the credential is absent there; no empty dotenv file is needed.
A local authenticated endpoint must set `api_key_env` in its provider JSON.

The source library, preflight, provider configs and ledgers live in
`outputs/analysis/baselines/molrag_evidence_low_20261107/`.
Scripts and implementation copies inside output directories are historical audit
snapshots, not maintained entry points. Do not rerun their old finalizers.

Preparation requires a new run name, a frozen `--reference-library` and an
explicit `--provider-config`. It fixes test-only evaluation, four references,
low effort and the backbone-specific token/temperature settings above; there
are no alternate scientific setting switches. Providers must use OpenRouter,
low effort and no automatic failovers. New inference needs an explicit provider
budget; never resume a selected frozen run just to regenerate scores.

```bash
python -m predict.baselines.molrag prepare --source-root "$SOURCE_ROOT" \
  --reference-library "$REFERENCE_LIBRARY" --provider-config "$PROVIDER_CONFIG" \
  --run-name "$NEW_RUN_NAME"
python -m predict.baselines.molrag run --run-root "$NEW_RUN_ROOT" \
  --provider-config "$PROVIDER_CONFIG" --env-file "$ENV_FILE" --parallelism 32
```

The library uses `experiments/baselines/molrag_evidence_sources.json`; frozen query
splits use the shared `tinker_sft_paper_sources.json`, whose filename reflects its
original consumer rather than a separate scientific split. Use `--help` for
operational controls: concurrency, dispatch spacing and a lower attempt cap.

## Execution caveats

The selected run authorized at most two requests per query, no retry after token
exhaustion, and $5 / $3 / $25 cost caps. All three processes resumed after SQLite
ledger lock failures; the shared ledger now closes connections and serializes local
transactions. Recovery used one saved attempt per unfinished query to allow for a
possibly unrecorded request, retaining successful outputs. Gemini kept route
throttling. Reported cost is $15.050384; $0.86 remains conservatively reserved,
not confirmed billed. Offline parser correction cost $0.

Runtime concurrency may be changed on checkpoint resume with `--parallelism`
and `--provider-max-inflight` (capacity per provider). The override is recorded
in `execution_sessions.json`; frozen provider configuration, model routing,
prompts, generation settings and completed outputs remain unchanged.
