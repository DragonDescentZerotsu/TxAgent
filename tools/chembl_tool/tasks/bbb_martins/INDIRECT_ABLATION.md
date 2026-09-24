# Run the TDC BBB indirect ablation

This is test-tuned exploratory work on 406 TDC BBB queries, not untouched-test
validation. It does not replace the original Direct 50 / Direct 50 + Indirect 50
baseline. All six completed configurations and per-query predictions are retained
in `artifacts/chembl_tool/bbb_indirect_20260924/`.

## What the maintained code does

`indirect_ablation.py` prepares inputs only. The existing family runner executes
requests through `paper_experiments/prepared_evidence.py`, using the shared provider,
JSON/citation validation, checkpoint/resume and racing-retry machinery. The shared
`common/record_budget.py` owns deterministic simple selection; no task-local copy
of the selector or inference client is used.

The compact prompt is `indirect_applicability_v1.txt`. It asks for query/donor
attribution, endpoint/condition applicability, and distinguishing predictions from
measurements. It does not establish that TDC BBB measures passive permeability.

Preparation uses **prediction filter v2**, the only maintained filter. It inspects the source
context, supporting text, qualifying conditions, extra details and selected assay
metadata. Experimental PAMPA/MDCK and related assay results remain eligible even
when their interpretation says “predicted/classified CNS+”. Explicit computational
methods (including SwissADME, QikProp, BOILED-Egg, pkCSM and QSAR) do not become
experiments merely by mentioning PAMPA, MDCK or an in-vitro endpoint. Negated
measurement statements do not protect prediction-only cards. Mixed evidence with
affirmative measurements is retained. Source files, Direct cards, the prompt and
the cached query prior are unchanged; filtering never reads query labels.

This remains a lexical screen, not a guarantee of source validity or query
applicability. Ambiguous records without prediction indicators are retained; a
measured result mentioned in a mixed passage may concern another molecule. Do
not interpret retained records as automatically experimental or transferable.

**All scores below and the frozen input-parity receipt used the retired filter.**
The old regex implementation and version-selection CLI have been removed. Git
history contains that earlier implementation if historical investigation is needed.
The current preparation manifest and receipt pin `prediction_filter: v2` as
provenance, not as a selectable mode. Use a fresh output root: old predictions
cannot be reused when the selected evidence changes. The current filter has only
offline verification; no new LLM accuracy/F1 result has been established.

| Arm | Indirect selection |
|---|---|
| `direct_guard` | None; matched compact-prompt control |
| `guard50` | Original Neighbor-fill 50 minus filtered records; no refill |
| `balanced20` | Group-balanced budget 20, at most 3 records per donor |
| `balanced50` | Identical settings with budget 50 |

Balanced arms use the ID-deduplicated union of the original saved `both_fill`,
`both_cap`, and `both_group` selections, not the complete retrieval index.
Groups take turns in sorted order; within groups, donors follow similarity order
and records use a deterministic seed-0 ID hash order. The donor cap applies across
all groups. Each balanced20 selection is the prefix of balanced50. With the current
filter, balanced20 fills all 406 budgets; balanced50 returns 24–50 records, with
168/406 queries reaching 50. Counts and offline checks are in the
[filter receipt](../../../../artifacts/chembl_tool/bbb_indirect_20260924/filter_review.json).
All arms use the same original Direct 50 cards and cached single-molecule prior.

The two-stage `anchored0` / `anchored20` experiments remain historical ablations
in the results; they are not additional maintained preparation modes. Their higher
absolute scores mostly reflect the second reasoning pass: adding indirect improves
Macro-F1 only 0.798533→0.802925, with one net additional correct prediction.

## Codex handoff

Use the **official TDC ADMET-group BBB_Martins scaffold test set: 406 rows**,
replacing Joseph's previous 530-row custom split. The exact rows and labels are
`evaluation_rows` in `artifacts/chembl_tool/bbb_indirect_20260924/inputs.json.gz`.
They match the upstream `admet_group/bbb_martins/test.csv` in row order, SMILES and
labels. Keep all 406 rows, including duplicates; the 375-molecule overlap subset
is not the full test set.

For the filter, use the single maintained `prediction_only(record)` function in
`indirect_ablation.py`. Joseph's indirect (L2+) cards can be passed directly as:

```python
prediction_only({"card": evidence_row["prompt_evidence"]})
```

True means exclude the card from the selected evidence, not delete source data.
Pass the complete card so assay context and nested experimental details remain
available. The predicate does not use query labels. Focused regression examples
are in `tests/chembl_tool/common/test_bbb_indirect_ablation.py`.

The filter has offline verification only, with no new LLM scores. It does not
resolve identity, condition applicability or source validity. Joseph can keep his
existing harness and run settings; the preparation and replay commands below are
an optional way to reproduce our separate compact-prompt ablations.

## Optional: reproduce our compact-prompt ablations

Run from the repository root in the existing TxAgent Python environment (on
node001/node002, activate `vllm`). The bundled inputs remove the dependency on the
author's `outputs/` directory, retrieval service and original absolute data paths.
The compressed bundle is approximately 15 MB. Its manifest pins its checksum and
all original source selections; labels are kept separately from rendered messages.

Prepare new v2 inputs for the matched control and 20/50 budget arms, without API calls:

```sh
python -m tools.chembl_tool.tasks.bbb_martins.indirect_ablation \
  --output-root outputs/bbb_indirect_filter_v2 \
  --arms direct_guard balanced20 balanced50
```

Put your key in the local, Git-ignored `.env` as `OPENROUTER_API_KEY=...`.
The public provider example stores only this variable name. It fixes
`deepseek/deepseek-v4-flash-0731`, temperature 1, low reasoning effort,
360-second HTTP timeout, and a shared 512-request admission socket.
The original run used 180 seconds, increased to 360 for two timeout recoveries;
timeout changes do not change scientific messages or generation settings.

Start the broker in another terminal:

```sh
python -m tools.chembl_tool.common.request_admission \
  --socket /tmp/txagent-bbb-indirect.sock --capacity 512
```

Run one arm (change the final directory to `direct_guard` or `balanced50` for the
other comparisons). For an initial single-query check, append `--indices 0`;
remove it for the full 406-query run. Append `--prepare-only` for offline validation.

```sh
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve \
  --run-prepared --tasks bbb_martins \
  --benchmark-manifest outputs/bbb_indirect_filter_v2/benchmark_manifest.json \
  --output-root outputs/bbb_indirect_filter_v2/balanced20 \
  --model deepseek/deepseek-v4-flash-0731 \
  --provider-pool-config artifacts/chembl_tool/bbb_indirect_20260924/provider.example.json \
  --max-tokens 16384 --parallelism 512 --endpoint-concurrency-budget 512 \
  --retry-race-width 6 --max-stage-requeues 2 --retry-delay-s 2
```

The arms can run concurrently in separate terminals; the broker enforces the
shared total capacity of 512, including retry races. Retries choose the first
schema/citation-valid response, never the response matching gold. Stop the broker
once all arms finish. Valid completed checkpoints are reused; use a fresh output
root when changing scientific inputs. Scores are written to
`<arm>/summary/bbb_martins/metrics.json`, with actual requests/results under
`<arm>/runs/bbb_martins/query_*/`. Check `execution_status.json` for completion.
Model outputs are stochastic; reproducing identical inputs does not guarantee
identical predictions or metrics.

## Observed full-test results (filter v1)

| Configuration | Macro-F1 | Accuracy |
|---|---:|---:|
| Original Direct 50 | 0.772563 | 0.869458 |
| Original Direct 50 + Indirect 50 | 0.764288 | 0.866995 |
| New Direct control | 0.762083 | 0.830049 |
| + Filtered ≤50 | 0.782772 | 0.844828 |
| + Balanced 20 | 0.792696 | 0.849754 |
| + Balanced budget 50 | 0.795046 | 0.854680 |

Balanced20 corrects 24 and damages 16 decisions versus its matched Direct control.
Balanced50 corrects 20 and damages 18 versus balanced20; median prompt tokens rise
16,219→21,433.5. These historical runs used 24–50 records (median 48), with
158/406 queries reaching 50. The budget increase has only a small observed benefit.
The prompt and compact rendering changed together; their Direct-only result
regresses versus the original. Therefore the experiment supports an incremental
indirect benefit under the new prompt, not a standalone benefit of the prompt.
Compared with original Direct, Macro-F1 improves but accuracy remains lower.
Triclofos/foscarnet identity interpretation errors remain unresolved.

`comparison.json` and `predictions.csv` contain all eight result rows including
original baselines and historical two-stage ablations. `verification.json` pins
2,436 new outputs; `refactor_parity.json` confirms exact messages and selected IDs
for all 1,624 historical one-pass requests after the earlier code cleanup; it is not a parity claim for the current filter. Raw outputs, provider retries
and historical preparation scripts remain frozen locally under the registered
experiment root; they are not required to generate the portable one-pass inputs.
