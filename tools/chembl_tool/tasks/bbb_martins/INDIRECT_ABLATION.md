# Reproduce the TDC BBB indirect ablation

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

For each indirect card, concatenate `str(assay_context) + ' ' + support_text`.
Remove the entire card only if a case-insensitive prediction pattern matches and
no measurement pattern matches. Exact regexes live in `indirect_ablation.py`:
QikProp, SwissADME, ADMETLab, ADMET(S)SAR, AdaBoost, in-silico, predicted, prediction,
or computational versus measured, experiment/experimental/experimentally, in vivo,
in vitro, microdialysis, or perfusion. Mixed text is retained. This is a lexical
heuristic: even “not measured” prevents removal. Source files and Direct cards are
not changed; the rule neither reads query labels nor evaluates correctness.

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
all groups. Each balanced20 selection is the prefix of balanced50. Budget 50
actually returns 24–50 records (median 48), with 158/406 queries reaching 50.
All arms use the same original Direct 50 cards and cached single-molecule prior.

The two-stage `anchored0` / `anchored20` experiments remain historical ablations
in the results; they are not additional maintained preparation modes. Their higher
absolute scores mostly reflect the second reasoning pass: adding indirect improves
Macro-F1 only 0.798533→0.802925, with one net additional correct prediction.

## Quick start

Run from the repository root in the existing TxAgent Python environment (on
node001/node002, activate `vllm`). The bundled inputs remove the dependency on the
author's `outputs/` directory, retrieval service and original absolute data paths.
The compressed bundle is approximately 15 MB. Its manifest pins its checksum and
all original source selections; labels are kept separately from rendered messages.

Prepare the matched control and 20/50 budget arms, without API calls:

```sh
python -m tools.chembl_tool.tasks.bbb_martins.indirect_ablation \
  --output-root outputs/bbb_indirect_replay \
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
  --benchmark-manifest outputs/bbb_indirect_replay/benchmark_manifest.json \
  --output-root outputs/bbb_indirect_replay/balanced20 \
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

## Observed full-test results

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
16,219→21,433.5. The budget increase has only a small observed benefit.
The prompt and compact rendering changed together; their Direct-only result
regresses versus the original. Therefore the experiment supports an incremental
indirect benefit under the new prompt, not a standalone benefit of the prompt.
Compared with original Direct, Macro-F1 improves but accuracy remains lower.
Triclofos/foscarnet identity interpretation errors remain unresolved.

`comparison.json` and `predictions.csv` contain all eight result rows including
original baselines and historical two-stage ablations. `verification.json` pins
2,436 new outputs; `refactor_parity.json` confirms exact messages and selected IDs
for all 1,624 one-pass requests after code cleanup. Raw outputs, provider retries
and historical preparation scripts remain frozen locally under the registered
experiment root; they are not required to generate the portable one-pass inputs.
