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
all groups. Each balanced20 selection is the prefix of balanced50. With historical filter v1, budget 50
actually returns 24–50 records (median 48), with 158/406 queries reaching 50.
All arms use the same original Direct 50 cards and cached single-molecule prior.

The two-stage `anchored0` / `anchored20` experiments remain historical ablations
in the results; they are not additional maintained preparation modes. Their higher
absolute scores mostly reflect the second reasoning pass: adding indirect improves
Macro-F1 only 0.798533→0.802925, with one net additional correct prediction.

## Codex handoff

Start here if integrating this change into Joseph's existing harness.

1. Fetch `codex/bbb-indirect-review-20260924` and inspect its latest commit. The
   relevant function is `prediction_only(record)` in `indirect_ablation.py`;
   regression examples are in `tests/chembl_tool/common/test_bbb_indirect_ablation.py`.
   There is one current implementation and no legacy filter flag.
2. Apply this predicate only to indirect (L2+) evidence. Joseph's archived records
   store the card under `evidence_row["prompt_evidence"]`; adapt with
   `prediction_only({"card": evidence_row["prompt_evidence"]})`. Preserve
   `assay_context`, `support_text`, `qualifying_conditions`, `extra_details`, and
   the nested `experimental_details` assay metadata. Do not flatten away assay
   qualifiers. True means exclude the card from this experiment, not delete it
   from the source database. Query labels are never passed to the predicate.
3. For the first isolated comparison, filter the existing selected 50 indirect
   records without refilling. Preserve Direct IDs/text, cached prior, prompt,
   query cohort, model, and generation settings; record the resulting budget and
   dropped IDs. Rebuild visible aliases/citation indices after filtering, and use
   fresh outputs. This isolates filtering without changing the donor selection.
   Later candidate-pool filtering/refill is a separate selection change.
4. Use the **official TDC ADMET-group BBB_Martins scaffold test set: 406 rows**,
   replacing Joseph's previous 530-row custom split. The exact test rows and labels
   are already included as `evaluation_rows` in
   `artifacts/chembl_tool/bbb_indirect_20260924/inputs.json.gz` on this branch.
   They match the upstream `admet_group/bbb_martins/test.csv` in row order,
   SMILES and labels. Keep all 406 rows, including duplicates; the 375-molecule
   overlap subset is not the full test set. This changes only which test set to
   use; it does not prescribe changes to Joseph's harness or other run settings.
5. Run the focused tests, then inspect the actual prepared requests before any
   provider run. At minimum, preserve the archived PAMPA classifications
   (#217 Record 22-1 and #472 Record 31-1), remove clearly computational-only
   SwissADME/QikProp/BOILED-Egg cards, and retain real mixed measurements. The
   prompt and 20/50 selector ablations in this guide are optional separate changes;
   do not bundle them into the first filter-only comparison.

```sh
python -m pytest -q tests/chembl_tool/common/test_bbb_indirect_ablation.py
```

The filter is heuristic. An experimental classification can still be inapplicable
to the query; this change does not resolve identity, disease-condition, or source
validity issues. Historical score improvements are not evidence that this filter
will improve Joseph's results. See the current offline receipt at
`artifacts/chembl_tool/bbb_indirect_20260924/filter_review.json`.

## Quick start

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
16,219→21,433.5. The budget increase has only a small observed benefit.
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
