# Bioavailability_Ma DeepSeek pipeline optimization memory

## Goal

Optimize the initial DeepSeek ChEMBL-only Bioavailability_Ma reasoning pipeline while preserving the original pipeline as a reproducible baseline/v1.

Constraints:

- Do not edit the baseline behavior in-place in a way that makes the old run unreproducible.
- Keep only the baseline and the best optimized version as stable runnable variants.
- Do not overfit to audited molecules; every rule must be source-agnostic and general.
- Start with the original retrieval source: ChEMBL-only.
- Make the pipeline retrieval-source agnostic so Starling or future providers can later reuse the same evidence compiler/final aggregation.
- After a meaningful generic improvement, run evaluation; if useful, spawn subagents for failure analysis, summarize, then iterate.
- Stop when macro-F1 is hard to improve or residual failures are mostly label-unclear / externally questionable.
- Final deliverable: detailed report of implemented optimizations and performance.

## Current hypothesis from audits

The largest failure mode is evidence fusion, not lack of retrieval:

- endpoint semantics are flattened into high/low votes;
- low-transferability numeric F values still influence final labels;
- related endpoint rows are repeatedly counted;
- prodrug / active-metabolite / salt / formulation / route context is not gated;
- the 20% threshold is not enforced strongly enough;
- final-level prompt variance can flip labels without meaningful evidence changes.

## Planned implementation direction

Create a ChEMBL-only optimized pipeline variant that reuses existing retrieval and group reasoning where possible, then adds a source-agnostic final-evidence compiler and stricter final synthesis prompt.

Likely implementation points:

- add explicit `pipeline_variant` / prompt variant without changing baseline default;
- compile group outputs into evidence clusters with:
  - endpoint role / hierarchy;
  - transferability gate;
  - source molecule de-duplication;
  - 20% threshold note;
  - endpoint-to-Fa/Fg/Fh mapping when possible;
- final prompt receives structured evidence summary and hard rules.

## Working log

- 2026-06-22: Started code inspection for optimized ChEMBL-only source-agnostic pipeline variant.
- 2026-06-22: Added first optimized variant without changing the original v1 entrypoints:
  - `tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py`
    - source-agnostic final evidence compiler;
    - maps endpoint groups/rows to endpoint roles and F/Fa/Fg/Fh/formulation context;
    - allows direct label votes only for direct oral F evidence that passes transferability, confidence, usefulness, and interpretable threshold gates;
    - tracks duplicate/correlated source molecules so repeated evidence is not counted independently.
  - `tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_v2.py`
    - reuses v1 retrieval and single/group reasoning;
    - monkeypatches only the final reasoning function to use the compiler and stricter final prompt.
  - `tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_v2.py`
    - thin batch wrapper pointing to the v2 pipeline module.
  - `tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py`
    - covers direct F vote gate, low-transferability blocking, proxy endpoint roles, and 20% threshold boundary.

Verification:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_neighbor_retrieval.py

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_v2.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_v2.py
```

Both passed. Next step: run a ChEMBL-only final-only v2 batch reusing an existing v1 ChEMBL retrieval/single/group batch, then compare metrics against v1.

- 2026-06-22: Ran ChEMBL all-tier final-only v2 experiments against source batch
  `outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_20260512`.

Baseline all-tier ChEMBL v1:

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_20260512
accuracy 0.71875
macro-F1 0.676208
confusion matrix from metrics: TN 22 / FP 9 / FN 27 / TP 70
```

First v2 final compiler run:

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_final_20260622
accuracy 0.71875
macro-F1 0.681064
confusion: TN 24 / FP 7 / FN 29 / TP 68
```

Unit-fix v2 run, current best:

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_unitfix_final_20260622
fix: standard_type=F_fraction values such as 0.48 are converted to 48%, and direct F cards include compiler_threshold_direction/threshold_summary so final reasoning can override inconsistent group-level evidence_direction.
accuracy 0.734375
macro-F1 0.686546
confusion: TN 22 / FP 9 / FN 25 / TP 72
prediction_distribution: high 81 / low 47
relative to v1: +2 correct, macro-F1 +0.010338
```

Molecule-level threshold-direction experiment, rejected:

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_molsummary_final_20260622
change tested: use source-molecule-level threshold direction rather than row-level threshold direction.
accuracy 0.710938
macro-F1 0.677014
confusion: TN 23 / FP 8 / FN 27 / TP 68, with 2 missing outputs
decision: worse than unit-fix and baseline; code was reverted to row-level compiler_threshold_direction while retaining source_molecule_direction_counts only as audit metadata.
```

Observed after current best run:

- v2 unit-fix changed 14 predictions versus v1 all-tier: 8 improved and 6 regressed.
- The strongest gain came from correcting direct F fraction scaling and forcing final reasoning to respect direct numeric threshold summaries.
- Remaining major errors are mostly false negatives: high-label molecules predicted low when direct analog evidence is weak/low-transferability, or when proxy metabolism/solubility risk is over-weighted.

- 2026-06-22: Spawned four subagents for local-artifact failure analysis of all 34 v2 unit-fix failures. Reports were summarized in:

```text
outputs/chembl_tool/tasks/bioavailability_ma/analysis/pipeline_optimization_v2_unitfix_failure_audit_20260622/decision_view_iteration_summary.md
```

Convergent subagent findings:

- blocked or low-transferability direct F values still leaked into final labels as weak directional evidence;
- proxy endpoints were still overinterpreted against the 20% F threshold;
- mixed/balanced direct F evidence was often turned into a moderate-confidence binary call via qualitative SAR speculation;
- no-direct-evidence cases were too dependent on single-molecule priors;
- several remaining errors may be label/source-coverage/exact-evidence issues rather than clear reasoning failures.

Implemented next generalizable change:

- `evidence_compiler.py` now emits `llm_decision_view`;
- `run_reasoning_pipeline_v2.py` sends `llm_decision_view` to final LLM instead of the full compiled evidence object;
- allowed direct F votes retain threshold values;
- blocked numeric evidence is redacted to gate reason/counts/caveats, so it can only act as uncertainty/context, not weak directional support;
- proxy evidence is explicitly labeled as mechanism context.

Decision-view run:

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_decisionview_final_20260622
accuracy 0.757812
macro-F1 0.712109
confusion: TN 23 / FP 8 / FN 23 / TP 74
prediction_distribution: high 82 / low 46
relative to v1: accuracy +0.0390625, macro-F1 +0.035901
relative to unit-fix: 11 prediction flips, 7 improved / 4 regressed
```

Verification after decision-view implementation:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_neighbor_retrieval.py

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_v2.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_v2.py
```

Both passed. Next step: run fresh failure analysis on the new current-best decision-view batch, which has 31 failures (23 FN / 8 FP).

- 2026-06-22: Ran a rejected deterministic direct-F post-hoc guardrail experiment.

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_directguard_final_20260622
change tested: after final LLM output, force the class to match unanimous allowed direct-F threshold votes.
accuracy 0.703125
macro-F1 0.659098
confusion: TN 22 / FP 9 / FN 29 / TP 68
prediction_distribution: high 77 / low 51
decision: rejected. This is worse than v1 and much worse than decision-view. Code was reverted; do not keep post-hoc deterministic answer rewriting as a stable optimization.
```

Current stable optimized variant remains the decision-view v2:

```text
tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py
tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_v2.py
tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_v2.py
```

It is retrieval-source agnostic by design: evidence is compiled by endpoint role, threshold relation,
transferability, and source molecule identity. ChEMBL-specific and Starling-specific fields are treated
as provider metadata and should not drive final rules.

Verification after reverting directguard:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_neighbor_retrieval.py

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_v2.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_v2.py
```

Both passed.

- 2026-06-22: Ran exact-overlap stratification audit for the current-best decision-view batch.

Output:

```text
outputs/chembl_tool/tasks/bioavailability_ma/analysis/pipeline_optimization_v2_decisionview_failure_audit_20260622/decisionview_exact_overlap_audit.json
outputs/chembl_tool/tasks/bioavailability_ma/analysis/pipeline_optimization_v2_decisionview_failure_audit_20260622/decisionview_exact_overlap_audit.tsv
outputs/chembl_tool/tasks/bioavailability_ma/analysis/pipeline_optimization_v2_decisionview_failure_audit_20260622/decisionview_exact_overlap_audit.md
```

Counts:

```text
n_total: 128
n_failures: 31
exact_match_found: 128
exact_relevant_rows_found: 115
exact_numeric_threshold_found: 109
failures_with_exact_numeric_threshold: 28
failures_with_concordant_exact_numeric: 13
failures_with_discordant_exact_numeric: 13
```

Interpretation:

- The current benchmark intentionally disables exact ChEMBL context, so this audit must not be treated as a new main result.
- A large share of residual failures has exact query evidence available locally in the ChEMBL-derived evidence library.
- Optimizing analog-only prompts around these residual samples risks fitting the test-set overlap rather than improving a prospective retrieval/reasoning pipeline.
- Next mainline work should keep ChEMBL-only retrieval as the fixed source, improve source-agnostic evidence handling only when the rule generalizes across providers, and later evaluate Starling/combined as retrieval-provider swaps using the same compiler.

- 2026-06-22: User found that blocked numeric evidence still leaked raw values through group `reasoning_summary`
  and `caveats`, e.g. `Fa=0.06 (6% oral absorption in humans)`. Implemented a strict redaction test version
  that removed raw `reasoning_summary` / `caveats` from blocked cards and reran a full final-only batch.

Strict redacted-blocked run:

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_redactedblocked_final_vllm_20260622
source batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_20260512
accuracy 0.726562
macro-F1 0.679611
confusion: TN 22 / FP 9 / FN 26 / TP 71
prediction_distribution: high 80 / low 48
```

Comparison:

```text
v1 baseline macro-F1: 0.676208
old v2 decision-view macro-F1: 0.712109
strict redacted-blocked macro-F1: 0.679611
```

Strict redaction is therefore rejected as a stable policy: it prevents numeric leakage but loses most of the
old v2 gain.

Failure audit:

```text
outputs/chembl_tool/tasks/bioavailability_ma/analysis/pipeline_optimization_v2_redactedblocked_failure_audit_20260622/
  failure_samples_compact.json
  redactedblocked_failure_audit_readme.md
  subagent_group1_report.md
  subagent_group2_report.md
  subagent_group3_report.md
  subagent_group4_report.md
  redactedblocked_failure_audit_summary_zh.md
```

Four subagents audited all 35 strict-redaction failures. Convergent conclusion:

- Most failures were already failures in old decision-view and reflect retrieval gaps, analog-transfer errors,
  proxy weighting, or label/source ambiguity.
- Five cases were clear regressions from old decision-view: indices 11, 21, 66, 82, 117.
- The regression was not caused by hiding raw numbers alone; it was caused by hiding nonnumeric qualitative
  transferability/context summaries and caveats.

Current code has been updated from strict redaction to fine-grained redaction:

- Blocked cards still hide raw `threshold_evidence` and `compiler_threshold_direction`.
- Raw numeric values, units, and percent tokens are scrubbed from blocked `reasoning_summary` / `caveats`.
- Qualitative `redacted_reasoning_summary` and `redacted_caveats` are retained so final LLM can still see
  transferability rationale, endpoint mismatch, species/formulation caveats, and mechanism context.

Verification after fine-grained redaction implementation:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_neighbor_retrieval.py

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_v2.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_v2.py
```

Both passed. Full final-only batch for the fine-grained version has not yet been run; the measured full-batch
metric above is for strict redaction.

Superseding verification on 2026-06-22:

The fine-grained redaction version was fully tested in the correct `vllm` environment and rejected on
performance grounds.

Fine-grained redaction full final-only run:

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_finegrainedredaction_final_20260622
source batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_20260512
accuracy 0.734375
macro-F1 0.686546
confusion: TN 22 / FP 9 / FN 25 / TP 72
prediction_distribution: high 81 / low 47
```

Comparison:

```text
v1 baseline macro-F1: 0.676208
old v2 decision-view macro-F1: 0.712109
strict redacted-blocked macro-F1: 0.679611
fine-grained redaction macro-F1: 0.686546
```

Final code decision:

- Do not keep the strict redaction or fine-grained redaction behavior as the current v2 implementation.
- The current v2 code has been reverted to the highest-tested old decision-view behavior:
  blocked numeric groups still omit structured `threshold_evidence` and `compiler_threshold_direction`,
  but retain the group-level `reasoning_summary` and `caveats` as qualitative context for final reasoning.
- This means the known blocked-summary numeric leakage risk remains documented, but the tested performance
  regression from redaction is not kept in the active code.

Post-revert verification:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_neighbor_retrieval.py

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_v2.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_v2.py
```

Both passed after reverting the redaction code.
