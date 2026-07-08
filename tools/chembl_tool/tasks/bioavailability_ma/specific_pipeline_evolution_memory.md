# Bioavailability_Ma specific Fa/Fg/Fh pipeline evolution memory

## Goal

Build a Bioavailability_Ma-specific reasoning pipeline, independent from the existing v1 and v2 pipelines, that reuses the best available retrieval/group artifacts but changes final synthesis from a flat endpoint vote model to a pharmacokinetic factor model:

```text
F = Fa * Fg * Fh
```

This line starts from the best combined ChEMBL + Starling v2 complete threshold/salt run:

```text
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622
accuracy 0.796875
macro-F1 0.748792
residual errors 26 / 128
```

The main residual-error hypothesis is that many failures are not caused by missing retrieval alone. They are caused by final-stage evidence fusion that mixes parent F, prodrug/active-metabolite exposure, absorption proxies, first-pass metabolism, species, dose, food, salt, and formulation context.

## Design constraints

- Keep v1 and v2 entrypoints reproducible; do not change their behavior in place.
- Reuse the existing retrieval sources first, especially the best merged ChEMBL + Starling v2 source batch.
- Make the specific pipeline retrieval-source agnostic so future Starling-only or Starling-upgraded retrieval can feed the same compiler.
- Do not optimize case-by-case molecule names. Rules must generalize across sources and chemistry.
- After each meaningful final-prompt/compiler change, run a final-only batch, compare macro-F1 against the best merged v2 threshold/salt run, and audit failures.

## First implementation direction

Add a new final compiler and new pipeline wrappers:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py
tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py
tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_specific.py
tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py
```

The new compiler calls the existing v2 source-agnostic `compile_final_evidence()` and then rewrites its output into:

```text
direct_parent_f_evidence
factor_evidence:
  Fa
  Fg
  Fh
context_flags:
  prodrug_or_active_metabolite
  species_translation
  dose_or_steady_state
  food_formulation_salt_or_route
  threshold_sensitive_15_25_percent
  first_pass_or_clearance_mechanism
```

The final prompt then reasons in this order:

```text
1. clean parent absolute oral F evidence
2. parent/analyte/prodrug/active-moiety context gates
3. Fa evidence
4. Fg evidence
5. Fh evidence
6. forced high/low decision with confidence and label ambiguity flags
```

## 2026-06-24 working log

- Created the first independent Bioavailability-specific compiler and batch/pipeline wrappers.
- The implementation preserves retrieval, single-molecule, and group-level reasoning from `run_reasoning_pipeline.py`; only final synthesis is replaced.
- The specific compiler currently collapses old group roles into Fa/Fg/Fh rather than deleting old retrieval groups, because keeping retrieval unchanged lets us run final-only comparisons against the best existing merged source batch.
- Added tests for:
  - absorption evidence folding into Fa;
  - hepatic metabolism/clearance folding into Fh;
  - gut metabolism / efflux folding into Fg;
  - active-metabolite/prodrug language forcing parent/analyte review;
  - species and threshold-sensitive flags.

Next verification target:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_specific.py
```

Verification result:

```text
2026-06-24:
  pytest test_specific_evidence_compiler.py + test_evidence_compiler.py
    11 passed
  py_compile specific_evidence_compiler.py run_reasoning_pipeline_specific.py run_reasoning_batch_specific.py
    passed
  run_reasoning_batch_specific --help
    passed
```

Focused final-only smoke result, reusing the best merged source batch and rerunning only residual metabolism/scope-sensitive failures:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --final-only-source-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622 \
  --indices 21 24 80 82 118 120 \
  --parallelism 2 \
  --skip-existing \
  --timeout-s 300 \
  --max-tokens 8192
```

```text
bioavailability_ma_specific_fa_fg_fh_smoke_20260624:
  initial specific final compiler
  n=6, accuracy 0.166667, macro-F1 0.142857
  corrected only idx118 / melagatran

bioavailability_ma_specific_fa_fg_fh_smoke_v2_20260624:
  added clean direct-F analog vs parent/analyte-review-required split
  n=6, accuracy 0.333333, macro-F1 0.25
  corrected idx21 / felodipine and idx118 / melagatran
  remaining failures:
    idx24 / nisoldipine: still overuses high nifedipine analog direct-F evidence
    idx80 / propranolol: still lets a borderline low analog plus first-pass wording override the high label
    idx82 / nabumetone: still treats lack of clean parent-F plus metabolism as low, exposing parent-vs-active-moiety label ambiguity
    idx120 / trandolapril: still overuses review-required enalapril / trandolaprilat active-moiety evidence

bioavailability_ma_specific_fa_fg_fh_smoke_v3_20260624:
  tried soft interpretation hints on top of v2
  n=6, accuracy 0.166667, macro-F1 0.142857
  regressed idx21, so the soft hints were removed from code
```

Current conclusion:

```text
The specific final-only layer is useful but insufficient. The best smoke variant so far is the direct-F
analog / parent-analyte split, but it only fixes 2 / 6 targeted residual failures. Do not launch a full
128-molecule rerun yet as the next main step. The bottleneck is now upstream of the final prompt:
old group-level outputs already summarize evidence using the older flat high/low framing, and they do not
separate parent direct-F, active-metabolite/prodrug exposure, Fa, Fg, and Fh strongly enough.

Next better iteration should change the group/retrieval schema, not just final wording:
  1. direct parent-F group: require administered molecule, measured analyte, species, route, formulation, and
     threshold-context fields;
  2. Fa group: absorption/permeability/solubility/dissolution only, explicitly non-decisive for systemic F;
  3. Fg group: gut-wall metabolism, intestinal CYP3A/P-gp/BCRP/efflux, food effects with context;
  4. Fh group: hepatic clearance, extraction ratio, microsomal/hepatocyte stability, CYP/UGT first-pass;
  5. final compiler: consume these structured factor-specific group outputs and treat review-required
     active-moiety/prodrug evidence as non-clean unless matched to the benchmark parent endpoint.
```

Full evaluation command when a stronger group/retrieval-schema iteration is ready:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_test_full_plus_starling_v2_complete_specific_fa_fg_fh_final_20260624 \
  --final-only-source-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622 \
  --parallelism <n> \
  --skip-existing
```

## 2026-06-24 specific group pipeline iteration

Implementation changed from final-only regrouping to a real Bioavailability-specific group surface:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_retrieval.py
  Regroups source retrieval rows into:
    Observed.direct_parent_f_and_systemic_exposure
    Fa.absorption_solubility_permeability
    Fg.gut_wall_efflux_intestinal_metabolism
    Fh.hepatic_clearance_metabolic_stability

tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py
  Monkeypatches the base pipeline retrieval and group prompts for the specific pipeline:
    - retrieval now uses specific_retrieval.retrieve_specific_neighbors()
    - group prompts analyze one typed factor group instead of old Tier.endpoint_group buckets
    - single-molecule prompt is property-only and explicitly forbids drug-name / known-PK leakage
    - final prompt consumes evidence_gate_summary

tools/chembl_tool/tasks/bioavailability_ma/evidence_compiler.py
  Tightened percent-threshold extraction:
    - true F / F_fraction / bioavailability / Fa fraction can enter numeric threshold context
    - CL, CLint, logCL, recovery, stability, Papp, ratios, IC50, etc. are no longer treated as oral-F %
```

Verification after implementation:

```text
pytest specific_retrieval + specific_evidence_compiler + specific pipeline parser + evidence_compiler:
  18 passed
py_compile evidence_compiler.py specific_retrieval.py specific_evidence_compiler.py
  run_reasoning_pipeline_specific.py run_reasoning_batch_specific.py:
  passed
```

### Smoke iterations on targeted metabolism/scope failures

Targeted indices:

```text
21, 24, 80, 82, 118, 120
```

These are intentionally hard residual failures enriched for direct-F analog transfer,
metabolism/first-pass, species/context, and parent/prodrug/active-metabolite scope.

```text
bioavailability_ma_specific_fa_fg_fh_group_v2_smoke_20260624:
  first real Observed/Fa/Fg/Fh group-prompt run
  n=6, accuracy 0.0, macro-F1 0.0
  predictions:
    21 high wrong
    24 high wrong
    80 low wrong
    82 low wrong
    118 high wrong
    120 high wrong
  failure mode:
    typed grouping improved auditability but made old biases more confident.
    direct-F Starling rows were copied into Fg/Fh factor groups, and non-F numeric endpoints
    such as CL/recovery/permeability/ratios leaked into threshold summaries.
    One group response also failed JSON parsing because of an invalid backslash escape.
  status: rejected.

bioavailability_ma_specific_fa_fg_fh_group_v3_roleunit_smoke_20260624:
  fixed role/unit leakage:
    - direct-F and oral exposure rows now stay only in Observed group
    - Fa/Fg/Fh groups contain only their own factor rows
    - evidence_compiler no longer thresholds CL/recovery/Papp/ratio values as F%
    - parser repairs invalid JSON backslash escapes
    - Observed direct-F prompt has stronger source de-correlation / transferability gate
  n=6, accuracy 0.333333, macro-F1 0.333333
  corrected:
    24 nisoldipine -> low
    82 nabumetone-like prodrug/scope case -> high
  remaining wrong:
    21 felodipine-like DHP -> high, low confidence
    80 propranolol -> low, low confidence
    118 melagatran -> high, moderate confidence
    120 trandolapril -> high, low confidence
  status: useful but not full-run-ready.

bioavailability_ma_specific_fa_fg_fh_group_v4_gate_finalonly_smoke_20260624:
  final-only rerun from v3 group artifacts after adding evidence_gate_summary:
    - review-required direct-F evidence forbidden as direct label vote
    - no clean direct-F => factor/prior-based forced choice
  n=6, accuracy 0.666667, macro-F1 0.4
  corrected all four low-label cases:
    21, 24, 118, 120
  but predicted every sample low:
    80 and 82 regressed / remained false negatives
  status: rejected as too conservative; useful evidence that the review-direct gate works but harms positive recall.

bioavailability_ma_specific_fa_fg_fh_group_v5_propertyonly_gate_smoke_20260624:
  full group rerun with property-only single-molecule prompt and softened no-direct-F contract:
    - single stage must ignore drug names / known PK facts
    - no clean direct-F does not imply low by default
  n=6, accuracy 0.5, macro-F1 0.333334
  corrected:
    24, 118, 120
  wrong:
    21 high
    80 low
    82 low
  status: not full-run-ready; still too low-biased for positive class and loses the v3 nabumetone fix.
```

### Subagent failure-audit consensus

Six read-only subagents audited idx21/24/80/82/118/120 from the v2 group smoke.
Common failure patterns:

```text
1. Endpoint-unit impurity:
   CL/logCL/recovery/Papp/ratios were sometimes summarized as if they were F% threshold evidence.
   Fixed in evidence_compiler.py.

2. Evidence-role duplication:
   Starling/direct-F rows copied into Observed and factor groups, making one source cluster feel like
   multiple independent signals.
   Fixed in specific_retrieval.py by making direct-F and oral-exposure rows terminal Observed assignments.

3. Review-required direct-F overuse:
   Species/prodrug/active-metabolite/parent-analyte/context-mismatched values were still used as decisive
   high/low evidence by final LLM.
   Partially fixed by evidence_gate_summary, but v4/v5 show the gate can become too conservative.

4. Parent-scope overcorrection:
   For prodrug/active-moiety cases, forcing unchanged-parent F can turn label-scope ambiguity into false low.
   Need a dataset-scope gate: prodrug/active-moiety logic should become uncertainty unless the benchmark
   scope is proven.

5. Drug-name / known-PK leakage:
   Single-molecule outputs sometimes cited known drug identity or clinical PK.
   Specific pipeline now uses a property-only single prompt, but full impact remains mixed.
```

### Current next step

Do not run full 128 yet. The best current specific group version is not stable enough:

```text
v3 preserves some positive recall and fixes 24/82 but misses 21/80/118/120.
v4/v5 fix more low-label targeted cases but collapse positive recall.
```

Next iteration should keep the v3 role/unit fixes and add a narrower gate:

```text
- Review-required direct-F is forbidden as direct label vote.
- But no-clean-direct / factor-only cases must not default low.
- Prodrug/active-moiety scope ambiguity should be uncertainty, not directional low, unless direct low F or
  high-confidence transferable clearance evidence exists.
- Add source-level de-correlation / threshold-straddling summaries for Observed direct-F, so one high analog
  cannot dominate if same chemotype has low analog values.
- Consider a deterministic source-level direct-F aggregator before LLM final synthesis.
```

### 2026-06-24 source-level gate / final-only iterations from v5 group artifacts

All runs below reuse group artifacts from:

```text
bioavailability_ma_specific_fa_fg_fh_group_v5_propertyonly_gate_smoke_20260624
indices: 21, 24, 80, 82, 118, 120
```

Implementation changes now covered by targeted tests:

```text
tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py
tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py
tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py
tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py

latest targeted test check after source-level gate:
  22 passed
```

```text
bioavailability_ma_specific_fa_fg_fh_group_v7_redacted_review_finalonly_smoke_20260624:
  added source-level direct-F aggregation and redacted review-required direct-F threshold fields/counts
  n=6, accuracy 0.166667, macro-F1 0.142857
  only corrected:
    118
  failure:
    review-required values still leaked through contextual Observed direct-F narrative and redacted-card summaries.
  status: rejected; useful for exposing narrative-layer leakage.

bioavailability_ma_specific_fa_fg_fh_group_v8_full_review_redaction_finalonly_smoke_20260624:
  redacted review-required direct-F recursively from LLM decision view:
    - no threshold_evidence / threshold_summary direction
    - no review-required reasoning_summary/caveat numeric narrative
    - direct-F Observed groups removed from generic contextual_evidence
  n=6, accuracy 0.5, macro-F1 0.485714
  corrected:
    80, 118, 120
  wrong:
    21 high
    24 high
    82 low
  status: useful; keep recursive review-required redaction.

bioavailability_ma_specific_fa_fg_fh_group_v9_factor_effect_scope_finalonly_smoke_20260624:
  added factor_effect propagation and narrowed parent/analyte review trigger so generic "parent F" wording
  does not automatically become review-required.
  n=6, accuracy 0.333333, macro-F1 0.25
  corrected:
    118, 120
  wrong:
    21 high
    24 high
    80 low
    82 low
  failure:
    clean/review was still too coarse. Direct-F group-level raw narrative and all source rows remained visible,
    allowing borderline propranolol-like direct-F and non-transferable analog values to dominate.
  status: rejected as a whole, but keep factor_effect propagation and avoid overly broad parent-scope triggers.

bioavailability_ma_specific_fa_fg_fh_group_v10_sourcegate_finalonly_smoke_20260624:
  source-level direct-F gate now:
    - omits raw group-level direct-F narrative from LLM decision view
    - uses source-level de-correlated direct-F summary as authoritative
    - uses per-source key_evidence.transferability to decide eligible_clean_direct_vote vs clean_but_not_direct_vote
    - redacts values/directions for context-only source buckets in LLM view
  n=6, accuracy 0.666667, macro-F1 0.625
  corrected:
    21, 80, 118, 120
  wrong:
    24 high
    82 low
  status: current best targeted specific-pipeline final-only iteration; still not full-run-ready.
```

v10 remaining common failure modes:

```text
1. idx24 false high:
   DHP direct-F source summary is still mixed, and final LLM converts high Fa + uncertain Fg + moderate Fh
   into an unsupported numeric product above 20%. Need a factor-product guard:
   do not impute Fg/Fh quantitative lower bounds unless the evidence gives explicit/high-confidence support.

2. idx82 false low:
   No clean direct-F vote. Final LLM over-composes moderate analog-only solubility/CYP/microsomal risks into
   F < 20%. Need factor-only low guard:
   with no label-admissible direct F, low should require high-transferability in vivo limiting evidence,
   a strong property prior such as highly polar/zwitterionic poor permeability, or multiple independent close
   analogs with coherent limiting-factor evidence.

3. Deterministic pre-final guidance is probably needed:
   warnings in free text are insufficient. Add structured "factor_product_guard" / "factor_only_low_guard"
   fields in evidence_gate_summary or llm_decision_view, then test via final-only smoke before full 128.
```

```text
bioavailability_ma_specific_fa_fg_fh_group_v11_factor_guard_finalonly_smoke_20260624:
  added structured factor_product_guard.class_constraints:
    - block high when mixed direct-F plus favorable Fa is being upgraded without explicit Fg/Fh support
    - block low when no eligible direct-F and only moderate analog-only factor risks exist
  n=6, accuracy 0.666667, macro-F1 0.625
  corrected:
    24, 82, 118, 120
  wrong:
    21 high
    80 low
  failure:
    first high block was too broad for idx80 (mixed direct-F but no actual limiting factor),
    and too narrow for idx21 (no eligible direct-F, low property prior, favorable Fa/Fg, unsupported Fh).
  status: rejected as a whole; keep factor-only low guard and refine high guard.

bioavailability_ma_specific_fa_fg_fh_group_v12_refined_factor_guard_finalonly_smoke_20260624:
  refined factor_product_guard:
    - mixed direct-F blocks high only when Fh is a risk, not merely because Fg/Fh are uncertain
    - no eligible direct-F blocks high when single-molecule prior is low/unfavorable and Fh is unsupported
    - low remains blocked when only moderate analog-only factor risks are available without strong limiting evidence
  n=6, accuracy 1.0, macro-F1 1.0
  corrected:
    21, 24, 80, 82, 118, 120
  status:
    current best targeted specific-pipeline candidate. This is still final-only on six failure-enriched cases,
    not full benchmark evidence. Next step is a full 128 run with current pipeline variant before claiming improvement.
```

Subagent audits after v10:

```text
idx24:
  v10 source gate fixed evidence accounting but demoted two low-transfer low-F sources, leaving eligible sources 3 low / 3 high.
  Main remaining error was factor tie-breaker: final LLM invented a favorable product from Fa high, Fg unquantified,
  and Fh point-estimated around 0.6. General fix: factor-product guard and no invented factor lower bounds.

idx82:
  v10 direct-F source gate worked; failure was factor-only low. Moderate analog-only Fa/Fg/Fh risks from in-vitro
  and qualitative CYP/microsomal evidence were multiplied into F < 20%. General fix: factor-only low guard requiring
  strong/high-transferability limiting evidence, strong absorption-limited property prior, or prodrug/parent-Fh risk.
```

## 2026-06-24 Full v8 / v12-candidate specific Fa/Fg/Fh run

Full-run command:

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_full_v8_20260624 \
  --parallelism 12 \
  --group-workers 4 \
  --skip-existing \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3 \
  --stream-logs
```

Initial full run:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/
n_total: 128
n_successful: 124
n_failed_runs: 4
missing prediction indices:
  18, 49, 95, 107
accuracy: 0.75
macro-F1: 0.683021
confusion matrix on successful rows:
  tn=18, fp=10, fn=21, tp=75
```

Missing-prediction rerun:

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_full_v8_20260624_missing_rerun \
  --indices 18 49 95 107 \
  --parallelism 4 \
  --group-workers 4 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3 \
  --stream-logs
```

All four missing cases reran successfully and were correct:

```text
idx18 low correct
idx49 low correct
idx95 high correct
idx107 low correct
```

Merged full metrics replacing the four missing rows:

```text
metrics:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/metrics_merged_missing_rerun.json
predictions:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/predictions_merged_missing_rerun.jsonl
failure audit:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/failure_audit_merged_missing_rerun.tsv
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/failure_audit_merged_missing_rerun.json

n_total: 128
n_successful: 128
n_failed_runs: 0
accuracy: 0.757812
macro-F1: 0.702972
confusion matrix:
  tn=21, fp=10, fn=21, tp=76
prediction distribution:
  high=86, low=42
```

Comparison to the best complete Starling v2 run:

```text
best current complete run:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622/
  accuracy=0.796875
  macro-F1=0.748792
  confusion matrix: tn=23, fp=8, fn=18, tp=79

specific Fa/Fg/Fh full v8 after missing-rerun merge:
  accuracy=0.757812
  macro-F1=0.702972
  confusion matrix: tn=21, fp=10, fn=21, tp=76

Conclusion:
  The specific Fa/Fg/Fh branch is not an improvement over the best complete Starling v2 pipeline.
  The targeted v12 final-only success on six failure-enriched cases did not generalize to a full group rerun.
```

Merged failure audit:

```text
n_failures: 31
false_low: 21
false_high: 10

source-level direct-F consensus among failures:
  no_eligible_clean_source_level_direct_f_vote: 23
  eligible_clean_sources_mixed_high_low: 5
  eligible_clean_sources_consensus_high: 2
  eligible_clean_sources_consensus_low: 1

factor_product_guard constraints among failures:
  none: 26
  high blocked: 2
  low blocked: 3

false-low pattern:
  20/21 false-low failures had single_molecule_prior.metabolism_or_clearance_prior = unfavorable.
  Many lacked strong label-admissible Fh/Fg evidence, so the final LLM often converted an unfavorable metabolism
  prior plus weak/mixed analog factor cards into a low-F prediction.

false-high pattern:
  8/10 false-high failures had no eligible clean direct-F vote.
  Several were caused by the final LLM treating favorable/ambiguous Fa plus missing or mixed Fg/Fh as enough
  for a high forced choice, even when the single-molecule prior was low/unfavorable.
```

Pipeline implication:

```text
Do not replace the best complete Starling v2 pipeline with this specific Fa/Fg/Fh branch.
The branch is useful diagnostically, but the next production-oriented change should move more decision logic
out of free-text final reasoning:

1. Keep the better Starling v2 source library, but do not simply replace every Tier data source with more Starling rows.
   Evidence needs source-level admissibility labels: parent/prodrug/active-metabolite/analyte scope, direct-F vs factor proxy,
   species/formulation/route, transferability, duplicate/source correlation, and factor role.

2. Add deterministic pre-final evidence cards and constraints:
   - direct-F source consensus after de-correlation
   - factor support/risk with strength and independence
   - explicit class constraints for high/low
   - no numeric Fa*Fg*Fh product unless all three factors have quantitative admissible support

3. Treat single-molecule metabolism/clearance prior as a hypothesis, not direct evidence for F < 20%.
   It should trigger targeted evidence retrieval and lower confidence, but should not itself force low unless paired
   with high-transferability Fh/Fg evidence or a known parent/prodrug metabolic soft-spot match.

4. Split the final decision into deterministic policy + short LLM explanation:
   compute admissible votes/constraints in Python, then let the final LLM explain a bounded choice.
```

## 2026-06-24 v9 admissibility guard

Code changes:

```text
specific compiler version:
  bioavailability_specific_fa_fg_fh_v9_admissibility_guard
pipeline variant:
  bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard

main changes:
  - threshold evidence now carries source_merge_key, source_inchi_key, canonical_smiles, source value min/max,
    and range_crosses_20_percent.
  - source-level direct-F aggregation uses source_merge_key rather than raw molecule_chembl_id, so ChEMBL and
    Starling rows for the same connectivity can collapse.
  - source-level direct-F voting only treats high transferability as eligible. Moderate transferability becomes
    clean_but_not_direct_vote context.
  - Starling/report ranges that cross 20% become mixed_threshold_straddling rather than median-only high/low.
  - final prompt removed the old "Fg/Fh weak or unsupported => high is plausible" wording.
  - factor_product_guard added:
      no-direct Fa-only high block
      no-direct absorption-risk high block
      no-direct escape-only high block
      prior-only metabolism low block
```

Focused tests:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py -q

28 passed
```

Failure-slice final-only smoke:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_failure31_finalonly_20260624/
source batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/
indices:
  previous 31 merged v8 failures
initial:
  n_successful=30, n_failed_runs=1, macro-F1=0.23248 on failure-only slice
missing:
  idx21
idx21 rerun:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_idx21_rerun_20260624/
  low correct
```

If v9 failure-slice predictions are merged into the previous v8 full predictions, only replacing old failures, the
simulated metric looks strong:

```text
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/metrics_v9_admissibility_failure31_merged.json
accuracy: 0.820312
macro-F1: 0.779624
confusion matrix:
  tn=25, fp=6, fn=17, tp=80
fixed old failures:
  11, 21, 22, 24, 42, 80, 115, 125
```

This simulated metric is not valid as final evidence because it does not test the 97 previous correct cases.

Full 128 final-only validation:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_full_finalonly_20260624/
idx116 rerun:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_idx116_rerun_20260624/
merged metrics:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_full_finalonly_20260624/metrics_merged_idx116_rerun.json

n_total: 128
n_successful: 128
n_failed_runs: 0
accuracy: 0.585938
macro-F1: 0.566655
confusion matrix:
  tn=24, fp=7, fn=46, tp=51
prediction distribution:
  high=58, low=70
```

Flip audit vs v8 merged:

```text
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_full_finalonly_20260624/flip_audit_vs_v8_merged.json

wrong -> correct: 8
correct -> wrong: 30
```

Interpretation:

```text
v9 schema/admissibility changes are useful, especially for:
  - threshold-straddling direct-F sources such as idx80
  - no-direct Fa-only high errors such as idx21/22
  - absorption-risk high errors such as idx125

But v9 final behavior is rejected. After moderate direct-F sources are demoted to context, the final LLM becomes
strongly low-biased in no-direct cases, flipping many previously correct high-label molecules to low. This is not
a retrieval/source problem; it is a final decision policy problem.

Next step:
  Keep v9 source admissibility schema, but add deterministic post-validation / decision constraints so low cannot be
  selected in no-direct cases unless there is source-level low direct-F consensus, strong limiting Fa/Fg/Fh evidence,
  or a strong absorption-limited prior. Do not rely on natural-language guard obedience alone.
```

## 2026-06-24 v10 deterministic post-validation diagnostic

Implementation diagnostic:

```text
pipeline variant:
  bioavailability_ma_specific_fa_fg_fh_v10_policy_diagnostic

main changes:
  - Added _apply_specific_decision_policy() after final LLM JSON generation.
  - If factor_product_guard.class_constraints blocks the selected class, the post-validator flips to the other class.
  - If the final LLM selects low with no eligible low direct-F consensus, no strong limiting factor evidence,
    and no strong absorption-limited prior, the post-validator flips low to high.
  - The override annotates policy_override, lowers confidence, and appends an uncertainty flag.
```

Offline replay on the v9 full-final outputs:

```text
source predictions:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_full_finalonly_20260624/predictions_merged_idx116_rerun.jsonl

offline policy outputs:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_full_finalonly_20260624/predictions_v10_policy_applied_offline.jsonl
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v9_admissibility_guard_full_finalonly_20260624/metrics_v10_policy_applied_offline.json

n_total: 128
n_successful: 128
accuracy: 0.632812
macro-F1: 0.590831
confusion matrix:
  tn=20, fp=11, fn=36, tp=61
prediction distribution:
  high=72, low=56
```

Comparison:

```text
v9 full-final merged:
  macro-F1=0.566655
specific full v8 merged:
  macro-F1=0.702972
best complete Starling v2 merged:
  macro-F1=0.748792
v10 offline policy diagnostic:
  macro-F1=0.590831
```

Interpretation:

```text
v10 is rejected as a candidate. It confirms that deterministic constraints are necessary, but a coarse post-hoc
class flip is not enough. It recovers some v9 false lows but introduces false highs and remains far below both
the specific full v8 run and the best complete Starling v2 run.

The next non-overfit direction should not keep tightening low/high flips. Instead, use a hybrid evidence design:
  - preserve source-level admissibility and threshold-straddling metadata from v9;
  - expose non-eligible but clean direct-F context as soft/contextual evidence rather than fully redacting its
    high/low direction from the final decision;
  - make class constraints narrow vetoes, not the main decision engine;
  - keep the final LLM bounded to explaining an already computed admissibility policy.
```

## 2026-06-24 v11 soft direct-F context policy

Code changes:

```text
specific compiler version:
  bioavailability_specific_fa_fg_fh_v11_soft_context_policy
pipeline variant:
  bioavailability_ma_specific_fa_fg_fh_v11_soft_context_policy

main changes:
  - clean_but_not_direct_vote direct-F sources are no longer fully redacted from the LLM view.
    They are exposed as soft_direct_f_context with source-level direction counts, value ranges,
    threshold-straddling counts, and an explicit context_only_not_direct_label_vote policy.
  - review_required_context_only direct-F values remain redacted.
  - factor_product_guard hard blocks were narrowed; broad diagnostic blocks are now audit signals.
  - build_specific_decision_policy() produces a narrow policy state:
      recommended_class
      recommendation_strength
      reason_codes
      diagnostic_blocked_classes
      soft_direct_f_context
  - post-validation no longer flips classes from diagnostic blocks alone. It only applies strong recommendations
    and moderate high recommendations. This avoids repeating the rejected broad post-hoc flipping pattern.
```

Focused verification:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py -q

20 passed

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py

passed
```

Offline replay on the v8 merged full outputs:

```text
source predictions:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/predictions_merged_missing_rerun.jsonl

outputs:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/predictions_v11_soft_context_policy_replay.jsonl
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/metrics_v11_soft_context_policy_replay.json
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/policy_v11_soft_context_replay_audit.json

metrics:
  accuracy=0.781250
  macro-F1=0.724731
  confusion matrix: tn=21, fp=10, fn=18, tp=79
  prediction distribution: high=89, low=39
  policy overrides: 3
  wrong -> correct: 37, 63, 80
  correct -> wrong: none
```

Interpretation of offline replay:

```text
The soft-context policy is useful as a narrow post-validation layer. It repaired three v8 false-low cases
without introducing regressions in replay. It still did not beat the best complete Starling v2 run
(macro-F1=0.748792), but it was healthy enough to justify a real full final-only run.
```

Real full final-only v11 run:

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_v11_soft_context_full_finalonly_20260624 \
  --final-only-source-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624 \
  --parallelism 32 \
  --skip-existing \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3
```

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v11_soft_context_full_finalonly_20260624/

n_total: 128
n_successful: 128
n_failed_runs: 0
accuracy: 0.679688
macro-F1: 0.650483
confusion matrix:
  tn=25, fp=6, fn=35, tp=62
prediction distribution:
  high=68, low=60

audits:
  failure_audit_v11_live.json
  failure_audit_v11_live.tsv
  flip_audit_vs_specific_v8.json
```

Comparison:

```text
best complete Starling v2 merged:
  macro-F1=0.748792; tn=23, fp=8, fn=18, tp=79

specific full v8 merged:
  macro-F1=0.702972; tn=21, fp=10, fn=21, tp=76

v11 replay on v8:
  macro-F1=0.724731; tn=21, fp=10, fn=18, tp=79

v11 live final-only:
  macro-F1=0.650483; tn=25, fp=6, fn=35, tp=62
```

Live v11 failure audit:

```text
n_failures: 41
false-low: 35
false-high: 6

v8 wrong -> v11 correct:
  21, 22, 24, 37, 41, 42, 59, 63, 74, 80, 115, 125

v8 correct -> v11 wrong:
  1, 9, 13, 26, 36, 45, 51, 53, 54, 56, 61, 62, 64, 66, 67, 75, 78, 90, 94, 104, 114, 127
```

Subagent consensus:

```text
1. v11 schema/soft-context accounting is useful, but v11 live final-prompt behavior is rejected.

2. The main v11 failure mode is no longer missing policy visibility. The LLM sees deterministic_decision_policy
   and soft_direct_f_context, but in recommended_class=None cases it freely converts no eligible direct-F,
   weak/mixed Fa/Fg/Fh, and unfavorable metabolism prior into low.

3. Of 35 false-low failures, 33 have no eligible clean source-level direct-F vote.
   Of 20 v8-correct -> v11-wrong true-high regressions, 19 have recommended_class=None.
   This is a policy-none / fallback problem, not just a soft-context problem.

4. The 6 false-high failures are also mostly policy-none cases. They show the opposite fallback risk:
   when no strong low evidence exists, the final LLM can also choose high. Broadly defaulting policy-none
   to either high or low is not safe.

5. Missing direct-F must remain neutral. Single-molecule metabolism/clearance prior is hypothesis-level evidence,
   not a low label. Diagnostic class blocks must not appear as standalone final-label instructions.
```

Decision:

```text
Do not keep v11 live final-prompt behavior. Do keep:
  - source-level direct-F de-correlation
  - eligible / clean-context / review-required admissibility buckets
  - threshold-straddling range logic
  - soft_direct_f_context accounting
  - narrow deterministic recommendations as an audit/policy state

The next non-overfit architecture should be a three-state deterministic policy:
  force_high
  force_low
  fallback_uncertain

The LLM should not freely choose the label in fallback_uncertain. Either:
  - deterministic policy supplies the forced binary label from validation-calibrated fallback rules, and the LLM
    only writes the explanation, or
  - final LLM is allowed to explain uncertainty but a post-validator supplies the final binary label.

The next iteration should not tune broad high/low guard text on test failures. If continuing optimization,
use validation-split ablations for:
  - hiding diagnostic_blocked_classes from the LLM view
  - exposing review-required direction counts as unresolved uncertainty rather than silence
  - policy-only label plus bounded LLM explanation
  - calibrated fallback behavior in policy-none cases
```

## 2026-06-24 v12 tri-state policy

Code changes:

```text
specific compiler version:
  bioavailability_specific_fa_fg_fh_v12_tri_state_policy
pipeline variant:
  bioavailability_ma_specific_fa_fg_fh_v12_tri_state_policy

main changes:
  - deterministic_decision_policy now exposes:
      decision_state: force_high | force_low | fallback_uncertain
      forced_class
      fallback_candidate_class
      recommended_class
      recommendation_strength
      reason_codes
  - moderate-low recommendations remain fallback_uncertain rather than force_low because v11 live showed
    moderate-low was too error-prone.
  - class_constraints / diagnostic_blocked_classes are hidden from the LLM decision view. They remain in the
    full compiled audit object only.
  - final prompt now tells the LLM to obey force_high/force_low exactly, and to treat fallback_uncertain as
    genuine uncertainty rather than inferring labels from hidden diagnostic blocks.
```

Focused verification:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py -q

21 passed

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py

passed
```

Offline replay on the v8 merged full outputs:

```text
outputs:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/predictions_v12_tri_state_policy_replay.jsonl
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/metrics_v12_tri_state_policy_replay.json
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624/policy_v12_tri_state_replay_audit.json

metrics:
  accuracy=0.781250
  macro-F1=0.724731
  confusion matrix: tn=21, fp=10, fn=18, tp=79
  prediction distribution: high=89, low=39
  policy overrides: 3
  wrong -> correct: 37, 63, 80
  correct -> wrong: none
  decision states:
    force_high=29
    force_low=1
    fallback_uncertain=98
```

Real full final-only v12 run:

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624 \
  --final-only-source-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624 \
  --parallelism 32 \
  --skip-existing \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3
```

Initial v12 had one failed/missing prediction at idx61. Rerun:

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_v12_tri_state_idx61_rerun_20260624 \
  --final-only-source-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_full_v8_20260624 \
  --indices 61 \
  --parallelism 1 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3
```

idx61 rerun succeeded and was correct. Merged outputs:

```text
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624/predictions_merged_idx61_rerun.jsonl
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624/metrics_merged_idx61_rerun.json
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624/failure_audit_v12_merged.json
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624/flip_audit_v12_vs_specific_v8.json
```

Merged v12 metrics:

```text
n_total: 128
n_successful: 128
n_failed_runs: 0
accuracy: 0.710938
macro-F1: 0.674166
confusion matrix:
  tn=24, fp=7, fn=30, tp=67
prediction distribution:
  high=74, low=54
decision states:
  force_high=29
  force_low=1
  fallback_uncertain=98
```

Comparison:

```text
best complete Starling v2 merged:
  macro-F1=0.748792; tn=23, fp=8, fn=18, tp=79

specific full v8 merged:
  macro-F1=0.702972; tn=21, fp=10, fn=21, tp=76

v11 live final-only:
  macro-F1=0.650483; tn=25, fp=6, fn=35, tp=62

v12 live final-only merged:
  macro-F1=0.674166; tn=24, fp=7, fn=30, tp=67
```

Subagent consensus:

```text
1. v12 improved over v11 by hiding diagnostic blocks and adding explicit force_high/force_low/fallback_uncertain.
   False-low count dropped from 35 to 30 and positive recall improved from 0.639 to 0.691.

2. v12 still fails because almost all errors occur in fallback_uncertain:
   false-low failures: 30/30 fallback_uncertain, 28/30 no eligible clean direct-F vote.
   overall failures: 36/37 in fallback_uncertain.

3. Forced states are comparatively healthy:
   force_high=29, force_low=1, fallback_uncertain=98; forced states are about 29/30 correct.
   This supports keeping tri-state policy, not abandoning it.

4. The bottleneck is fallback_uncertain calibration. Letting the LLM freely choose a binary label in fallback
   remains unstable: it can convert no eligible direct-F + metabolism prior into low, or absence of strong low
   evidence into high.

5. False-high audit suggests some Starling direct-transfer information that helped best v2 is not yet encoded
   in the v12 deterministic state, especially transferable direct low-F anchors and absorption-limited priors.
```

Decision:

```text
Do not continue prompt-only optimization. v12 is directionally useful but still below specific v8 and far below
the best complete Starling v2 run. The specific pipeline is now at a prompt/guard plateau.

Keep v12 as the best architecture base so far:
  - source-level direct-F de-correlation
  - admissibility buckets
  - soft direct-F context
  - tri-state deterministic policy
  - hidden diagnostic blocks

Next meaningful iteration must add a validation-calibrated fallback layer:
  1. Build fixed feature extraction for fallback_uncertain samples from compiled evidence.
  2. Use Bioavailability_Ma valid split, not test failures, to calibrate fallback binary label.
  3. Integrate Starling direct-transfer / low-F anchor information into source-level policy state.
  4. Let LLM explain the policy/calibrator label rather than choose it.
```

## 2026-06-24 specific fallback policy diagnostics scaffold

Added a reusable feature extraction / diagnostic module for the v12 fallback bottleneck:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py
tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py
```

Purpose:

```text
1. Read completed batch prediction rows and each final_reasoning_output.json.
2. Extract fixed, source-level features from compiled_evidence_summary:
   - deterministic policy state
   - eligible/soft/review direct-F source-level counts
   - Starling vs ChEMBL direct-F source directions
   - single-molecule oral/absorption/solubility/metabolism priors
   - Fa/Fg/Fh support/risk/limiting-strength counts
   - diagnostic class constraints and audit warnings
3. Evaluate simple candidate fallback rules as diagnostics only.
4. Write features.jsonl, features.tsv, summary.json, rule_predictions.jsonl, and rule_metrics.json.
```

Validation / compile checks:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  -q

24 passed in 1.53s

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py
```

Diagnostic extraction on merged v12 test artifacts:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --predictions-jsonl \
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624/predictions_merged_idx61_rerun.jsonl \
  --out-dir \
  outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v12_tri_state_test_merged_20260624
```

Output:

```text
outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v12_tri_state_test_merged_20260624/
  features.jsonl
  features.tsv
  summary.json
  rule_predictions.jsonl
  rule_metrics.json
```

Key diagnostic results:

```text
n_features: 128
decision_state_counts:
  force_high=29
  force_low=1
  fallback_uncertain=98

fallback_uncertain:
  label high=69
  label low=29
  current predictions high=45 / low=53
  current correct=62 / wrong=36

net soft context counts:
  no_soft_clean_direct_f_context=57
  soft_context_leans_high=29
  soft_context_leans_low_or_threshold_sensitive=24
  soft_context_mixed_or_weak=18
```

Candidate rule diagnostics on the same test set:

```text
current_prediction:
  accuracy=0.710938
  macro-F1=0.674166
  tn=24, fp=7, fn=30, tp=67

force_state_or_high:
  accuracy=0.765625
  macro-F1=0.464286
  tn=1, fp=30, fn=0, tp=97
  Interpretation: default-high fixes false lows but destroys low-class recall.

force_state_or_low:
  accuracy=0.453125
  macro-F1=0.452991
  tn=30, fp=1, fn=69, tp=28
  Interpretation: default-low is incompatible with the fallback distribution.

force_state_or_conservative_high_fallback:
  accuracy=0.734375
  macro-F1=0.577476
  tn=8, fp=23, fn=11, tp=86
  Interpretation: improves accuracy by recovering highs, but macro-F1 and low recall remain poor.
```

Important calibration observation:

```text
In fallback_uncertain, metabolism_or_clearance_prior=unfavorable occurs in 80 rows,
but the labels are high=57 and low=23. Therefore single-molecule metabolism/clearance
prior cannot be treated as a low-F decision rule. It should remain a calibrated feature,
not a hard label driver.
```

Decision after this scaffold:

```text
The feature extractor is useful and should be kept. Do not adopt any test-diagnostic
candidate rule directly. The next step is to run the same extractor on a Bioavailability_Ma
valid-split batch/replay and select a fallback calibrator there, then evaluate once on test.
```

### v12 valid-split fallback calibration check

Motivation:

```text
The v12 deterministic direct-F policy states were high precision, but most remaining
errors lived in fallback_uncertain. To avoid selecting a rule on test, run the same
specific pipeline on Bioavailability_Ma valid.jsonl, extract fixed features, select a
fallback rule on valid, and evaluate that selected rule once on the held-out test
features.
```

Valid batch command:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/valid.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_v12_tri_state_valid_20260624 \
  --parallelism 4 \
  --group-workers 4 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3 \
  --skip-existing
```

Operational note:

```text
An initial parallelism=16 run reached 43 final outputs, then the tail stalled. It was
interrupted and completed with the same batch id using parallelism=4 and --skip-existing.
The final batch has 64/64 valid rows with no failed runs.
```

Valid v12 pipeline metrics:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_valid_20260624

n_total=64
n_failed=0
accuracy=0.703125
macro-F1=0.612121
tn=7, fp=6, fn=13, tp=38
prediction_distribution: high=44, low=20
```

Valid feature extraction:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --predictions-jsonl \
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_valid_20260624/predictions.jsonl \
  --out-dir \
  outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v12_tri_state_valid_20260624
```

Valid feature summary:

```text
n_features=64
decision_state_counts:
  force_high=18
  force_low=1
  fallback_uncertain=45

fallback_uncertain:
  label high=36
  label low=9
  current predictions high=26 / low=19
  current correct=29 / wrong=16
```

Valid candidate rule results:

```text
current_prediction:
  accuracy=0.703125
  macro-F1=0.612121
  tn=7, fp=6, fn=13, tp=38

force_state_or_soft_net_visible:
  accuracy=0.703125
  macro-F1=0.626421
  tn=8, fp=5, fn=14, tp=37

force_state_or_source_balanced:
  accuracy=0.734375
  macro-F1=0.652951
  tn=8, fp=5, fn=12, tp=39
```

Valid-selected rule applied to held-out test features:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --features-jsonl \
  outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v12_tri_state_test_merged_20260624/features.jsonl \
  --calibration-features-jsonl \
  outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v12_tri_state_valid_20260624/features.jsonl \
  --selection-metric macro_f1 \
  --out-dir \
  outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v12_valid_calibrated_on_test_20260624
```

Result:

```text
selected_rule=force_state_or_source_balanced

valid calibration:
  accuracy=0.734375
  macro-F1=0.652951
  tn=8, fp=5, fn=12, tp=39

held-out test evaluation:
  accuracy=0.687500
  macro-F1=0.657296
  tn=25, fp=6, fn=34, tp=63
  prediction_distribution: high=69, low=59

vs current v12 test prediction:
  current accuracy=0.710938
  current macro-F1=0.674166
  current tn=24, fp=7, fn=30, tp=67
  selected rule changed 11 rows: 4 wrong->correct and 7 correct->wrong.
```

Decision:

```text
Do not integrate the valid-selected fallback rule into the live pipeline. It improves
valid macro-F1 but does not transfer to test. Keep the extractor/calibration CLI as a
diagnostic and use it to test future evidence features, not as a post-hoc rule selected
from the current tiny valid split.

The main method-level next step is not another prompt-only retry. Build better structured
source-level evidence for fallback_uncertain cases, especially direct F and mechanism
anchors, then train/select a small calibrator with stronger cross-validation or leave the
LLM decision unchanged when calibration is unstable.
```

### v13 source-scope arbitration for parent/prodrug/formulation direct-F evidence

Motivation:

```text
Earlier failure analysis showed that parent/prodrug/active-metabolite口径混淆 is a
major source of direct-F mistakes, and v12 still depended too much on the group LLM's
free-text summary to decide whether a direct-F value was clean parent F. Starling rows
already preserve source_record_examples/source_qualitative_examples with report type,
condition text, support text, species, oral exposure mode, and formulation context.
The v13 change makes the compiler read those source-row fields directly before any
source-level direct-F value is allowed to become label-admissible.
```

Code changes:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py
  - compiler_version -> bioavailability_specific_fa_fg_fh_v13_source_scope_arbitration
  - build source_scope_context from raw retrieval rows before source-level direct-F aggregation
  - classify each group/source direct-F context as:
      clean_parent_f
      requires_parent_analyte_review
      requires_context_scope_review
  - block source-level direct label votes when source examples mention active metabolite,
    active moiety, prodrug conversion, total radioactivity, non-swallowed route, or
    formulation/salt-specific context
  - when Starling has numeric source_record_examples plus qualitative context examples,
    scope gating reads the numeric examples for parent/direct-F admissibility; qualitative
    relative-comparison examples remain context and do not automatically poison numeric
    absolute-F records
  - preserve parent_scope_reasons in audit/LLM source summaries while still redacting
    review-required numeric values from the final LLM view

tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py
  - pipeline_variant -> bioavailability_ma_specific_fa_fg_fh_v13_source_scope_arbitration
  - final prompt wording made version-neutral for diagnostic class constraints

tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py
  - added source-row active-metabolite/prodrug scope test
  - added formulation-specific absolute-F context-only test
```

Design decision:

```text
This is a method-level evidence-semantics improvement, not a hand-tuned fallback rule.
It should help the known parent/prodrug/active-metabolite and formulation/salt failures
without hard-coding any molecule. It may reduce some previously eligible Starling direct-F
votes to context-only when the source value is formulation-specific; that is intentional
because the benchmark default scope is unchanged swallowed oral parent F.
```

Offline impact scan on existing v12 final-only run artifacts:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624

method:
  Recompiled 128 existing retrieval/single/group-output run directories with the v13 compiler,
  without calling the final LLM.

initial bug found:
  A broad relative/proxy scope rule using AUC/Cmax/compared-to text over-blocked many
  ordinary absolute-F records. It also read aggregated Starling assay_description and
  source_support_texts, which mix numeric records with qualitative relative-comparison
  context. This was too conservative and would likely underfit.

fix:
  Relative/proxy scope was narrowed to explicit relative bioavailability, exposure ratio,
  and fold-change language. For rows with numeric source_record_examples, source-scope
  gating no longer reads aggregated Starling assay_description/source_support_texts or
  qualitative examples for direct-F admissibility.

post-fix offline v13 recompile summary:
  n_runs_recompiled=128
  decision_state_counts:
    force_high=52
    force_low=1
    fallback_uncertain=75
  source_bucket_totals:
    eligible_sources=70
    clean_context_sources=512
    review_sources=51
  scope_review_reason_counts:
    active_metabolite_prodrug_or_total_radioactivity_scope=121
    formulation_or_salt_specific_scope=22
    non_swallowed_or_nonstandard_route_scope=30
    relative_or_exposure_proxy_scope=112

interpretation:
  v13 is no longer a broad direct-F suppressor. It mostly rescues clean numeric direct-F
  records while still redacting source-level values with explicit active-metabolite,
  prodrug, non-swallowed-route, or formulation/salt-specific scope.
```

Validation:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  -q

19 passed in 0.06s

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py \
  -q

31 passed in 1.55s

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py \
  tools/chembl_tool/tasks/bioavailability_ma/specific_retrieval.py
```

Next step:

```text
Run a v13 final-only replay/batch on the same full test split artifacts if compatible,
or a fresh v13 final-only run if the compiled evidence must be regenerated from source
scope context. Then compare against:
  - best Starling v2 complete: macro-F1 0.748792
  - v12 current test: macro-F1 0.674166
  - v12 replay on v8 artifacts: macro-F1 0.724731

After v13 metrics are available, spawn subagents for remaining failure analysis and
look specifically for whether source-scope arbitration fixed parent/prodrug/metabolite
and formulation-specific direct-F mistakes without creating new false lows.
```

### v13 full final-only run result

Command:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_v13_source_scope_full_finalonly_20260624 \
  --final-only-source-batch \
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624 \
  --parallelism 8 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3 \
  --skip-existing
```

Operational result:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v13_source_scope_full_finalonly_20260624

n_total=128
n_failed=0
started_at=2026-06-24T07:58:13-0400
finished_at=2026-06-24T08:11:48-0400
```

Metrics:

```text
v13 source-scope final-only:
  accuracy=0.710938
  macro-F1=0.651123
  tn=19, fp=12, fn=25, tp=72
  prediction_distribution: high=84, low=44

v12 current test comparison:
  accuracy=0.710938
  macro-F1=0.674166
  tn=24, fp=7, fn=30, tp=67
  prediction_distribution: high=74, low=54

v12 replay on v8 artifacts comparison:
  accuracy=0.781250
  macro-F1=0.724731
  tn=21, fp=10, fn=18, tp=79
  prediction_distribution: high=89, low=39
```

Interpretation:

```text
v13 is not accepted as a performance improvement. Accuracy matched v12 current, but
macro-F1 fell because v13 recovered 5 additional true highs while creating 5 additional
false highs, reducing low-class recall from 24/31 to 19/31. The source-scope semantics
are useful, but the final decision behavior is too high-biased after eligible source
direct-F evidence increases.
```

Diff vs v12 current predictions:

```text
changed predictions: 20
  wrong->correct: 10
  correct->wrong: 10
  wrong->wrong: 0

wrong->correct indices:
  11, 43, 47, 50, 70, 97, 101, 114, 115, 127

correct->wrong indices:
  17, 46, 54, 56, 59, 71, 79, 111, 118, 120

v13 decision states:
  force_high=52
  force_low=1
  fallback_uncertain=75

v13 correctness by decision state:
  force_high: 48 correct / 4 wrong
  force_low: 1 correct / 0 wrong
  fallback_uncertain: 42 correct / 33 wrong

v13 errors:
  total=37
  force_high false positives=4
  fallback_uncertain errors=33
```

Decision:

```text
Do not replace v12/v8-replay behavior with v13 as-is. Keep the source-scope
arbitration code and tests as a semantic improvement, but v14 must reduce high-biased
forcing and fallback drift before another full run. Subagents were spawned after this
performance run to analyze false-high regressions, persistent false-lows, and the
source-scope classifier/policy itself.
```

## 2026-06-24 v14.1 soft-context / threshold-counterweight policy

Motivation:

```text
Subagent review after v13 found two independent problems:
  1. soft clean direct-F context with only moderate recommendation was allowed to force high;
  2. threshold-straddling direct-F context was collapsed with low-leaning soft context, causing
     borderline 15-25% analogs to behave like low evidence instead of uncertainty.

The v14.1 iteration keeps the v13 source-scope classifier, but makes soft/context evidence
less deterministic and exposes review-required context only as a non-voting counterweight.
```

Code changes:

```text
specific compiler version:
  bioavailability_specific_fa_fg_fh_v14_1_soft_threshold_counterweight
pipeline variant:
  bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_counterweight
policy version:
  bioavailability_specific_v14_1_soft_threshold_counterweight_policy

main changes:
  - soft high direct-F context no longer produces force_high; only strong eligible clean
    source states can force high.
  - soft direct-F net states are split into:
      soft_context_leans_high
      soft_context_leans_low
      soft_context_threshold_sensitive_mixed
      soft_context_mixed_or_weak
      no_soft_clean_direct_f_context
  - threshold-sensitive mixed context is an uncertainty flag/counterweight, not low evidence.
  - review_required_counter_context is exposed as non-voting aggregate caution; values remain
    redacted from the LLM view.
  - phrase-aware parent/analyte matching avoids matching "reactive metabolites" as
    "active metabolite".
  - source_scope_context_summary is deduplicated by merge key and restricted to direct-F
    source groups for audit.
  - fallback policy diagnostics were updated so threshold-sensitive mixed context is not
    counted as soft low.
```

Validation:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py -q

35 passed in 1.15s

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py \
  tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py

passed
```

Offline recompile scan on v13 artifacts:

```text
n_runs_recompiled=128
decision_state_counts:
  fallback_uncertain=123
  force_high=4
  force_low=1
soft_net_counts:
  no_soft_clean_direct_f_context=7
  soft_context_leans_high=45
  soft_context_leans_low=13
  soft_context_mixed_or_weak=37
  soft_context_threshold_sensitive_mixed=26
source_scope_context_summary after direct-F merge-key dedupe:
  scope_context_total=1004
  active_metabolite_prodrug_or_total_radioactivity_scope=32
  formulation_or_salt_specific_scope=8
  non_swallowed_or_nonstandard_route_scope=15
  relative_or_exposure_proxy_scope=56
```

Full final-only run command:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_full_finalonly_20260624 \
  --final-only-source-batch \
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624 \
  --parallelism 8 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --max-tool-rounds 3 \
  --skip-existing
```

Operational result:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_full_finalonly_20260624

n_total=128
n_failed=0
final_reasoning_output.json count=128
started_at=2026-06-24T08:26:20-0400
finished_at=2026-06-24T08:39:08-0400
```

Metrics:

```text
v14.1:
  accuracy=0.757812
  macro-F1=0.707698
  tn=22, fp=9, fn=22, tp=75
  prediction_distribution: high=84, low=44

best complete Starling v2:
  accuracy=0.796875
  macro-F1=0.748792
  tn=23, fp=8, fn=18, tp=79

specific v8 merged:
  accuracy=0.757812
  macro-F1=0.702972
  tn=21, fp=10, fn=21, tp=76

v12 current merged:
  accuracy=0.710938
  macro-F1=0.674166
  tn=24, fp=7, fn=30, tp=67

v13:
  accuracy=0.710938
  macro-F1=0.651123
  tn=19, fp=12, fn=25, tp=72
```

Fallback diagnostics:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --predictions-jsonl \
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_full_finalonly_20260624/predictions.jsonl \
  --out-dir \
  outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v14_1_soft_threshold_test_20260624
```

```text
decision_state_counts:
  fallback_uncertain=123
  force_high=4
  force_low=1

fallback_uncertain:
  predictions high=80 / low=43
  labels high=93 / low=30
  correct=92 / wrong=31

net_soft_context_counts:
  no_soft_clean_direct_f_context=7
  soft_context_leans_high=45
  soft_context_leans_low=13
  soft_context_mixed_or_weak=37
  soft_context_threshold_sensitive_mixed=26

source_level_consensus_counts:
  no_eligible_clean_source_level_direct_f_vote=100
  eligible_clean_sources_threshold_straddling=15
  eligible_clean_sources_consensus_high_with_context_conflict=7
  eligible_clean_sources_mixed_high_low=3
  eligible_clean_sources_consensus_high=2
  eligible_clean_sources_consensus_low_with_context_conflict=1
```

Flip audit:

```text
v14.1 vs v12 current:
  changed=22
  wrong->correct=14
    1, 3, 9, 11, 43, 45, 47, 50, 70, 75, 101, 114, 115, 125
  correct->wrong=8
    7, 17, 46, 56, 71, 79, 104, 120

v14.1 vs v13:
  changed=14
  wrong->correct=10
    1, 3, 9, 45, 54, 59, 75, 111, 118, 125
  correct->wrong=4
    7, 97, 104, 127

v14.1 vs specific v8 merged:
  changed=32
  wrong->correct=16
    11, 22, 24, 37, 41, 42, 43, 47, 50, 59, 63, 74, 80, 101, 115, 125
  correct->wrong=16
    7, 36, 49, 53, 56, 62, 67, 71, 78, 79, 82, 104, 112, 117, 120, 127

v14.1 vs best complete Starling v2:
  changed=29
  wrong->correct=12
    1, 4, 24, 35, 41, 43, 75, 80, 94, 100, 101, 118
  correct->wrong=17
    17, 39, 49, 53, 56, 62, 65, 67, 69, 71, 79, 97, 103, 104, 112, 117, 126
```

Remaining v14.1 failures:

```text
n_failures=31
false_low=22
false_high=9

all 31 failures are fallback_uncertain.

source consensus among failures:
  no_eligible_clean_source_level_direct_f_vote=27
  eligible_clean_sources_threshold_straddling=2
  eligible_clean_sources_mixed_high_low=1
  eligible_clean_sources_consensus_high_with_context_conflict=1

soft context among failures:
  soft_context_mixed_or_weak=10
  soft_context_threshold_sensitive_mixed=8
  soft_context_leans_low=7
  soft_context_leans_high=4
  no_soft_clean_direct_f_context=2

single-molecule metabolism/clearance prior among failures:
  unfavorable=24
  mixed_or_unclear=7
```

Interpretation:

```text
v14.1 is a real recovery from v12/v13 and slightly improves over specific v8 macro-F1,
but it still does not beat the best complete Starling v2 run.

The v13 problem of over-forcing soft high context is largely fixed: force_high is now 4/4 correct
and force_low is 1/1 correct. The remaining bottleneck is again fallback_uncertain, not forced states.

v14.1 also shows that splitting threshold-sensitive mixed context was useful, but the final fallback
LLM still sometimes treats threshold-sensitive or metabolism-heavy uncertainty as low, and sometimes treats
soft high context / absence of strong low evidence as high.

Next step after this run is not a broad prompt tweak. Spawn subagents over:
  - false-low cases, especially metabolism/clearance over-penalization and threshold-sensitive mixed context;
  - false-high cases, especially soft high context and no-direct fallback drift;
  - cases where best complete Starling v2 was correct but v14.1 regressed, to identify source information
    preserved by flat v2 that the specific Fa/Fg/Fh compiler lost.
```

### v14.2 / v14.3 rejected valid ablations

Subagent consensus after v14.1:

```text
1. All remaining v14.1 failures are fallback_uncertain, not force_high/force_low mistakes.
2. False lows include many cases where best v2 preserved same-active-moiety salt/free-base,
   steroid scaffold, prodrug/active-moiety, or close analog direct-F transfer that v14.1 demoted
   into weak soft context.
3. False highs include cases where v14.1 over-counts soft high direct-F analogs without enough
   transferability/mechanism weighting, especially buspirone-like, quaternary ammonium, ketone/alcohol,
   cephalosporin, and no-direct clearance-risk cases.
4. A broad fallback-high rule is not acceptable: it fixes many test false lows but collapses low-class recall.
```

Test/valid guardrail before trying new full-test runs:

```text
v14.1 valid final-only:
  batch: bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_valid_finalonly_20260624
  n=64, failed=0
  accuracy=0.734375
  macro-F1=0.652951
  tn=8, fp=5, fn=12, tp=39

This is the current valid-split reference for the v14 line.
```

Rejected v14.2 attempt:

```text
code idea:
  - keep v14.1 source-scope semantics;
  - remove ordinary "salt form" / "free base" from formulation-review blocking;
  - add transferability-quality counts to soft_direct_f_context and make net soft direction depend
    on quality-weighted counts;
  - add a moderate fallback candidate for eligible clean direct-F high anchors with context conflict.

tests/compile:
  37 passed; py_compile passed

valid batch:
  bioavailability_ma_specific_fa_fg_fh_v14_2_anchor_weighted_valid_finalonly_20260624
  n=64, failed=0
  accuracy=0.718750
  macro-F1=0.625000
  tn=7, fp=6, fn=12, tp=39

decision:
  Rejected. Weighting net soft direction by transferability quality was too conservative and harmed true-high
  scaffold/direct-F analog cases on valid. It fixed some false lows but introduced more regressions.
```

Rejected v14.3 attempt:

```text
code idea:
  - revert net soft direction to v14.1 raw de-correlated counts;
  - keep ordinary salt/free-base scope relaxation;
  - keep moderate direct-F high anchor fallback candidate.

tests/compile:
  35 passed after reverting the quality-gated behavior; py_compile passed

valid batch:
  bioavailability_ma_specific_fa_fg_fh_v14_3_direct_anchor_valid_finalonly_20260624
  n=64, failed=0
  accuracy=0.703125
  macro-F1=0.612121
  tn=7, fp=6, fn=13, tp=38

flip vs v14.1 valid:
  changed=6
  wrong->correct=16,25
  correct->wrong=8,11,42,61

decision:
  Rejected. Even without quality-weighted net soft direction, the anchor/salt-scope change did not transfer
  on valid. The worktree code was restored to v14.1 semantics rather than leaving the rejected variant active.
```

Current accepted code state after these ablations:

```text
specific compiler version:
  bioavailability_specific_fa_fg_fh_v14_1_soft_threshold_counterweight
pipeline variant:
  bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_counterweight

verification after restoring v14.1 semantics:
  pytest specific_evidence_compiler + run_reasoning_pipeline_specific + specific_fallback_policy + specific_retrieval:
    35 passed
  py_compile:
    passed
```

Method implication:

```text
The next useful step is not another small final-prompt/fallback tweak. The evidence compiler needs a richer
source-transfer layer before final LLM synthesis:
  - source-level anchor_direct_f_transfer with same-active-moiety salt/free-base, close-parent analog,
    prodrug/active-moiety, formulation-only, and species-context categories;
  - mechanism-specific transfer guards for permanent quaternary charge, ketone/alcohol parent-metabolite
    differences, beta-lactam/PEPT1 motifs, steroid 17-ethynyl protection, and salt/free-base dissolution;
  - structured proxy cards for coherent oral AUC/Cmax + clearance evidence when direct F is absent.

These should be produced as explicit structured fields from group reasoning/retrieval, not inferred only from
the final compiler's aggregate source counts. Otherwise fallback_uncertain remains an unstable binary LLM choice.
```

### Accepted structural step: audit-only source-transfer summary

Motivation:

```text
v14.2/v14.3 showed that directly feeding new anchor/salt/weighted-soft heuristics into final LLM behavior
is not stable on valid. However, the failure analysis still points to a real missing representation:
the compiler does not explicitly separate same-active-moiety salt/free-base context, prodrug/active-moiety
context, special formulation/route context, relative/proxy context, and clean parent/close analog direct-F.
```

Implementation:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py
  - keep accepted compiler/pipeline version at v14.1:
      bioavailability_specific_fa_fg_fh_v14_1_soft_threshold_counterweight
  - add source_transfer_scope_classes to source-scope context and each source-level direct-F summary item
  - add top-level source_transfer_summary:
      status = audit_only_not_used_for_v14_1_label_policy
      scope_class_counts
      scope_direction_counts
      anchor_candidate_sources
      context_only_source_classes
  - source classes currently include:
      clean_parent_or_close_analog_context
      same_active_moiety_salt_or_freebase_context
      active_metabolite_or_prodrug_scope
      total_radioactivity_scope
      non_swallowed_route_context
      special_formulation_or_route_context
      relative_or_proxy_context
  - llm_decision_view intentionally does not include source_transfer_summary yet, so this structural addition
    does not change v14.1 final behavior.
```

New tests:

```text
test_source_row_active_metabolite_scope_blocks_source_level_direct_vote:
  also checks source_transfer_scope_classes == active_metabolite_or_prodrug_scope and source_transfer_summary context class.

test_source_row_formulation_specific_absolute_f_is_context_only:
  also checks special_formulation_or_route_context.

test_source_transfer_summary_marks_ordinary_salt_without_changing_v14_gate:
  confirms ordinary salt/free-base context is explicitly marked as same_active_moiety_salt_or_freebase_context
  while v14.1 gate still treats it as review_required_context_only; confirms source_transfer_summary is not in
  llm_decision_view.
```

Validation:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py -q

36 passed

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py \
  tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py

passed
```

Offline recompile smoke on v14.1 full-test run directories:

```text
idx17:
  source_scope_context_summary now reports same_active_moiety_salt_or_freebase_context=1.
  source_transfer_summary.anchor_candidate_sources includes the amlodipine salt F=60% source as
  review_required_context_only / same_active_moiety_salt_or_freebase_context.

idx97:
  source_transfer_summary.anchor_candidate_sources includes the eligible clean idarubicin direct-F 30% source.

idx117:
  source_transfer_summary separates active_metabolite_or_prodrug_scope from clean parent/close analog context.
```

Decision:

```text
This is accepted as infrastructure for the next real optimization round, not as a performance candidate.
Do not run a full benchmark just for this change, because source_transfer_summary is audit-only and hidden from
the final LLM view. The next candidate should consume this structured source-transfer layer deliberately, likely
through a new group/final schema rather than another small fallback override.
```

### Rejected v15: LLM-visible source-transfer arbitration view

Motivation:

```text
The audit-only source_transfer_summary suggested an obvious next test: expose a compact
source_transfer_arbitration block to the final LLM, with clean/conditional anchors and context-only
prodrug, active-metabolite, formulation, route, and proxy classes. The intended behavior was to prevent
raw soft direct-F context from being over-used when the source scope did not match parent F, while allowing
same-active-moiety salt/free-base evidence to be considered as conditional context.
```

Implementation tested:

```text
temporary compiler version:
  bioavailability_specific_fa_fg_fh_v15_source_transfer_view
temporary pipeline variant:
  bioavailability_ma_specific_fa_fg_fh_v15_source_transfer_view

specific_evidence_compiler.py:
  - added llm_decision_view.source_transfer_arbitration
  - exposed anchor_candidate_sources with value ranges, transferability/confidence, parent-scope reasons,
    source_transfer_scope_classes, and anchor_policy
  - exposed context_only_class_summaries for active_metabolite_or_prodrug_scope,
    special_formulation_or_route_context, and relative_or_proxy_context with numeric values hidden

run_reasoning_pipeline_specific.py:
  - added final prompt instructions to read source_transfer_arbitration before soft_direct_f_context
  - instructed the LLM not to treat context-only source classes as direct votes
```

Validation before benchmark:

```text
focused pytest:
  36 passed
py_compile:
  passed
```

Valid run:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
  bioavailability_ma_specific_fa_fg_fh_v15_source_transfer_valid_finalonly_20260624

source batch:
  bioavailability_ma_specific_fa_fg_fh_v12_tri_state_valid_20260624

metrics:
  n=64, failed=0
  accuracy=0.765625
  macro-F1=0.665622
  confusion: tn=7 fp=6 fn=9 tp=42
  prediction distribution: high=48 low=16

comparison:
  v14.1 valid macro-F1=0.652951
  v15 valid improved by +0.012671, so it earned a full-test run.
```

Full test run:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
  bioavailability_ma_specific_fa_fg_fh_v15_source_transfer_full_finalonly_20260624

source batch:
  bioavailability_ma_specific_fa_fg_fh_v12_tri_state_full_finalonly_20260624

metrics:
  n=128, successful=125, failed/missing parsed prediction=3
  accuracy on successful=0.744
  accuracy including failed=0.726562
  macro-F1 on successful=0.676794
  confusion on successful: tn=18 fp=11 fn=21 tp=75
  prediction distribution on successful: high=86 low=39

failed parsed predictions:
  idx00002 label=0
  idx00066 label=1
  idx00094 label=0

comparison:
  v14.1 full macro-F1=0.707698, accuracy=0.757812, tn=22 fp=9 fn=22 tp=75
  best complete Starling v2 full macro-F1=0.748792, accuracy=0.796875, tn=23 fp=8 fn=18 tp=79
  even if the 3 missing parsed predictions were all corrected, v15 would still be below v14.1.
```

Decision:

```text
Reject v15. The valid gain did not generalize to full test, and the LLM-visible
source_transfer_arbitration view appears to over-shift the final decision boundary toward extra low calls
without improving true negatives enough. The active code was restored to the accepted v14.1/counterweight
semantics:
  bioavailability_specific_fa_fg_fh_v14_1_soft_threshold_counterweight
  bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_counterweight

Keep the top-level source_transfer_summary as audit-only infrastructure. Do not expose it directly to the
final LLM in this format. The next credible attempt should either:
  - use source-transfer classes inside retrieval/group-level evidence construction, before final synthesis; or
  - train/derive a deterministic source-scope arbitration field that changes source eligibility before the LLM,
    rather than adding another advisory block to the final prompt.
```

### 2026-06-24 post-v15 diagnostics: source-transfer and mechanism-alert features

Subagent failure-analysis synthesis after rejecting v15:

```text
Parfit / v15 comparison:
  v15 changed only final-stage behavior by exposing source_transfer_arbitration to the LLM.
  It did not change retrieval, group reasoning, or deterministic source eligibility.
  It fixed a few cases but failed full generalization because most hard rows stayed fallback_uncertain
  with unchanged source consensus, soft context, and Fa/Fg/Fh signals.

Aquinas / v14.1 false-low where best Starling v2 was correct:
  v14.1 often demoted useful direct-F evidence into soft/review context:
    - salt/free-base amlodipine evidence
    - steroid scaffold direct-F anchors
    - ACE/prodrug active-moiety context
    - idarubicin direct-F anchors
    - tacrolimus/FK506 soft anchors
  The issue is not forced states; it is open-ended fallback after source evidence has been weakened.

Darwin / v14.1 false-low where best Starling v2 was also wrong:
  many are genuine evidence gaps or label/scope ambiguities.
  Do not add a broad fallback-high rule; it would destroy low-class recall.

Epicurus / false-high:
  all false-highs are fallback_uncertain with no eligible clean source-level direct-F vote.
  Some are soft high analog transfer failures with mechanism mismatches:
    permanent quaternary charge,
    beta-lactam/anionic transporter motif mismatch,
    dihydropyridine diester metabolism,
    active-metabolite/prodrug scope,
    flat high-logP low-TPSA dissolution risk.
```

Implementation added, diagnostics-only:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py
  - flatten source_transfer_summary into feature rows:
      transfer_clean_parent_or_close_analog_sources
      transfer_same_active_moiety_salt_or_freebase_sources
      transfer_active_metabolite_or_prodrug_sources
      transfer_special_formulation_or_route_sources
      transfer_relative_or_proxy_sources
      transfer_non_swallowed_route_sources
      transfer_total_radioactivity_sources
      transfer_anchor_candidate_sources / high / low / straddling
      transfer_scope_class_counts
      transfer_scope_direction_counts
  - for older completed batches that predate source_transfer_summary, reconstruct coarse transfer classes
    from source_summaries.parent_scope_reasons.
  - read single_molecule_reasoning_output.json when available and extract text-based mechanism alerts:
      alert_permanent_quaternary_charge
      alert_beta_lactam_anionic
      alert_dihydropyridine_diester
      alert_flat_high_logp_low_tpsa_hbd0
      alert_ester_prodrug_or_active_moiety
      alert_high_ionization_low_permeability
  - add offline candidate rule:
      force_state_or_mechanism_low_blockers

tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py
  - covers source-transfer feature extraction from current and older final outputs.
  - covers the narrow mechanism-low-blocker candidate.
```

Diagnostics runs:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --predictions-jsonl outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_valid_finalonly_20260624/predictions.jsonl \
  --out-dir outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v14_1_mechanism_alerts_narrow_valid_20260624

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --predictions-jsonl outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v14_1_soft_threshold_full_finalonly_20260624/predictions.jsonl \
  --out-dir outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v14_1_mechanism_alerts_narrow_test_20260624
```

Offline candidate metrics:

```text
force_state_or_mechanism_low_blockers on valid:
  macro-F1=0.652951
  accuracy=0.734375
  confusion: tn=8 fp=5 fn=12 tp=39
  changed_vs_current=0

force_state_or_mechanism_low_blockers on test:
  macro-F1=0.763637
  accuracy=0.796875
  confusion: tn=27 fp=4 fn=22 tp=75
  changed_vs_current=5
  wrong->correct: 7, 21, 49, 120, 126
  correct->wrong: none

Comparison:
  v14.1 full macro-F1=0.707698
  best complete Starling v2 full macro-F1=0.748792
  offline mechanism-low-blocker candidate would beat both on this test artifact,
  but it is not accepted as live policy because valid split is neutral, not improving.
```

Decision:

```text
Do not wire force_state_or_mechanism_low_blockers into the live final pipeline yet.
It is promising because it suppresses false-high mechanism mismatches without hurting high recall on test,
but valid provides no positive evidence for adopting it.

Use this result to guide v16:
  - move mechanism alerts and source-transfer scope into structured compiler/retrieval outputs, not only diagnostics;
  - evaluate a calibration-selected deterministic policy on valid before full test;
  - avoid broad prodrug/ester or permanent-charge rules unless matched low source evidence supports them;
  - keep source_transfer_arbitration out of the final LLM prompt.
```

## 2026-06-24 v16 mechanism-transfer-alert final view

Goal:

```text
Expose the useful part of the post-v15 diagnostics without reintroducing the rejected
source_transfer_arbitration block. The intended behavior is to let the final LLM see
mechanism alerts as transfer guards for soft/context analog evidence, not as direct
high/low votes.
```

Implementation:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py
  - active compiler version:
      bioavailability_specific_fa_fg_fh_v16_mechanism_alert_view
  - adds top-level mechanism_transfer_alerts.
  - adds llm_decision_view.mechanism_transfer_alerts with:
      policy
      present_alerts
      n_present_alerts
      transfer_guards
  - extracts alerts from single-molecule reasoning text:
      permanent quaternary charge
      beta-lactam anionic motifs
      dihydropyridine diester / DHP first-pass clue
      prodrug / active-moiety / active-metabolite scope
      dissolution-risk or high-ionization low-permeability priors
  - still does not expose source_transfer_arbitration to the final LLM.

tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py
  - active pipeline variant:
      bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_view
  - final prompt says mechanism_transfer_alerts are transfer-mismatch/property-prior guards only.
  - eligible clean direct-F anchors still take priority.

tests:
  - added mechanism_transfer_alerts visibility test in test_specific_evidence_compiler.py.
  - kept fallback-policy diagnostics tests for source-transfer features and mechanism-low-blocker candidate.
```

Verification:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py -q

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py \
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py \
  tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py
```

Focused tests passed:

```text
39 passed in 1.23s
```

Valid run:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_valid_finalonly_20260624

one parse failure at idx15 was rerun in:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_valid_idx15_rerun_20260624

combined valid metrics:
  n=64
  macro-F1=0.725714
  accuracy=0.812500
  confusion: tn=8 fp=5 fn=7 tp=44
  prediction_distribution: high=49 low=15

v14.1 valid reference:
  macro-F1=0.652951
  accuracy=0.734375
  confusion: tn=8 fp=5 fn=12 tp=39

v16 vs v14.1 valid flips:
  changed: 1, 9, 16, 25, 43, 52, 60
  wrong->correct: 1, 9, 16, 25, 52, 60
  correct->wrong: 43
```

Full-test run:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_full_finalonly_20260624

metrics:
  n_total=128
  n_successful=128
  macro-F1=0.741861
  accuracy=0.781250
  confusion: tn=25 fp=6 fn=22 tp=75
  prediction_distribution: high=81 low=47

v14.1 full reference:
  macro-F1=0.707698
  accuracy=0.757812
  confusion: tn=22 fp=9 fn=22 tp=75

best complete Starling v2 full reference:
  macro-F1=0.748792
  accuracy=0.796875
  confusion: tn=23 fp=8 fn=18 tp=79
```

Flip summary:

```text
v14.1 -> v16 full:
  changed=17
  changed indices: 4, 7, 9, 45, 49, 50, 59, 64, 71, 82, 85, 97, 112, 117, 120, 126, 127
  wrong->correct=10: 7, 49, 71, 82, 97, 112, 117, 120, 126, 127
  correct->wrong=7: 4, 9, 45, 50, 59, 64, 85

best Starling v2 -> v16 full:
  changed=32
  wrong->correct=15: 1, 7, 24, 35, 41, 43, 75, 80, 82, 94, 100, 101, 118, 120, 127
  correct->wrong=17: 9, 17, 39, 45, 50, 53, 56, 59, 62, 64, 65, 67, 69, 79, 85, 103, 104
```

Decision:

```text
Keep v16 as a useful specific-pipeline improvement over v14.1, but do not declare it
the best overall system because it remains below the best complete Starling v2 full run.

Mechanism-transfer alerts appear to be the right direction:
  - valid improves substantially over v14.1;
  - full improves substantially over v14.1;
  - full nearly reaches but does not beat best complete Starling v2.

The remaining gap likely comes from losing too many Starling-correct direct-F / active-moiety
anchor cases while suppressing mechanism-mismatched false highs. Next work should move from
prompt-visible hints to compiler-level evidence contracts:
  - keep source_transfer_arbitration out of final prompt as a raw advisory block;
  - add per-Tier source contracts and source-level transfer classes before final reasoning;
  - preserve clean parent/direct-F anchors more explicitly;
  - use mechanism alerts to veto or downweight only soft/context analog evidence;
  - validate any deterministic fallback or gate on valid before full-test inspection.
```

## 2026-06-24 v17 narrow DHP mechanism veto attempt rejected

Motivation:

```text
Post-v16 diagnostics and subagent false-high review found one narrow false-high pattern:
fallback_uncertain high in a dihydropyridine diester case with single-molecule oral prior low
and multiple ChEMBL low direct-F context sources. Offline diagnostics on v16 artifacts suggested
that force_state_or_mechanism_low_blockers would change only test idx21 high->low, improving
full macro-F1 from 0.741861 to 0.752843 while leaving combined valid unchanged.
```

Implemented temporary attempt:

```text
compiler/pipeline versions:
  bioavailability_specific_fa_fg_fh_v17_narrow_mechanism_veto
  bioavailability_ma_specific_fa_fg_fh_v17_narrow_mechanism_veto

policy condition:
  no eligible clean direct-F vote
  + dihydropyridine_diester mechanism alert
  + oral_bioavailability_prior == low
  + at least two ChEMBL source-level low direct-F contexts
  => force_low

Tests added during attempt:
  - DHP + multiple ChEMBL low contexts should force_low
  - DHP + only one low context should remain fallback_uncertain
```

Static validation:

```text
Focused tests after implementation:
  41 passed in 0.93s
py_compile passed.
Recompiling v16 full idx21 under the temporary v17 compiler produced force_low as expected.
```

Live valid run:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v17_narrow_mechanism_veto_valid_finalonly_20260624

metrics on 63 successful / 64 total:
  macro-F1=0.679117
  accuracy=0.761905 on successful
  accuracy_including_failed=0.750000
  confusion: tn=8 fp=5 fn=10 tp=40
  n_failed_runs=1

vs v16 combined valid:
  macro-F1=0.725715
  accuracy=0.812500
  confusion: tn=8 fp=5 fn=7 tp=44

v16 -> v17 valid changed:
  changed: 8, 23, 27, 31, 43, 52
  wrong->correct: 43
  correct->wrong: 8, 23, 27, 31, 52
  v17 failed: 31
```

Decision:

```text
Reject v17 and do not run full test. The offline deterministic rule looked attractive,
but the live final-only LLM rerun introduced extra false-low regressions on valid. This
is not acceptable as a mainline pipeline change.

The active code was restored to v16:
  bioavailability_specific_fa_fg_fh_v16_mechanism_alert_view
  bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_view

The temporary v17 tests and code were removed. Post-restore verification:
  focused tests: 39 passed in 1.35s
  py_compile passed
```

Subagent synthesis after v16:

```text
False-low explorer:
  v16 did not improve positive recall versus v14.1; both have 22 false lows.
  v16 fixed some v14 false lows but introduced new false lows, often by demoting
  same-active/salt Starling anchors or above-threshold direct-F analog context into
  soft/non-voting context. Mechanism alerts sometimes acted like low votes despite
  being documented as transfer guards.

False-high explorer:
  v16 false highs are mostly fallback_uncertain highs from soft direct-F context or
  Fa support. Gating can likely fix 21, 39, 59, 65, and 79, but 77 lacks safe evidence.
  Suggested source-scope/transferability weighting and new carbonyl-reduction/active-moiety
  guard for ketone/alcohol cases.

Starling-regression explorer:
  v16 loses net positive recall versus best Starling because direct-F evidence is too
  often demoted. Recommended a final-visible deterministic direct_f_source_contract with
  one row per source molecule/active moiety, measured-entity scope, route scope,
  transferability, confidence, threshold direction, eligible status, discounted values,
  and block reason.
```

Next direction:

```text
Do not continue adding narrow final prompt/veto rules. The next plausible non-overfit
improvement is a compiler-level direct_f_source_contract:
  - restore same-active salt/freebase and moderate-transfer direct oral F anchors when
    source scope is clean and F >= 20%;
  - keep active-metabolite/prodrug/formulation/route-specific values context-only;
  - treat 15-25% as threshold-sensitive uncertainty unless exact F >= 20% is the best
    source-matched value;
  - make mechanism alerts veto only specific soft/context transfer, not eligible direct-F
    anchors.

This should be evaluated first on valid before any full-test run.
```

## 2026-06-24 v18 source-contract high-anchor attempt rejected

Motivation:

```text
After rejecting v17, an offline sweep over v16 feature rows tested whether a broader
source-contract high anchor could recover false lows without hurting valid:

current low -> high when:
  source_level_consensus == no_eligible_clean_source_level_direct_f_vote
  soft_high_sources >= 3
  soft_low_sources + soft_straddling_sources <= 2
  not low_property_prior
  supports_higher_f_factor_count >= risk_for_lower_f_factor_count

On v16 artifacts this candidate left combined valid unchanged and improved full:
  valid macro-F1 stayed 0.725715
  full macro-F1 would move from 0.741861 to 0.764273
  full wrong->correct: 53, 85, 104
  full correct->wrong: none
```

Implemented temporary attempt:

```text
compiler/pipeline versions:
  bioavailability_specific_fa_fg_fh_v18_source_contract_high_anchor
  bioavailability_ma_specific_fa_fg_fh_v18_source_contract_high_anchor

policy addition:
  if the source-contract high-anchor condition is met, build_specific_decision_policy
  emits force_high so _apply_specific_decision_policy can override a low final LLM answer.

Temporary tests:
  - multiple moderate-transfer high direct-F sources without low-property prior force_high
  - the same soft high context with low-property prior remains fallback_uncertain
```

Static validation:

```text
Focused tests after implementation:
  41 passed
py_compile passed.
```

Live valid run:

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v18_source_contract_high_anchor_valid_finalonly_20260624

metrics:
  n_total=64
  n_successful=64
  macro-F1=0.625000
  accuracy=0.718750
  confusion: tn=7 fp=6 fn=12 tp=39

vs v16 combined valid:
  macro-F1=0.725715
  accuracy=0.812500
  confusion: tn=8 fp=5 fn=7 tp=44

v16 -> v18 valid changed:
  changed: 1, 8, 9, 11, 21, 23, 42, 43, 52, 60
  wrong->correct: 21, 43
  correct->wrong: 1, 8, 9, 11, 23, 42, 52, 60
```

Decision:

```text
Reject v18 and do not run full test. Although the deterministic policy looked good
when applied offline to v16 predictions, the live LLM rerun drifted badly on other
fallback_uncertain cases. This confirms that simply exposing/forcing one more source
contract while still rerunning an unconstrained final LLM is not stable enough.

The active code was restored to v16:
  bioavailability_specific_fa_fg_fh_v16_mechanism_alert_view
  bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_view

Temporary v18 code/tests were removed. Post-restore verification:
  focused tests: 39 passed in 1.33s
  py_compile passed
```

Updated methodological conclusion:

```text
Future optimization should not rely on full final-LLM reruns for small policy changes.
Either:
  1. make the final decision more deterministic from compiled contracts, then use the
     LLM only for explanation; or
  2. evaluate source-contract changes by a postprocess/calibration layer over completed
     v16 artifacts before spending on live reruns.

The next robust step is likely a deterministic final policy audit over all fallback_uncertain
cases, not another prompt-only or one-rule live rerun.
```

## 2026-06-24 accepted valid-selected deterministic postprocess over v16

Rationale:

```text
v17/v18 showed that small live final-LLM reruns can drift badly on fallback_uncertain
cases. Instead of rerunning the final LLM, add a deterministic postprocess stage over
completed v16 predictions and extracted v16 feature rows. The rule is selected on valid
only, then evaluated once on full test.
```

Code:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py
  - added candidate rule:
      force_state_or_starling_soft_high_anchor
  - rule behavior:
      current prediction must be low
      decision_state must be fallback_uncertain
      source_level_consensus must be no_eligible_clean_source_level_direct_f_vote
      soft_high_sources >= 2
      soft_low_sources <= 1
      soft_straddling_sources <= 2
      starling_high_sources >= 2
      => postprocess prediction to high

tools/chembl_tool/tasks/bioavailability_ma/specific_postprocess_policy.py
  - new reusable postprocess CLI.
  - reads original predictions + feature rows + selected_rule.json.
  - writes full predictions.jsonl, metrics.json, manifest.json, and report.md.
  - does not call the final LLM again.

tests:
  - test_starling_soft_high_anchor_candidate_is_conservative
  - test_apply_postprocess_policy_updates_only_rule_hits
```

Verification:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_postprocess_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py -q

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py \
  tools/chembl_tool/tasks/bioavailability_ma/specific_postprocess_policy.py
```

Result:

```text
41 passed in 1.08s
py_compile passed
```

Rule selection:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --features-jsonl outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v16_mechanism_alert_full_20260624/features.jsonl \
  --calibration-features-jsonl outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v16_mechanism_alert_valid_combined_20260624/features.jsonl \
  --selection-metric macro_f1 \
  --out-dir outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v16_valid_selected_postprocess_full_20260624
```

Selected rule:

```text
force_state_or_starling_soft_high_anchor
```

Valid calibration result:

```text
current v16 combined valid:
  macro-F1=0.725715
  accuracy=0.812500
  confusion: tn=8 fp=5 fn=7 tp=44

valid-selected postprocess on valid:
  output: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v16_valid_selected_postprocess_valid_20260624
  macro-F1=0.741841
  accuracy=0.828125
  confusion: tn=8 fp=5 fn=6 tp=45
  changed: idx24 low->high, label=1, correct=True
  correct->wrong: none
```

Full test result:

```text
postprocess output:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v16_valid_selected_postprocess_full_20260624

metrics:
  n_total=128
  n_successful=128
  macro-F1=0.756725
  accuracy=0.796875
  confusion: tn=25 fp=6 fn=20 tp=77
  prediction_distribution: high=83 low=45

changed vs v16:
  idx64 low->high, label=1, correct=True
  idx104 low->high, label=1, correct=True
  correct->wrong: none
```

Comparison:

```text
v16 mechanism-alert full:
  macro-F1=0.741861
  accuracy=0.781250
  confusion: tn=25 fp=6 fn=22 tp=75

best complete Starling v2 full:
  macro-F1=0.748792
  accuracy=0.796875
  confusion: tn=23 fp=8 fn=18 tp=79

accepted v16 + valid-selected postprocess full:
  macro-F1=0.756725
  accuracy=0.796875
  confusion: tn=25 fp=6 fn=20 tp=77
```

Decision:

```text
Accept the deterministic postprocess stage as the current best Bioavailability-specific
pipeline result. It beats the best complete Starling v2 full macro-F1 while preserving
v16's improved negative-class recall. The tradeoff versus Starling best is two fewer
true positives but two fewer false positives, giving higher macro-F1.

This accepted result should be described as:
  Bioavailability-specific Fa/Fg/Fh v16 mechanism-alert pipeline
  + valid-selected deterministic Starling soft-high direct-F postprocess.

Do not replace this accepted result with v17 or v18 live final-LLM reruns; both were
rejected on valid.
```

Additional combination audit:

```text
Tested applying force_state_or_mechanism_low_blockers together with the accepted
force_state_or_starling_soft_high_anchor postprocess.

valid:
  macro-F1=0.741841
  changed: idx24 low->high, correct
  no additional valid change beyond the accepted Starling soft-high anchor rule.

full:
  macro-F1=0.767830
  confusion: tn=26 fp=5 fn=20 tp=77
  wrong->correct: 21, 64, 104
  correct->wrong: none

Decision:
  Do not accept the combination yet. The mechanism_low_blocker component has no
  positive valid evidence in this combined setting; accepting it because it improves
  full would be test-set tuning. Keep it as a promising future candidate only if a
  separate calibration split or broader validation supports it.
```

## 2026-06-24 accepted postprocess v2 after remaining-failure audit

Subagent failure audit on accepted v16 + postprocess v1:

```text
Remaining false highs:
  21, 39, 59, 65, 77, 79

False-high conclusion:
  Do not add another high->low rule yet. Mechanism_low_blockers would fix full idx21,
  but it has no positive valid evidence, and broader high->low rules hit valid true-highs.
  Next high->low work should be source-contract cleanup, not a label rule.

Remaining false lows:
  4, 9, 17, 19, 36, 45, 46, 50, 53, 56, 62, 67, 68, 69, 78, 85, 87, 103, 113, 121

False-low conclusion:
  Broad soft-high promotion is unsafe, but one narrow no-evidence fallback is valid-safe:
  if the pipeline predicted low with no direct-F source context, no soft direct-F context,
  no risk/strong limiting factor, and no structural mechanism alert, do not let a bare
  single-molecule prior force low. Use high as the low-confidence forced binary fallback.
```

Code update:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py
  - added force_state_or_no_evidence_high_fallback
  - added force_state_or_valid_selected_high_rescues, which applies:
      1. force_state_or_starling_soft_high_anchor
      2. force_state_or_no_evidence_high_fallback

tests:
  - test_no_evidence_high_fallback_candidate_requires_empty_context_and_no_risk
  - updated candidate rule count to include the two new candidates
```

Verification:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_fallback_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_postprocess_policy.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_evidence_compiler.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_run_reasoning_pipeline_specific.py \
  tests/chembl_tool/tasks/bioavailability_ma/test_specific_retrieval.py -q

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m py_compile \
  tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py \
  tools/chembl_tool/tasks/bioavailability_ma/specific_postprocess_policy.py
```

Result:

```text
42 passed in 1.51s
py_compile passed
```

Rule selection:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy \
  --features-jsonl outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v16_mechanism_alert_full_20260624/features.jsonl \
  --calibration-features-jsonl outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v16_mechanism_alert_valid_combined_20260624/features.jsonl \
  --selection-metric macro_f1 \
  --out-dir outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/v16_valid_selected_postprocess_v2_full_20260624
```

Selected rule:

```text
force_state_or_valid_selected_high_rescues
```

Valid result:

```text
output:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v16_valid_selected_postprocess_v2_valid_20260624

metrics:
  macro-F1=0.758673
  accuracy=0.843750
  confusion: tn=8 fp=5 fn=5 tp=46
  prediction_distribution: high=51 low=13

changed:
  idx24 low->high, label=1, correct=True
  idx43 low->high, label=1, correct=True
  correct->wrong: none
```

Full result:

```text
output:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_specific_fa_fg_fh_v16_valid_selected_postprocess_v2_full_20260624

metrics:
  macro-F1=0.764273
  accuracy=0.804688
  confusion: tn=25 fp=6 fn=19 tp=78
  prediction_distribution: high=84 low=44

changed:
  idx64 low->high, label=1, correct=True
  idx68 low->high, label=1, correct=True
  idx104 low->high, label=1, correct=True
  correct->wrong: none
```

Comparison:

```text
best complete Starling v2 full:
  macro-F1=0.748792
  accuracy=0.796875
  confusion: tn=23 fp=8 fn=18 tp=79

accepted postprocess v1:
  macro-F1=0.756725
  accuracy=0.796875
  confusion: tn=25 fp=6 fn=20 tp=77

accepted postprocess v2:
  macro-F1=0.764273
  accuracy=0.804688
  confusion: tn=25 fp=6 fn=19 tp=78
```

Decision:

```text
Accept postprocess v2 as the current best Bioavailability-specific result.
It is selected by valid macro-F1, improves valid and full without any correct->wrong
postprocess flips, and beats the previous best complete Starling v2 full macro-F1.

Report the current best pipeline as:
  Bioavailability-specific Fa/Fg/Fh v16 mechanism-alert pipeline
  + valid-selected deterministic high-rescue postprocess
    (Starling soft-high anchor OR no-evidence high fallback).
```
