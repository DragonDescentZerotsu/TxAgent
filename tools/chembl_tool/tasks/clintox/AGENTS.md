# ClinTox canonical task contract

This file contains only current task-specific rules. Shared retrieval,
reasoning, service, batching, and artifact contracts live in the repository
root `AGENTS.md` and `tools/chembl_tool/tasks/AGENTS.md`.

## Target and lineage

The active evaluation dataset is the Conditioned Benchmark. Its ClinTox target
retains the frozen source-defined `CT_TOX` scientific contract at molecular-parent grain:

```text
Y=1: at least one frozen AACT toxicity-failure source record maps to the parent
Y=0: an FDA-approved comparator maps to the parent and no AACT positive does
```

This is a historical source-class prediction, not a fresh clinical causality
adjudication and not a general-toxicity label. FDA approval and an AACT positive
may coexist; the 65 overlapping parents remain positive.

Canonical source provenance and active benchmark root:

```text
data/starling_data/clintox/tdc_clintox_v1/
data/starling_data/clintox/canonical_clinical_trial_failure_v1/
data/conditioned_benchmark/ClinTox/scaffold/
```

Frozen counts:

```text
parents: 1,428 (Y=0: 1,324; Y=1: 104)
train/valid/test: 1,144/142/142
valid and test: Y=0/Y=1 = 132/10
parent identity overlap: 0
Bemis-Murcko scaffold overlap: 0
```

ClinTox has no accepted external-condition group, so every row uses the shared
null condition and the prompt omits a condition sentence. Old source-build,
`data/processed/ClinTox/`, and ChEMBL-native results are provenance only.

Full provenance, source hashes, limitations, and results are in
`CLINTOX_BENCHMARK.md`.

## Label and evidence separation

Only frozen AACT/FDA source roles create labels. No Starling extraction,
literature category, ChEMBL assay, majority vote, confidence threshold, or LLM
output may create or change gold labels.

Literature evidence is an independent retrieval lineage:

```text
data/starling_data/clintox/clintox_base_v1/
data/starling_data/clintox/raw_v1/
data/starling_data/clintox/canonical_retrieval_v1/
```

Current retrieval hierarchy:

```text
Direct.clinical_trial_failure
Clinical.clinical_human_safety
Mechanism.in_vivo_toxicology
Mechanism.organ_specific_toxicity
Mechanism.genotoxicity_carcinogenicity
Mechanism.cellular_stress_pathways
Mechanism.general_cytotoxicity
Mechanism.off_target_ddi_exposure
```

`Direct.clinical_trial_failure` requires explicit trial-, recruitment-, or
development-level stoppage caused by toxicity or safety. Patient treatment
discontinuation, dose reduction, DLT/MTD, adverse events, organ injury, market
withdrawal alone, hERG/CYP/transporter/DDI activity, cytotoxicity, and animal
toxicity are context or mechanism evidence, not observed trial failure.

`full_flat` is one branch, but every row retains its own
`minimal_evidence.group.id`. A direct row never promotes adjacent contextual or
mechanistic rows to direct evidence.

The direct gate remains `pending_manual_review` because `clintox_base_v1` does
not contain `qualifying_conditions`. Missing is not equivalent to empty. Keep
that limitation explicit in manifests and reports.

## Prompt contract

The only supported profile is:

```text
tdc_source_aligned_v3
label_scope: aact_toxicity_failure_association_vs_fda_comparator.v1
```

It must state that `toxic`/`non_toxic` are evaluator tokens for the frozen
source classes, not universally toxic/safe claims. A transferable direct analog
is strong evidence but is not a hard positive gate. Absence of direct evidence
is an evidence gap, not proof of the comparator class. FDA approval is context,
not a veto. Generic hazard evidence may adjust probability but must not decide
the class by itself.

Earlier prompt profiles were removed. Reusable single/group/final artifacts
must explicitly record `task_prompt_profile=tdc_source_aligned_v3`; old or
unversioned branches are rejected.

Task prompt code:

```text
tools/chembl_tool/tasks/clintox/prompt_profiles.py
tools/chembl_tool/tasks/clintox/run_reasoning_pipeline.py
```

Do not add a task-specific deterministic label gate or postprocess an otherwise
valid model prediction. Structured validation may enforce schema and direct-row
provenance only.

## Current result and stop boundary

The 2026-08-16 DeepSeek run used:

```text
model: deepseek-ai/DeepSeek-V4-Flash-0731
endpoint: local 127.0.0.1:50001 -> PARCC epyc-1-4:50000
visibility: deployment_visible_prefetched
identity policy: parent_disjoint
top_k_per_group: 3
min_similarity: 0.3
parallelism: 64
```

Scaffold-valid macro-F1:

```text
none:           0.6198
direct:         0.5528
full-flat:      0.5547
full-mechanism: 0.5322
```

Retrieval failed the valid promotion gate. The later scaffold-test is only a
`post-test diagnostic` because the test split had already been inspected:

```text
none/direct/full-flat/full-mechanism: 0.5506/0.5703/0.5028/0.5112
```

Direct evidence covered 13/142 valid and 20/142 test molecules. Most
direct-vs-none flips occurred with zero direct neighbors (valid 9/11; test
11/14), so they are not direct-evidence effects. Broad retrieval also caused
false positives by turning general hazard into the source label.

Do not tune another prompt, selector, threshold, or label policy on these same
valid/test splits. Any future method claim needs a new independently frozen
evaluation set or a source/data-contract change.

## Canonical entry points

Build the source-reconstructed benchmark:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.clintox.build_clinical_trial_failure_benchmark
```

Build the full literature library and heldout-filtered scaffold index:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.clintox.starling_retrieval --workers 8

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.build_starling_benchmark_indices \
    --splits scaffold --indices clintox_starling_full \
    --output-root outputs/paper \
    --benchmark-data-root data/conditioned_benchmark \
    --benchmark-lineage conditioned_benchmark
```

Reasoning entry points:

```text
tools/chembl_tool/tasks/clintox/run_reasoning_pipeline.py
tools/chembl_tool/tasks/clintox/run_reasoning_batch.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
```

The task defaults use the current scaffold split, heldout-filtered index,
`tdc_source_aligned_v3`, and the PARCC DeepSeek tunnel. The batch wrapper uses
the shared `common/task_workflows/reasoning_batch.py`; do not duplicate batch,
cache, prompt-pool, or tool-prefetch logic locally.

Audit a completed matrix:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.clintox.audit_clinical_trial_failure_agent \
    --run-root <matrix-root>/runs_deployment_visible_parent_disjoint/clintox \
    --conditions clintox__none clintox__starling_direct \
      clintox__starling_full_flat clintox__starling_full_mechanism
```

The audit checks retrieval coverage, direct provenance, prompt label leakage,
structured retries, condition metrics, and paired flips. Automated provenance
success does not prove semantic prompt compliance; inspect the real retrieval
payload -> group summary -> final adoption chain for discordant samples.

## Service and LLM visibility

Neighbor retrieval is harness-side evidence prefetch, not an LLM function tool.
The resident service exposes only the shared tools:

```text
molecule_properties
properties_compare
mmp_structure_compare
```

LLM prompts receive only tool `output.text`, not full structured internals.
Every evidence row must pass through `minimal_evidence.v1`; deterministic
direction, strength, label-vote, and override fields must not be shown to the
model.

## Verification

Run at minimum:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  pytest -q tests/chembl_tool/tasks/clintox
```

Before a benchmark run or publication claim also verify source hashes,
canonical manifests, heldout-parent overlap, scaffold overlap, index metadata,
model identity, profile, split, visibility, failed-stage count, and the trace
audit artifact.
