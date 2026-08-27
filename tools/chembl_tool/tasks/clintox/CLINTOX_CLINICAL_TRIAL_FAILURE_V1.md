# ClinTox clinical-trial-failure benchmark v1

`clinical_trial_failure_v1` is the definition-aligned ClinTox benchmark. It
predicts whether a molecular parent appears in the frozen AACT-derived list of
drugs associated with a clinical trial that failed for toxicity reasons.

It does **not** predict any toxicity finding. Ordinary adverse events,
preclinical toxicity, organ injury, market withdrawal alone, hERG/CYP/off-target
liability, and general cytotoxicity cannot create a positive label.

## Source contract

Canonical immutable inputs are under:

```text
data/starling_data/clintox/tdc_clintox_v1/
```

- `aacttox.csv.gz`: positive source;
- `sweetfda_approved_processed.csv.gz`: FDA-approved comparator source;
- `clintox.csv.gz`: upstream joined reference used only for reconciliation;
- the frozen AACT archive, phase table, SMILES cache, and upstream join script
  are retained for provenance.

The upstream DeepChem dataset files are pinned to commit
`61620011ac2d376606219e2f3fa539139f80d385`. The original PrOCTOR paper defines
the positive source as trials annotated `Terminated`, `Suspended`, or
`Withdrawn` and described as failing for toxicity reasons. TDC describes
ClinTox as drugs that failed clinical trials for toxicity versus successful
trial/approved comparators.

References:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC5074862/
- https://tdcommons.ai/single_pred_tasks/tox/
- https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/clintox_datasets.py

## Parent label rule

Raw source structures are normalized with the shared
`rdkit_fragment_parent.v1` identity policy.

```text
Y=1: at least one AACT toxicity-failure source record maps to the parent
Y=0: at least one FDA-approved comparator record maps to the parent and no
     AACT positive maps to it
```

FDA approval does not erase a toxicity-failed trial event. Therefore the 65
parents present in both source roles are positive. No record-majority vote,
confidence threshold, Starling category, or free-text inference is used.

## Frozen result

| Item | Count |
|---|---:|
| Source rows audited | 1,509 |
| Source rows with valid parent identity | 1,505 |
| Invalid/unresolved source structures | 4 |
| Labeled molecular parents | 1,428 |
| Y=0 | 1,324 |
| Y=1 | 104 |
| AACT/FDA source-role overlap parents | 65 |

The source-first reconstruction has 1,421 parents in common with the upstream
joined `clintox.csv.gz`; all 1,421 labels agree. Seven valid FDA-comparator
parents are retained only by the reconstruction because the historical join
used a lossy lower-case/remove-`=` SMILES key.

## Split

The benchmark root is:

```text
data/processed_clintox_clinical_trial_failure_v1/ClinTox/
```

| Split | Parents | Y=0 | Y=1 |
|---|---:|---:|---:|
| train | 1,144 | 1,060 | 84 |
| valid | 142 | 132 | 10 |
| test | 142 | 132 | 10 |

The split uses whole Bemis–Murcko scaffold groups and a deterministic
label-balance MILP. Parent identity and scaffold overlap are zero. The 129
parents without a Bemis–Murcko scaffold remain train-only so the empty
scaffold cannot leak across subsets.

## Audits and limitations

Derived canonical artifacts are under:

```text
data/starling_data/clintox/canonical_clinical_trial_failure_v1/
```

They contain every source identity decision, parent labels, joined-reference
reconciliation, a deterministic QA sample, `send_v2` human-clinical coverage,
the nine-source Stage-1 inventory, and the frozen direct-residual candidate
mapping. All literature coverage is descriptive: zero Starling rows create
labels.

The negative class is an approved-drug comparator, not proof that a molecule
can never cause toxicity. The source snapshot is historical, and the upstream
dataset license is unspecified. Before retrieval evaluation, valid/test
parents must be removed from every evidence index and retrieval must remain
`parent_disjoint`.

## Stage-1 evidence organization

The strict benchmark labels and literature evidence are separate lineages.
AACT/FDA rows are preserved as direct voting records. The seven independent
`send_v2` datasets remain source-local and indirect:

```text
Clinical.clinical_human_safety
Mechanism.in_vivo_toxicology
Mechanism.organ_specific_toxicity
Mechanism.genotoxicity_carcinogenicity
Mechanism.cellular_stress_pathways
Mechanism.general_cytotoxicity
Mechanism.off_target_ddi_exposure
```

The historical `literal_toxicity_trial_failure.v4` gate identified 338 rows in
the earlier clinical subset. Stage 1 reconciles those rows exactly to
`send_v2/human_clinical_toxicity` and stores only a keyed
`direct_residual_candidate` mapping. They remain indirect,
`pending_manual_review`, and retrieval-ineligible. Running the same regex over
the expanded source is not equivalent to the frozen mapping and must not
silently broaden it.

Stage 1 stops after source inventory, voting-record provenance, and light
cleaning. Endpoint normalization, measurement/unit construction, clustering,
pair buckets, collapse, heldout filtering, and retrieval indices are later
contracts.

## Reasoning prompt

The only supported profile is `tdc_source_aligned_v3`. It describes the target as a
source-defined frozen AACT toxicity-failure-associated class versus an
FDA-approved comparator-only class, rather than a re-adjudicated molecule-level
causal fact. It separates two decisions:

1. evidence provenance: only a concrete `Direct.clinical_trial_failure` row may
   be called an observed analog trial/development-failure outcome;
2. binary prediction: direct, clinical-context, mechanistic, and molecular
   evidence may all contribute according to relevance and transferability.

FDA approval is contextual rather than a hard negative because approval and a
source-positive association can coexist. Ambiguous status-only records,
combination attribution, patient-level discontinuation, withdrawal before
dosing, and postmarketing action are downweighted as direct analog evidence.
The output tokens remain `toxic`/`non_toxic` for evaluator compatibility, but
the prompt explicitly states that they do not mean universally toxic/safe.

Absence of a retrieved direct analog is an evidence gap, not a forced negative
prediction. `full_flat` keeps one LLM branch but every evidence row retains its
own `minimal_evidence.group.id`; the presence of one direct row does not promote
the rest of the mixed group to direct evidence. Structured validation checks
the reported direct-evidence status against the concrete retrieval rows but
does not change or restrict the predicted label.

Earlier prompt profiles were removed after the source-aligned contract was
frozen. A reusable single/group/final artifact must explicitly record
`task_prompt_profile=tdc_source_aligned_v3`; unversioned and older-profile
artifacts are rejected rather than silently reinterpreted.

Build the full source index and the valid/test-filtered benchmark index with:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.clintox.starling_retrieval --workers 8

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.build_starling_benchmark_indices \
    --splits scaffold --indices clintox_starling_full \
    --output-root outputs/paper \
    --benchmark-data-root data/processed_clintox_clinical_trial_failure_v1 \
    --benchmark-lineage clinical_trial_failure_v1
```

Visible evaluation may use `--harness-prefetch-tools`. This preserves visible
query and neighbor structures and visible tool text while moving the three
deterministic tool calls into the harness. It is a distinct manifest contract,
`deployment_visible_prefetched`, not an identity-blind run. DeepSeek runs use a
PARCC SSH tunnel from local `127.0.0.1:50001` to `epyc-1-4:50000`, with model
`deepseek-ai/DeepSeek-V4-Flash-0731`, not OpenRouter. Endpoint/model identity
must be recorded in the manifest.

Audit a completed strict run with:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.clintox.audit_clinical_trial_failure_agent \
    --run-root <matrix-run-root>/runs_deployment_visible_parent_disjoint/clintox \
    --conditions clintox__none clintox__starling_direct \
      clintox__starling_full_mechanism
```

The audit checks actual direct-row retrieval, positive-label coverage,
positive predictions with and without direct analogs, false direct-provenance
claims, non-direct event votes, full-mechanism direct-group presence,
structured-output retries, and whether evaluation labels entered LLM messages.
Positive-without-direct remains a coverage diagnostic rather than a hard error.

## Frozen DeepSeek v3 evaluation

The run contract was `deployment_visible_prefetched + parent_disjoint`,
`top_k_per_group=3`, `min_similarity=0.3`, temperature 0, and global
parallelism 64. All four conditions completed every sample with zero failed
runs and zero prompt-label leaks or direct-provenance violations.

Scaffold-valid (`n=142`, Y=0/Y=1 `132/10`):

| condition | accuracy | macro-F1 | positive precision | positive recall | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|
| none | 0.8521 | **0.6198** | 0.2381 | 0.5000 | 116 / 16 / 5 / 5 |
| direct | 0.8310 | 0.5528 | 0.1500 | 0.3000 | 115 / 17 / 7 / 3 |
| full-flat | 0.8028 | 0.5547 | 0.1538 | 0.4000 | 110 / 22 / 6 / 4 |
| full-mechanism | 0.8028 | 0.5322 | 0.1250 | 0.3000 | 111 / 21 / 7 / 3 |

The subsequently executed scaffold-test is explicitly a **post-test
diagnostic**, because the test split had already been inspected before this
prompt was frozen:

| condition | accuracy | macro-F1 | positive precision | positive recall | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|
| none | **0.8662** | 0.5506 | 0.1538 | 0.2000 | 121 / 11 / 8 / 2 |
| direct | 0.8521 | **0.5703** | 0.1765 | 0.3000 | 118 / 14 / 7 / 3 |
| full-flat | 0.7958 | 0.5028 | 0.0870 | 0.2000 | 111 / 21 / 8 / 2 |
| full-mechanism | 0.8099 | 0.5112 | 0.0952 | 0.2000 | 113 / 19 / 8 / 2 |

Direct evidence covered only `13/142` valid and `20/142` test molecules; only
`2/10` valid positives and `4/10` test positives had a direct analog. More
importantly, `9/11` valid and `11/14` test direct-vs-none prediction flips
occurred when the direct condition retrieved zero neighbors. Those flips are
run/prompt-surface instability, not direct-evidence effects. Broad retrieval
also produced false positives by letting transferable general hazards decide
the class despite the prompt's explicit prohibition. Therefore no retrieval
condition is promoted, and the same valid/test splits must not be used for
further prompt or selector tuning.

Canonical artifacts:

```text
valid:
  outputs/paper/molecular_evidence_agent_clintox_clinical_trial_failure_v1_scaffold_valid_epyc_deepseek_v4_flash_0731_tdc_source_aligned_v3/
test post-test diagnostic:
  outputs/paper/molecular_evidence_agent_clintox_clinical_trial_failure_v1_scaffold_test_epyc_deepseek_v4_flash_0731_tdc_source_aligned_v3_posttest_diagnostic/
trace audit:
  <root>/runs_deployment_visible_parent_disjoint/clintox/clinical_trial_failure_trace_audit.json
```

## Build and verification

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.clintox.build_clinical_trial_failure_benchmark

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  pytest -q tests/chembl_tool/tasks/clintox/test_clinical_trial_failure_benchmark.py \
    tests/chembl_tool/common/test_record_supported_benchmark.py
```
