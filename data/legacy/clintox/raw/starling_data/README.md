# ClinTox source registry

This directory is the canonical local registry for ClinTox source material.
The source families are intentionally separate because they answer different
questions and must not vote on the same gold label.

| Directory | Role | May create strict clinical-trial-failure labels? |
|---|---|---|
| `tdc_clintox_v1/` | Frozen AACT positive source, FDA-approved comparator, joined reference, and upstream build inputs | Yes |
| `clintox_base_v1/` | Human clinical toxicity and FDA-status literature extractions | No; evidence and coverage only |
| `raw_v1/` | Six nonclinical, organ, genotoxicity, stress, cytotoxicity, and off-target evidence families | No; retrieval context only |
| `canonical_clinical_trial_failure_v1/` | Derived parent identities, labels, reconciliation, QA, and coverage audits | Generated from `tdc_clintox_v1/` |
| `canonical_retrieval_v1/` | Direct-gate decisions, accepted claims, QA sample, and retrieval build receipt | No; generated retrieval audit only |

The strict benchmark is built with:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.clintox.build_clinical_trial_failure_benchmark
```

Raw files are immutable. Every source directory carries checksums or a source
manifest, and derived outputs never overwrite raw inputs.

The complete task contract, current entry points, and frozen DeepSeek v3
results are recorded in
`tools/chembl_tool/tasks/clintox/CLINTOX_CLINICAL_TRIAL_FAILURE_V1.md`.
