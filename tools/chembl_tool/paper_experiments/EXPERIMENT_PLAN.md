# Experiment Plan

## Research questions

### RQ1: Does similar-molecule retrieval improve molecular property classification?

For each task, compare `none` with `chembl_direct`. The single-molecule prior is frozen within each task/query, so the paired difference is attributable to retrieved evidence and final use of that evidence.

### RQ2: Is literature-derived Starling evidence more useful than curated ChEMBL evidence?

- BBB: compare ChEMBL direct with Starling direct.
- Bioavailability: compare ChEMBL direct with Starling direct full.
- Bioavailability mechanisms: compare ChEMBL full mechanism with Starling full mechanism.

Coverage, evidence volume, and per-group availability are reported next to performance because source usefulness is a joint property of quality and retrievability.

### RQ3: Does task-mechanism decomposition improve reasoning?

For every task, compare `full_flat` with `full_mechanism` within the same source. These conditions contain the same selected evidence rows; only grouping and parallel mechanism reasoning differ. Bioavailability provides both ChEMBL and Starling versions of this ablation.

### RQ4: Does non-numerical literature evidence add value?

For Bioavailability direct Starling evidence, compare:

- scalar KNN over numerical direct-F values
- GLM with numerical direct-F evidence only
- GLM with numerical plus non-numerical evidence and experimental context

Only two Starling content settings are used: `numeric_only` and `full`. No finer content taxonomy is part of the main experiment.

## Frozen experiment matrix

| Task | None | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism | Numeric Starling | Scalar KNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | yes | yes | yes | yes | yes | no | no | no | no |
| Skin_Reaction | yes | yes | yes | yes | no | no | no | no | no |
| ClinTox | yes | yes | yes | yes | no | no | no | no | no |
| Bioavailability_Ma | yes | yes | yes | yes | yes | yes | yes | yes | yes |

This is 21 GLM conditions plus one scalar KNN control.

## Fixed model and retrieval settings

```text
requested model: zai-org/GLM-5.2-FP8
temperature: 0
maximum output tokens: 20480
identity blind: true
exact query exclusion: true
Morgan radius/bits: 2 / 2048
top k per group: 3
minimum similarity: 0.30
single prior: frozen from each task's none condition
SDK transport retry limit: 2; structured JSON validation: at most 4 total attempts; the last two attempts deterministically reserialize the same input to recover provider-side degeneration; remaining incomplete samples are excluded and rerun visibly at batch level

Provider request-size guard: cleaned group payloads above 750 KB use deterministic even-spacing with at most 100 evidence rows per oversized neighbor; truncation counts and byte metadata remain visible in the prompt trace.
```

The endpoint-returned model identifier is recorded from every response rather than inferred from the requested alias.

## Primary and secondary outcomes

Primary outcome: test macro-F1.

Secondary outcomes:

- accuracy and class-specific confusion matrix
- retrieval coverage overall and by mechanism group
- prompt, completion, and total token count
- structured-output retries and failed runs
- number of LLM calls and frozen-prior reuse count
- query-SMILES trace leak count
- prompt-boundary structure, molecule-identifier, and source-name leak counts

## Statistical analysis

- 95% test-set bootstrap interval for each condition's macro-F1
- paired bootstrap interval for each predeclared macro-F1 difference
- exact two-sided McNemar test on paired correctness
- Holm-adjusted McNemar p-values across the predeclared comparison family
- no test-set threshold selection or post-hoc deterministic label correction

The paired comparisons are predeclared in `summarize_results.py`. Results with failed samples or identity leaks are not paper-ready until repaired and re-audited. Prompt audits inspect saved system/user/tool request inputs; assistant responses are not treated as upstream identity disclosure.

## Contextual non-agent baselines

Existing MiniMol results can be reported as contextual learned baselines, but they are not part of the retrieval ablation and should not be used to select agent settings. Historical DeepSeek and Bioavailability expert-policy runs are provenance references only; they are not directly comparable to the frozen GLM identity-blind matrix and are excluded from the main paper table.
