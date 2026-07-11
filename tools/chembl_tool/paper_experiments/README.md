# Molecular Evidence Agent Paper Experiments

This directory freezes the paper-facing experiment matrix without embedding task logic in the runner.

## Shared contract

All LLM experiments use the same conditions:

- GLM-5.2 through the Penn LiteLLM OpenAI-compatible endpoint
- temperature 0 and maximum output length 20,480 tokens
- exact-query exclusion during retrieval
- identity-blind prompting: the harness uses structures for retrieval and tool execution, then removes query and neighbor structures and identifiers before the LLM call
- one frozen single-molecule analysis per task/query, reused across retrieval conditions so paired ablations differ only in retrieved evidence and grouping
- three retrieved neighbors per evidence group at Morgan similarity at least 0.30
- the same single-molecule, group-reasoning, final-synthesis, JSON validation, retry, trace, and batch-evaluation workflow

The experiment modes are defined in `common/experiment_retrieval.py`:

- `none`: query properties only; no retrieved evidence
- `direct`: only task-declared direct outcome evidence
- `full_flat`: the union of all task-declared mechanism evidence in one group
- `full_mechanism`: exactly the same evidence union, separated into mechanism groups

Each task declares only how source endpoint groups map to direct and mechanism families in its `experiment_config.py`.

## Frozen matrix

Build the frozen Starling indices before running the matrix:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_starling_evidence_library \
  --mode all \
  --out-root outputs/paper/molecular_evidence_agent/evidence/bbb_starling \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope full \
  --evidence-content full \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --scope direct \
  --evidence-content numeric_only \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_direct_numeric \
  --workers 128
```

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent --list
```

Run selected experiments:

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --experiments bioavailability_ma__none bioavailability_ma__starling_full_mechanism
```

The API key is read from `GLM_API_KEY` by default. The value is never written to commands, manifests, or traces.

## Analysis

```bash
python -m tools.chembl_tool.paper_experiments.summarize_results
```

This produces per-condition metrics and 95% bootstrap intervals, retrieval coverage, token and retry counts, query-SMILES trace checks, prompt-boundary structure/identifier/name audits, paired macro-F1 deltas, and exact McNemar tests under:

```text
outputs/paper/molecular_evidence_agent/analysis/
```

The frozen full-run results and interpretation are summarized in [RESULTS.md](RESULTS.md).

The Bioavailability scalar KNN control is separate because it consumes only numeric direct-F values and performs no LLM reasoning:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_scalar_knn
```
