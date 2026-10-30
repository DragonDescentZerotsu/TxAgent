# TxAgent — archived paper implementation

This branch is a historical source archive. Read [archive scope and restoration limits](ARCHIVE.md)
before running it. Active handoff work is on `merge_release_v1`; the sections below
retain their historical context and may refer to separately retained experiment assets.

Molecular evidence retrieval and reasoning over condition-aware benchmarks.
Shared tools compute molecular properties and structural comparisons; task-specific
experimental records provide evidence for the reasoning model.

## Current paper results

Use the [current comparison](outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md)
and [complete ML Appendix](outputs/paper/llm_vs_standard_baselines_20260924/ML_APPENDIX.md).
The 198-cell ML table includes RASAR and RASAR+indirect across both benchmarks,
with the corrected seed-42 Skin/Carcinogens TDC baselines. The
[paper guide](tools/chembl_tool/paper_experiments/README.md#current-comparison-figures)
owns entrypoints and reproduction. Both final pastel bar layouts (vertical task panels
or all-horizontal task panels) and the separate mean figure use one shared renderer;
left-line/right-point figures are retired.
Earlier seed-1 scores are historical.

The repository is now in [release preparation](tools/chembl_tool/paper_experiments/README.md#release-preparation).
Code/documentation consolidation and clean-environment reproduction remain pending.

For post-paper maintenance, the molecular tool service is stopped. Its caches
are archived on shared storage; [restore and launch instructions](tools/service/README.md#production-launch-on-node002)
are available when tool computation is needed again. MiniMol's preserved runtime
dependencies now live on shared storage; see the [runtime setup](baselines/minimol/README.md).
The `storage maintenance receipt` (local-only historical record)
records verified moves, duplicate removal and retained historical files.
The `build-cache cleanup receipt` (local-only historical record)
records the subsequent verified shared archive and removal of the local build cache;
archive locations and restoration guidance are linked from the
[cache audit](outputs/paper/local_build_cache_audit_20260930/README.md).

## Score a DILI evidence record

[DILI record score](dili_score/README.md) is a lightweight Python package for
rule-based relevance and structured completeness. It runs offline with no
third-party runtime dependencies: `pip install .`, then
`from dili_score import score_dili_record`. Pass one record dictionary and read
`score_dili_record(record)["score"]`; the full response includes explanations and
unresolved-result flags. See the guide for lightweight installation and limits.

## Find the current retrieval records

Start with the **[current level-record dataset](artifacts/chembl_tool/starling/current_level_records/README.md)**.
It contains ordinary Parquet files with source text, provenance and record-level
family assignments. Split-specific tables reference the indexed representative cards.
These are retrieval candidates; a query's selected cards live in its experiment trace.

| Task | Browse source membership and split card references |
|---|---|
| Skin reaction | [skin_reaction](artifacts/chembl_tool/starling/current_level_records/skin_reaction/) |
| Ames | [ames](artifacts/chembl_tool/starling/current_level_records/ames/) |
| DILI | [dili](artifacts/chembl_tool/starling/current_level_records/dili/) |
| Carcinogens | [carcinogens](artifacts/chembl_tool/starling/current_level_records/carcinogens/) |

BBB and Bioavailability are in the same dataset directory. Read its
[manifest](artifacts/chembl_tool/starling/current_level_records/manifest.json)
for table paths, counts and file hashes. Current means the snapshot in the checked-out
commit; historical experiment results can use different frozen inputs.

For complete canonical Stage-03 snapshots, including excluded records and audit files,
use the [archive manifest](artifacts/chembl_tool/starling/current_records/manifest.json).
The [retrieval contract](tools/chembl_tool/paper_experiments/current_starling_retrieval.json)
selects the current source, overlay, catalog and both split indices. Follow the
**[restore, rebuild and verification guide](tools/chembl_tool/paper_experiments/CURRENT_STARLING_RETRIEVAL.md)**;
do not choose inputs by the largest version number in a directory name.

## Run and inspect experiments

- [Benchmark definition and splits](tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md)
- [Shared experiment entrypoints](tools/chembl_tool/paper_experiments/README.md)
- [DeepSeek/Luna scaffold-indirect follow-up and reuse entrypoints](tools/chembl_tool/paper_experiments/README.md#scaffold-disjoint-indirect-follow-up)
- [Current results and limitations](tools/chembl_tool/paper_experiments/RESULTS.md)
- [Machine-readable result registry](tools/chembl_tool/paper_experiments/current_conditioned_results.json)
- [Molecular tool service](tools/service/README.md)
- [Artifact retention and historical-version policy](tools/chembl_tool/paper_experiments/TRACE_RETENTION.md)

Use the manifests and linked receipts to establish reproducibility. Historical source
reviews are provenance; obsolete materialized indices are not alternate defaults.

### TDC BBB indirect ablation

The [BBB guide and Codex handoff](tools/chembl_tool/tasks/bbb_martins/INDIRECT_ABLATION.md#codex-handoff) provide the official 406-row TDC test set and the current assay-aware filter. Historical test-tuned scores used the retired filter; the current filter has offline verification only. Optional ablations reuse the shared runner.
