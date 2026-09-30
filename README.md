# TxAgent

Molecular evidence retrieval and reasoning over condition-aware benchmarks.
Shared tools compute molecular properties and structural comparisons; task-specific
experimental records provide evidence for the reasoning model.

## Environment configuration

Use the Python environment containing the project dependencies. Python orchestration
defaults to that interpreter; existing `--python-executable` overrides still apply.
MiniMol uses an installed package or a sibling `minimol/` checkout; set
`TXAGENT_MINIMOL_SOURCE` or the existing `--minimol-source` option for another location.
Conda launchers use `CONDA_EXE` or `conda` on PATH; Hugging Face caches follow `HF_HOME`.
The optional SSH watchdog accepts `--ssh-host` or `TXAGENT_SSH_HOST`.

The historical activity-transfer helpers accept explicit HF dataset IDs through
`--repos` / `TXAGENT_TRANSFER_DATASETS` (space-separated), or
`--hf-repo` / `TXAGENT_TRANSFER_HF_REPO`; feature preparation also accepts local
`--train-jsonl`. Use the existing input/output path arguments for saved runs.
The ChEMBL data symlink points to a sibling data checkout; replace it with a link
to your local ChEMBL data when using those optional tools.

Review copies use repository-relative paths and neutral archive ownership.
Historical path strings are normalized; scientific record payloads, labels and
scores retain their original values. Artifact manifests bind the distributed bytes.
Hashes changed by release metadata normalization refer to the distributed copy;
this does not represent a new experiment or revised scientific result. In the
Carcinogens archive, only document-location fields within audit provenance were
normalized; all other record cells were checked for equality. Provider-pool examples
use local loopback endpoints and generic credential variable names; pass your
existing provider configuration when resuming a saved deployment.
Reading saved results or using the offline scorer does not start a tool service.

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
- [Current results and limitations](tools/chembl_tool/paper_experiments/RESULTS.md)
- [Machine-readable result registry](tools/chembl_tool/paper_experiments/current_conditioned_results.json)
- [Molecular tool service](tools/service/README.md)
- [Artifact retention and historical-version policy](tools/chembl_tool/paper_experiments/TRACE_RETENTION.md)

Use the manifests and linked receipts to establish reproducibility. Historical source
reviews are provenance; obsolete materialized indices are not alternate defaults.
