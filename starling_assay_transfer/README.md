# Starling assay transfer

This directory contains the canonical package copied into TxAgent. Install and
run it independently from `starling_assay_transfer/`; scientific inputs remain
in TxAgent's existing `../data/`. The source checkout is not a runtime dependency.
Copy provenance is recorded in [the source inventory](provenance/source_inventory.json).

Two independent builders create record and context-molecule datasets from pinned V10 evidence and Gold splits. The code follows the scientific workflow:

**split → IID/OOD evaluation → filtering → sampling → augmentation → targets → Jinja dataset creation → explicit HF upload**

```text
assay_transfer/
  canonical.json / canonical.py   task settings and validation
  inputs.py                       pinned V10/Gold source readers
  split.py                        Gold memberships and disjointness rules
  evaluation.py                   IID/OOD selection, queries and rankings
  filtering.py                    eligibility, geometry, variance, filtered pools
  sampling/                       record/Gold pair selection and task samplers
  augmentation.py                 indirect/direct mixtures and Skin donor policy
  targets.py                      sigmoid and empirical binary targets
  templates/                      record.jinja and context.jinja
  dataset.py                      prompts, dataset files and HF publication
  build_record.py                 record pipeline
  build_context.py                context-molecule pipeline
  build.py / storage.py            CLI, immutable stages and file IO
  verify.py                       scientific and artifact verification
```

The split, evaluation and filtering modules prepare `pool/`. Both builders then run the visible stages `sample`, `augment`, `targets`, `dataset`, and `verify`. `all` runs through verification; upload is a separate command.

```bash
cd starling_assay_transfer  # from the TxAgent root
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m assay_transfer.build pin \
  --source-root ../data --input-manifest runs/example/inputs.json

python -m assay_transfer.build_record all \
  --task BBB_Martins --input-manifest runs/example/inputs.json \
  --output-root runs/example --smoke

python -m assay_transfer.build_context all \
  --task BBB_Martins --input-manifest runs/example/inputs.json \
  --output-root runs/example --smoke
```

Pinning selects the current named V10 releases and Gold-v1 splits and hashes the
actual files, including the copied package's geometry policies. Builds consume
that explicit manifest. Keep historical manifests unchanged; a fresh pin is a
new snapshot and must not silently replace an old run's inputs. After relocating
the checkout, pin a fresh manifest at a new path before starting a new build.

Both builders default to task-wide `--split-disjointness scaffold`; `parent` permits shared scaffolds while excluding shared training/heldout parents. Every level checks the same Gold validation/test cohort. OOD buckets derive from V10 records. BBB has no legacy parent lists or evaluation copies. BBB and Oral exclude L5.

Context training defaults to **1:1 indirect:direct augmentation**. Set `--augmentation-ratio` or use `--no-augment` (also ratio zero) for Gold-only training. Gold-only pinning accepts `--gold-only` and needs no evidence library. Context validation/test contain direct Gold only.

Sampling records pair identities and near/far classes. Targets are generated after augmentation. Empirical binary targets retain their declared fit population: the full sampled numeric pool before augmentation, rather than the selected mixture. Geometry/calibration statistics are fitted during pool preparation because filtering and sampling need them.

Use a fresh output root and omit `--smoke` for a full release. Full builds enforce scientific support and mixture budgets. Smoke outputs report shortfalls and cannot be uploaded as full releases. Every stage records input, effective contract, implementation, upstream and artifact hashes; `--resume` requires matching verified dependencies.

HF publication uses the [Hub commit API](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/package_reference/hf_api#create_commit). Install the optional upload dependency and choose the destination when publishing a verified full build:

```bash
python -m pip install '.[upload]'
python -m assay_transfer.build_context upload \
  --task BBB_Martins --input-manifest runs/full/inputs.json \
  --output-root runs/full --repo-id YOUR_ACCOUNT/YOUR_DATASET
```

Repeat the build's contract, disjointness and augmentation options when uploading. The default upload branch is named by the verified build hash; `--revision` selects a branch and `--private` creates a private repository. Upload includes dataset files and a verification receipt and records the returned commit locally. No upload happens during `all`.

The existing regression suite includes mocked upload checks and requires the
optional upload dependency. From this directory, run:

```bash
python -m pip install '.[upload]'
python -m unittest discover -s tests -v
```

See [the filesystem guide](docs/filesystem.md), [task settings](docs/canonical_contract.md), and [verification](docs/verification.md). Historical archives and generated runs remain in the original source checkout and are not included here.
