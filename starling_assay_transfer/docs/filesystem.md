# Workspace layout

- `assay_transfer/`: active scientific pipeline and the two builders. This is the implementation to edit and package.
- `tests/`: runnable regression checks; `docs/`: contract, workflow and verification documentation.
- `runs/`: generated build inputs, intermediate data, datasets, validation outputs and package staging. Ignored by Git. A pinned input manifest and the verification receipt identify a run's source and completion.
- `provenance/`: source-copy inventory and validation receipt. The inventory hashes the original copied files; documentation is adapted for this location.
- `.venv/`: optional isolated installation environment, ignored by Git.

From this directory, supply `--source-root ../data` to reuse TxAgent's authoritative scientific inputs: `../data/evidence_libraries/<task>/CURRENT` and `../data/gold_labels/<Task>/v1/scaffold/`. Pinning resolves the named V10 releases and hashes the required files, including the Skin routing sidecar under `../data/artifacts/evidence_library_assets/`. No scientific payloads are duplicated here. Builds use the frozen manifest rather than re-reading `CURRENT`.

Historical manifests retain their original pinned inputs. New builds use fresh manifests generated from this package, including its local geometry policies. Relocating the checkout requires a fresh manifest at a new path for new builds because pinned paths are absolute.

A generated task directory looks like this:

```text
runs/<run>/record/<task>/           # or context_molecule/<task>/
  pool/                           # memberships, OOD decisions, filtering/calibration audits
  sampling/                       # raw selected pairs; probabilities unset
  augmentation/                   # selected mixtures; probabilities unset
  targets/                        # scored pairs and target calibration
  dataset/<family>/
    train/data.parquet
    validation_ranking/data.parquet
    test_ranking/data.parquet
    README.md
    manifest.json
  verification/<family>/          # completion receipt
  uploads/                        # HF destination, branch and returned commit, if uploaded
```

Augmented context pools contain `pool/indirect/`; Gold-only builds omit it. Full versus representative scope is recorded explicitly. A folder's name alone is not evidence of a complete release.

Historical archives, generated runs and assistant task workspaces remain in the original `/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer` checkout. They are not runtime dependencies or part of this copy. Historical path references inside package provenance and documentation retain their original meaning.
