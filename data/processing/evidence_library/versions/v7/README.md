# Evidence library V7 construction release

This directory is the frozen, independently runnable release-local source for
V7. It pins `data.processing.evidence_library.shared.v1`; release-wide builders
live directly in this directory, while task policies and mappings live under
`tasks/`. Downstream heldout views are intentionally outside the release. These
sources build the corresponding data under
`data/evidence_libraries/<task>/v7/`.

Run a task builder from the repository root, for example:

```bash
python -m data.processing.evidence_library.versions.v7.tasks.bbb_martins.build_normalized_starling_evidence_library
```

New construction changes must be made in a copied release directory, never in
V7. Shared V1 modules are imported directly from
`data.processing.evidence_library.shared.v1`.
