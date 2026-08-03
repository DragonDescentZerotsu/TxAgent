# Bioavailability normalized-v6 artifacts

This directory stores deterministic, Git-trackable snapshots of the eight local numbered artifact stages.
Each `stage.tar.zst.part-*` file is at most 90 MB. The root `manifest.json` records every part hash and every
restored file hash. `build_manifest.json` is the frozen scientific build manifest and is restored as the
local root `manifest.json` during a full restore. The local, directly readable copy remains under
`outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v6/` and is ignored by Git.

Restore only the finalized records, evidence catalog, and neighbor index:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.starling_artifact_store restore \
  --stages 03_records 04_evidence_catalog 05_neighbor_index
```

Verify tracked parts or restored local files:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.starling_artifact_store verify-tracked
python -m tools.chembl_tool.tasks.bioavailability_ma.starling_artifact_store verify-local
```

Packaging is deterministic and stage-selective. Repackage a completed stage with `package --stages <stage>`;
do not hand-edit archive parts or the generated manifest.

The comparison layers are `06_pair_buckets` (immutable v8 membership) and
`07_assay_transfer_policy` (one compressed pair-bucket-keyed n=25 eligibility,
variance-gate, SD, and local-percentile mapping). There is no record-level endpoint-policy stage.
