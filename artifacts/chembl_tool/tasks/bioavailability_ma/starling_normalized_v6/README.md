# Bioavailability normalized-v6 artifacts

This directory stores deterministic, Git-trackable snapshots of the nine local numbered artifact stages.
Each `stage.tar.zst.part-*` file is at most 90 MB. The root `manifest.json` records every part hash and every
restored file hash. `build_manifest.json` is the frozen scientific build manifest and is restored as the
local root `manifest.json` during a full restore. The local, directly readable copy remains under
`outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v6/` and is ignored by Git.

Restore the complete records/bucket policy and the split-filtered evidence/index layers:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.starling_artifact_store restore \
  --stages 03_records 04_pair_buckets 05_assay_transfer_policy \
           06_remove_heldout_overlap 07_molecule_evidence 08_neighbor_index
```

Verify all tracked parts, then verify exactly the local stages restored above:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.starling_artifact_store verify-tracked

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.starling_artifact_store verify-local \
  --stages 03_records 04_pair_buckets 05_assay_transfer_policy \
           06_remove_heldout_overlap 07_molecule_evidence 08_neighbor_index
```

Packaging is deterministic and stage-selective. Repackage a completed stage with `package --stages <stage>`;
do not hand-edit archive parts or the generated manifest.

`04_pair_buckets` and `05_assay_transfer_policy` use the complete Stage-03 records.
`06_remove_heldout_overlap` then removes parent-matching records from Direct HF only and
materializes random/scaffold record views. Only those views feed `07_molecule_evidence`
and `08_neighbor_index`; no unfiltered evidence/index branch is stored. The pair-bucket
policy remains one compressed global n=25 eligibility, variance-gate, SD, and
local-percentile mapping. There is no record-level endpoint-policy stage.
