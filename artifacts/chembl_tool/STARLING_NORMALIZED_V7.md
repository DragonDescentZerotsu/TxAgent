# Starling normalized V7 artifact archives

The current BBB Martins, Bioavailability, and Skin Reaction normalized V7
evidence libraries are stored under:

```text
artifacts/chembl_tool/tasks/<task>/starling_normalized_v7/
```

Each numbered stage is a deterministic `tar.zst` archive split into files of at
most 90,000,000 bytes. The task-level `manifest.json` records every restored
file hash, compressed archive hash, part hash, and size. `build_manifest.json`
is the packaged copy of the scientific build manifest.

Use the task-specific module to restore or verify a bundle. For example:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.starling_v7_artifact_store \
  verify-tracked

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.starling_v7_artifact_store \
  restore
```

Replace `bbb_martins` with `bioavailability_ma` or `skin_reaction` for the
other tasks. Restore refuses to overwrite a non-empty stage unless `--force`
is supplied. `--stages` can restore or verify selected stages without reading
the entire artifact.
