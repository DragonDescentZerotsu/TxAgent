# BBB Martins normalized Starling v6 artifacts

This directory stores the reproducible, stage-separated output of the BBB Martins
normalization pipeline. The payload mirrors the Bioavailability v6 architecture:
source cleaning, scalar and categorical normalization, organized records, pair
buckets, transfer policy, split-specific held-out filtering, relational molecule
evidence, compact neighbor indices, and final audits.

The source-complete normalized artifact contains 581,946 rows before organization
and 581,702 valid organized records. It contains 80,851 immutable pair buckets, of
which 640 are eligible for transfer. Transfer-policy calibration excludes the
1,947-parent union of the random and scaffold held-out sets from `direct_bbb`
only; passive, efflux, and influx mechanism evidence is retained.

| Benchmark split | Filtered records | Direct rows excluded | Index molecules | Molecule-family memberships |
|---|---:|---:|---:|---:|
| random | 574,737 | 6,965 | 36,717 | 46,389 |
| scaffold | 572,319 | 9,383 | 36,695 | 46,410 |

Both indices expose the same four exact groups:

- `Tier 1.starling_direct_bbb_evidence`
- `Mechanism.passive_permeability`
- `Mechanism.efflux_transport`
- `Mechanism.influx_transport`

`manifest.json` records the hashes of every source file, archive, and archive
part. `build_manifest.json` is the packaged copy of the root build manifest.
Parts are capped at 90,000,000 bytes to remain repository-portable.

Restore and verify from the repository root with:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.starling_artifact_store restore

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.starling_artifact_store verify-tracked

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.starling_artifact_store verify-local
```

The globally reconciled auxiliary mapping is versioned in
`tools/chembl_tool/tasks/bbb_martins/data_processing/` and is pinned in stage 02
by SHA-256 `9298f3572eacfbfbad3c7ca2c0c2894aeecb97feed9f37408717072806d32aa4`.
