# ClinTox send_v2 source

This directory is the active immutable seven-source ClinTox delivery.

Archive SHA-256: `bf6da36bf2ac347d4763e5c3a234292c7f5c04576cccc4b75f6a4741246ac4ea`.

| source | rows | RDKit-valid | RDKit-invalid |
|---|---:|---:|---:|
| `human_clinical_toxicity` | 584,307 | 582,931 | 1,376 |
| `nonclinical_in_vivo_toxicity` | 578,856 | 575,914 | 2,942 |
| `organ_specific_toxicity` | 674,318 | 670,632 | 3,686 |
| `genotoxicity_carcinogenicity` | 570,749 | 566,646 | 4,103 |
| `cellular_stress` | 526,737 | 521,643 | 5,094 |
| `general_cytotoxicity` | 807,290 | 800,382 | 6,908 |
| `off_target_ddi_exposure` | 1,104,657 | 1,098,331 | 6,326 |

All source rows contain a nonempty `SMILES`. Invalid structures remain auditable and fail closed.
The clinical source lacks `qualifying_conditions`; it remains indirect evidence and does not define the benchmark label.

The exact supplied archive is tracked as verified sub-100 MB parts under
`artifacts/chembl_tool/tasks/clintox/clintox_send_v2_source/`. Extracted
Parquets are byte-identical local restores and are not duplicated as Git
blobs. Restore them in a fresh checkout with:

```bash
python -m tools.chembl_tool.tasks.clintox.starling_source_artifact_store restore-source
```
