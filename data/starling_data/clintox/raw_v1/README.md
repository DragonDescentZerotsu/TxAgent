# ClinTox Starling raw_v1

This directory contains the exact six Parquet sources and extraction guides from the supplied archive.
No upstream dataset revision, extraction-model version, mapping method, or license was supplied; see `SOURCE_MANIFEST.json`.
The raw Parquets retain `global_identifier` for fidelity. Derived code never consults it and uses only the source `SMILES` field.

| Source | Rows | Purpose |
|---|---:|---|
| `nonclinical_in_vivo_toxicity` | 345,174 | Nonclinical in vivo toxicity |
| `organ_specific_toxicity` | 720,399 | Organ-specific toxicity |
| `genotoxicity_carcinogenicity` | 641,417 | Genotoxicity and carcinogenicity |
| `cellular_stress` | 611,750 | Cellular stress |
| `general_cytotoxicity` | 826,011 | General cytotoxicity |
| `off_target_ddi_exposure` | 1,267,205 | Off-target, DDI, and exposure liability |

The derived pipeline performs only basic null/whitespace cleaning, source organization, molecule validation, evidence aggregation, and indexing. It does not perform v7 assay-transfer normalization.
