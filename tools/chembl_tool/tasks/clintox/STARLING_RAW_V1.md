# ClinTox Starling raw_v1

## Scope

`starling_raw_v1` is a raw-first literature evidence source. It supports toxicity evidence retrieval and a
new candidate benchmark, `ClinTox_Human_Toxicity`, for explicit human clinical organ injury. It is not a
reconstruction of the TDC/MoleculeNet CT_TOX clinical-trial-failure label.

The supplied archive has SHA-256
`5f34b79a9b8740ef2371b2f691644f52245bbc6903963abccc92e6000656170b`. It does not provide an upstream
dataset revision, extraction model/software version, entity-mapping method, or license. Those gaps are
recorded rather than inferred in `data/starling_data/clintox/raw_v1/SOURCE_MANIFEST.json`.

## Raw sources and columns

All sources share `paragraph_idx`, `support_text`, `global_identifier`, `confidence`,
`needs_more_context`, `pmid`, `extraction_id`, and `SMILES`.

| Source | Rows | Source-specific columns |
|---|---:|---|
| `nonclinical_in_vivo_toxicity` | 345,174 | `evidence_type`, `endpoint_value`, `endpoint_unit`, `administered_dose`, `animal_context`, `exposure_context`, `observed_effect`, `qualifying_conditions`, `extra_details` |
| `organ_specific_toxicity` | 720,399 | `organ_system`, `toxicity_endpoint`, `effect_status`, `evidence_context`, `biological_system`, `exposure_regimen`, `quantitative_result`, `qualifying_conditions`, `extra_details` |
| `genotoxicity_carcinogenicity` | 641,417 | `evidence_category`, `assay_type`, `study_context`, `endpoint`, `result_direction`, `biological_system`, `exposure_conditions`, `qualifying_conditions`, `extra_details` |
| `cellular_stress` | 611,750 | `stress_endpoint`, `effect_direction`, `evidence_basis`, `mechanistic_effect`, `target_or_pathway`, `biological_model`, `dose_and_duration`, `qualifying_conditions`, `extra_details` |
| `general_cytotoxicity` | 826,011 | `cell_model`, `endpoint_type`, `result_value`, `result_unit`, `test_concentration`, `exposure_time_h`, `assay_method`, `qualifying_conditions`, `extra_details` |
| `off_target_ddi_exposure` | 1,267,205 | `evidence_type`, `target_or_endpoint`, `target_identifier`, `result_metric`, `result_value`, `result_unit`, `assay_context`, `qualifying_conditions`, `extra_details` |
| **Total** | **4,411,956** | |

The exact raw Parquets retain `global_identifier` for source fidelity. All derived stages omit it and use
only source `SMILES`; the supplied identifier map is never consulted.

## File-system architecture

```text
data/starling_data/clintox/raw_v1/
  <source>/{extractions.parquet,extraction_guidance.json}
  SOURCE_MANIFEST.json
  README.md

outputs/chembl_tool/tasks/clintox/evidence_library/starling_raw_v1/
  01_cleaned/records.parquet
  02_organized/{records.parquet,structure_exclusions.parquet}
  03_evidence_catalog/{evidence.jsonl,record_bridge.parquet,summary.json}
  04_neighbor_index/{neighbor_index.pkl,index_meta.json}
  05_audits/{summary.json,gold_qa_sample.parquet,gold_qa_status.json}
  06_record_supported_v2_scaffold_view/{evidence.jsonl,neighbor_index.pkl,index_meta.json,summary.json}

data/processed_starling/ClinTox_Human_Toxicity/
  parent-vote provenance and first-version random/scaffold artifacts

data/processed_starling_record_supported_v2/ClinTox_Human_Toxicity/scaffold/
  train.jsonl, valid.jsonl, test.jsonl, molecule-label audits, held-out contract

artifacts/chembl_tool/tasks/clintox/starling_raw_v1/
  deterministic stage.tar.zst.part-* files and manifests
```

The cleaning stage only trims/collapses whitespace and maps the shared basic-null sentinels to null. It
does not canonicalize endpoints or units, parse scalar values, infer reference semantics, construct pair
buckets, calibrate distances, or decide assay-transfer eligibility.

## Qualifying conditions

The gold adapter uses the existing shared `has_reported_text` helper. A qualifying condition is empty only
when the field is an actual null/NaN, blank, or exactly `nan`, `none`, `null`, `n/a`, or `na` after
case-folding and trimming. Values such as `-` and `unspecified` are nonempty. A nonempty value means the
claim depends on a material context—such as a susceptible genotype, disease model, co-treatment, unusual
route/formulation, or critical pregnancy stage—so it is retained as evidence but excluded from the broadly
applicable gold benchmark.

## Candidate benchmark result

The task adapter accepts v2 rows only when:

- `evidence_context == human_clinical`;
- `effect_status` is exactly `injury_observed` or `no_injury_observed`;
- `organ_system` is one of the nine declared systems;
- `qualifying_conditions` is empty under the shared helper;
- `needs_more_context` is false;
- source `SMILES`, `support_text`, `pmid`, and `toxicity_endpoint` are present; and
- source `SMILES` resolves through `rdkit_fragment_parent.v1`.

There is no confidence threshold and no label inference from prose or numeric values. Each accepted source
row is one vote, including multiple rows from the same PMID. Parent labels require at least 70% agreement;
exact ties are rejected.

The frozen candidate has 79,571 labeled source rows before structure normalization and 3,319 accepted
binary parents (`Y=0`: 454; `Y=1`: 2,865). The `record_supported_v2` scaffold split is:

| Split | Parents | Y=0 | Y=1 | Multi-record | Singleton |
|---|---:|---:|---:|---:|---:|
| train | 2,657 | 364 | 2,293 | 1,637 | 1,020 |
| valid | 331 | 45 | 286 | 331 | 0 |
| test | 331 | 45 | 286 | 331 | 0 |

Identity and scaffold overlap are zero. The status remains `candidate_pending_qa`; the deterministic QA
artifact contains 360 rows, 20 per organ system × label stratum, and has not been manually approved.

The scaffold benchmark view removes held-out parents from the direct gold source only. It removes 710
direct evidence rows and has zero held-out direct-source parent overlap. Non-direct mechanism evidence is
retained by design; query-time retrieval must use `parent_disjoint`.

## Commands

```bash
python -m tools.chembl_tool.tasks.clintox.starling_raw import-raw
python -m tools.chembl_tool.tasks.clintox.starling_raw build-library --workers 8
python -m tools.chembl_tool.tasks.clintox.starling_raw build-benchmark
python -m tools.chembl_tool.tasks.clintox.starling_raw build-heldout-view --workers 8

python -m tools.chembl_tool.tasks.clintox.starling_raw_artifact_store package
python -m tools.chembl_tool.tasks.clintox.starling_raw_artifact_store verify-tracked
python -m tools.chembl_tool.tasks.clintox.starling_raw_artifact_store restore --force
python -m tools.chembl_tool.tasks.clintox.starling_raw_artifact_store verify-local
```

`resume-library` can reuse completed `01_cleaned` and `02_organized` Parquets if a later catalog/index
build is interrupted.
