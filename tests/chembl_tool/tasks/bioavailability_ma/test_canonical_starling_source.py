from pathlib import Path

import pandas as pd

from tools.chembl_tool.tasks.bioavailability_ma.build_canonical_starling_source import (
    build_canonical_frames,
)
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    LOCAL_PARTITION_AMBIGUOUS,
    LOCAL_PARTITION_DIRECT,
    LOCAL_PARTITION_NON_BIOAVAILABILITY,
    LOCAL_PARTITION_RELATIVE,
    classify_local_record,
)


def test_local_partition_requires_an_absolute_anchor():
    assert classify_local_record({"exposure_measure": "AUC"})[0] == LOCAL_PARTITION_NON_BIOAVAILABILITY
    assert (
        classify_local_record(
            {
                "exposure_measure": "bioavailability",
                "support_text": "Absolute oral bioavailability was 45%.",
            }
        )[0]
        == LOCAL_PARTITION_DIRECT
    )
    assert (
        classify_local_record(
            {
                "exposure_measure": "bioavailability",
                "support_text": "Relative bioavailability of test versus reference was 95%.",
            }
        )[0]
        == LOCAL_PARTITION_RELATIVE
    )
    assert (
        classify_local_record(
            {
                "exposure_measure": "bioavailability",
                "support_text": "Oral bioavailability was reported as 45%.",
            }
        )[0]
        == LOCAL_PARTITION_AMBIGUOUS
    )


def test_build_canonical_frames_partitions_and_cross_source_deduplicates():
    hf = pd.DataFrame(
        [
            {
                "source_index": 0,
                "pmid": "10",
                "support_text": "Absolute oral bioavailability was 50%.",
                "molecule_name": "ethanol",
                "oral_bioavailability_value": "50%",
                "bioavailability_report_type": "absolute",
                "species_or_population": "healthy human volunteers",
                "dose": "10 mg",
                "oral_exposure_mode": "solution",
                "qualifying_conditions": None,
                "comparator": "vs IV",
                "extra_details": None,
                "smiles": "CCO",
            },
            {
                "source_index": 1,
                "pmid": "11",
                "support_text": "Relative comparison was 95%.",
                "molecule_name": "propanol",
                "oral_bioavailability_value": "95%",
                "bioavailability_report_type": "relative_comparison",
                "species_or_population": "healthy human volunteers",
                "dose": "10 mg",
                "oral_exposure_mode": "solution",
                "qualifying_conditions": None,
                "comparator": "test/reference",
                "extra_details": None,
                "smiles": "CCCO",
            },
        ]
    )
    local = pd.DataFrame(
        [
            {
                "source_index": 0,
                "pmid": "10",
                "extraction_id": "ext_1",
                "global_identifier": "ethanol",
                "support_text": "Absolute bioavailability after oral dosing was 50.5%.",
                "exposure_measure": "bioavailability",
                "parameter_value": 50.5,
                "parameter_units": "%",
                "oral_dose": "10 mg",
                "study_context": "healthy human volunteers",
                "comparator_exposure": "intravenous reference",
                "qualifying_conditions": None,
                "extra_details": None,
                "smiles": "CCO",
            },
            {
                "source_index": 1,
                "pmid": "12",
                "extraction_id": "ext_2",
                "global_identifier": "propanol",
                "support_text": "Relative bioavailability was 95%.",
                "exposure_measure": "bioavailability",
                "parameter_value": 95.0,
                "parameter_units": "%",
                "oral_dose": "10 mg",
                "study_context": "healthy human volunteers",
                "comparator_exposure": "test versus reference",
                "qualifying_conditions": None,
                "extra_details": None,
                "smiles": "CCCO",
            },
            {
                "source_index": 2,
                "pmid": "13",
                "extraction_id": "ext_3",
                "global_identifier": "butanol",
                "support_text": "Oral AUC was measured.",
                "exposure_measure": "AUC",
                "parameter_value": 10.0,
                "parameter_units": "ng h/mL",
                "oral_dose": "10 mg",
                "study_context": "healthy human volunteers",
                "comparator_exposure": None,
                "qualifying_conditions": None,
                "extra_details": None,
                "smiles": "CCCCO",
            },
        ]
    )

    result = build_canonical_frames(
        hf,
        local,
        hf_revision="test-revision",
        local_source_path=Path("local.parquet"),
    )

    assert result["stats"]["n_local_absolute_rows_removed_from_residual"] == 1
    assert result["stats"]["n_local_absolute_rows_with_valid_structure"] == 1
    assert result["stats"]["n_local_residual_rows"] == 2
    assert result["stats"]["n_cross_source_matches"] == 1
    assert result["stats"]["n_direct_source_rows_before_dedup"] == 2
    assert result["stats"]["n_canonical_direct_claims"] == 1
    assert result["stats"]["local_partition_reconciliation"]["reconciles"] is True
    assert LOCAL_PARTITION_DIRECT not in set(result["residual_records"]["partition"])
    claim = result["direct_claims"].iloc[0]
    assert list(claim["source_origins"]) == ["hf", "local"]
    assert claim["n_source_records"] == 2


def test_cross_source_dedup_does_not_collapse_threshold_crossing_interval():
    hf = pd.DataFrame(
        [
            {
                "source_index": 0,
                "pmid": "20",
                "support_text": "Absolute oral bioavailability was 21% versus IV.",
                "oral_bioavailability_value": "10-30%",
                "bioavailability_report_type": "absolute",
                "species_or_population": "human volunteers",
                "smiles": "CCO",
            }
        ]
    )
    local = pd.DataFrame(
        [
            {
                "source_index": 0,
                "pmid": "20",
                "extraction_id": "ext_crossing",
                "support_text": "Absolute oral bioavailability was 21% versus IV.",
                "exposure_measure": "bioavailability",
                "parameter_value": 21.0,
                "parameter_units": "%",
                "study_context": "human volunteers",
                "comparator_exposure": "intravenous reference",
                "smiles": "CCO",
            }
        ]
    )

    result = build_canonical_frames(
        hf,
        local,
        hf_revision="test-revision",
        local_source_path=Path("local.parquet"),
    )

    assert result["stats"]["n_cross_source_matches"] == 0
    assert result["stats"]["n_canonical_direct_claims"] == 2
