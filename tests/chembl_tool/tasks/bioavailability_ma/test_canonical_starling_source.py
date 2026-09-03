from pathlib import Path

import pandas as pd
import pytest

from tools.chembl_tool.tasks.bioavailability_ma.build_canonical_starling_source import (
    apply_paper_direct_contract,
    build_canonical_frames,
    deduplicate_paper_direct_claims,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.canonical_source import (
    LOCAL_PARTITION_AMBIGUOUS,
    LOCAL_PARTITION_DIRECT,
    LOCAL_PARTITION_NON_BIOAVAILABILITY,
    LOCAL_PARTITION_RELATIVE,
    classify_local_record,
    direct_measurement_fields,
    nondirect_measurement_fields,
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
                "support_text": "Oral bioavailability was reported as 45%.",
            }
        )[0]
        == LOCAL_PARTITION_AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("value", "measurement", "unit", "numeric", "status"),
    [
        ("95%", "95", "%", 95.0, "explicit_atomic_scalar_unit"),
        ("2.5-fold", "2.5", "fold", 2.5, "explicit_atomic_scalar_unit"),
        ("ratio of 1.4", "1.4", "ratio", 1.4, "explicit_atomic_scalar_unit"),
        ("2.5-fold higher", "2.5-fold higher", "", None, "directional_or_comparative"),
        ("94% versus 71%", "94% versus 71%", "", None, "directional_or_comparative"),
        ("-50.9%", "-50.9%", "", None, "signed_value_not_atomic"),
        ("6,47-fold higher", "6,47-fold higher", "", None, "directional_or_comparative"),
        ("similar", "similar", "", None, "no_explicit_unit"),
    ],
)
def test_nondirect_measurement_extraction_is_explicit_and_conservative(
    value, measurement, unit, numeric, status
):
    fields = nondirect_measurement_fields(value)
    assert fields == {
        "measurement_text": measurement,
        "numeric_value": numeric,
        "value_units": unit,
        "measurement_unit_extraction_status": status,
    }
    assert (
        classify_local_record(
            {
                "exposure_measure": "bioavailability",
                "support_text": "Relative bioavailability of test versus reference was 95%.",
            }
        )[0]
        == LOCAL_PARTITION_RELATIVE
    )


@pytest.mark.parametrize(
    ("value", "measurement", "unit", "numeric", "status"),
    [
        ("50.6%", "50.6", "%", 50.6, "explicit_atomic_scalar_unit"),
        ("0.61 fraction", "0.61", "fraction", 0.61, "explicit_atomic_scalar_unit"),
        ("0.61", "0.61", "", None, "no_explicit_unit"),
        ("0.0246", "0.0246", "", None, "no_explicit_unit"),
        ("not high", "not high", "", None, "no_explicit_unit"),
    ],
)
def test_direct_measurement_extraction_requires_explicit_unit(
    value, measurement, unit, numeric, status
):
    assert direct_measurement_fields(value) == {
        "measurement_text": measurement,
        "numeric_value": numeric,
        "value_units": unit,
        "measurement_unit_extraction_status": status,
    }


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
    assert result["stats"]["n_hf_nondirect_rows"] == 1
    assert result["stats"]["hf_snapshot_reconciliation"]["reconciles"] is True
    assert result["stats"]["n_direct_source_rows_before_dedup"] == 2
    assert result["stats"]["n_canonical_direct_claims"] == 1
    assert result["stats"]["local_partition_reconciliation"]["reconciles"] is True
    assert LOCAL_PARTITION_DIRECT not in set(result["residual_records"]["partition"])
    claim = result["direct_claims"].iloc[0]
    assert list(claim["source_origins"]) == ["hf", "local"]
    assert claim["n_source_records"] == 2
    nondirect = result["hf_nondirect_records"].iloc[0]
    assert nondirect["source_index"] == 1
    assert nondirect["endpoint_name"] == "oral_bioavailability"
    assert nondirect["value_units"] == "%"


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


def test_paper_dedup_prefers_hf_for_the_same_supported_claim():
    shared = {
        "parent_identity_key": "MCGSCOLBFJQGHM-SCZZXKLOSA-N",
        "pmid": "10453964",
        "value_percent": 83.0,
        "value_lower_percent": 83.0,
        "value_upper_percent": 83.0,
        "support_text": "The absolute bioavailability of oral abacavir was 83% in HIV-infected patients.",
        "species_or_population": "HIV-infected patients",
        "dose": "300 mg",
        "oral_exposure_mode": "oral",
        "qualifying_conditions": "",
        "comparator": "intravenous dose",
        "extra_details": "",
    }
    rows = [
        {**shared, "source_origin": "local", "source_record_id": "local:1:ext_1"},
        {**shared, "source_origin": "hf", "source_record_id": "hf:1"},
    ]

    retained, audit, stats = deduplicate_paper_direct_claims(rows)

    assert retained == {"hf:1"}
    assert audit[0]["discarded_source_record_id"] == "local:1:ext_1"
    assert stats["cross_source_duplicates"] == 1


def test_paper_dedup_keeps_context_conflicts_and_collapses_within_source():
    common = {
        "parent_identity_key": "LINOMUASTDIRTM-QGRHZQQGSA-N",
        "pmid": "29774371",
        "value_percent": 52.7,
        "value_lower_percent": 52.7,
        "value_upper_percent": 52.7,
        "dose": "",
        "oral_exposure_mode": "oral",
        "qualifying_conditions": "",
        "comparator": "intravenous",
        "extra_details": "",
    }
    rows = [
        {
            **common,
            "source_origin": "hf",
            "source_record_id": "hf:human",
            "support_text": "Human oral bioavailability of deoxynivalenol was 52.7%.",
            "species_or_population": "human subjects",
        },
        {
            **common,
            "source_origin": "local",
            "source_record_id": "local:pig:ext_1",
            "support_text": "Oral bioavailability of deoxynivalenol was 52.7% in piglets.",
            "species_or_population": "piglets",
        },
        {
            **common,
            "source_origin": "hf",
            "source_record_id": "hf:human-copy",
            "support_text": "Human oral bioavailability of deoxynivalenol was 52.7%.",
            "species_or_population": "human subjects",
        },
    ]

    retained, audit, stats = deduplicate_paper_direct_claims(rows)

    assert retained == {"hf:human", "local:pig:ext_1"}
    assert audit[0]["discarded_source_record_id"] == "hf:human-copy"
    assert stats["within_source_duplicates"] == 1


def test_paper_direct_contract_removes_duplicate_and_marks_local_residual(tmp_path):
    shared = {
        "parent_identity_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "pmid": "10",
        "value_percent": 50.0,
        "value_lower_percent": 50.0,
        "value_upper_percent": 50.0,
        "support_text": "Absolute oral bioavailability was 50% in human volunteers.",
        "species_or_population": "human volunteers",
        "dose": "10 mg",
        "oral_exposure_mode": "solution",
        "qualifying_conditions": "",
        "comparator": "intravenous",
        "extra_details": "",
    }
    source_path = tmp_path / "direct_source_rows.parquet"
    pd.DataFrame(
        [
            {**shared, "source_origin": "hf", "source_record_id": "hf:7"},
            {**shared, "source_origin": "local", "source_record_id": "local:2:ext_3"},
        ]
    ).to_parquet(source_path, index=False)
    records = [
        {
            "canonical_record_id": "hf-record",
            "source_id": "hf_bioavailability",
            "source_record_id": "7",
            "source_row_number": 8,
            "group_id": "Observed.direct_oral_bioavailability",
        },
        {
            "canonical_record_id": "local-direct-record",
            "source_id": "oral_exposure",
            "source_record_id": "ext_3",
            "source_row_number": 3,
            "group_id": "Observed.direct_oral_bioavailability",
        },
        {
            "canonical_record_id": "local-residual-record",
            "source_id": "oral_exposure",
            "source_record_id": "ext_4",
            "source_row_number": 4,
            "group_id": "Observed.direct_oral_bioavailability",
        },
    ]

    prepared, audit, stats = apply_paper_direct_contract(
        records, direct_source_rows_path=source_path
    )

    assert [row["canonical_record_id"] for row in prepared] == [
        "hf-record",
        "local-residual-record",
    ]
    assert prepared[1]["canonical_paper_direct_scope"] == "residual"
    assert prepared[0]["group_id"] == "Observed.direct_oral_bioavailability"
    assert prepared[1]["group_id"] == "Observed.oral_auc_cmax_exposure"
    assert audit[0]["discarded_canonical_record_id"] == "local-direct-record"
    assert stats["input_direct_source_rows"] == 2
    assert stats["dedup_input_mapped_direct_source_rows"] == 2
    assert stats["mapped_direct_source_rows"] == 2
