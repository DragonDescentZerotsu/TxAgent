import json

import pandas as pd
import pytest

from tools.chembl_tool.common.starling.normalized_evidence import (
    NormalizedSourceProfile,
    normalize_measurement_and_unit,
    normalize_source_rows,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    parse_point_measurement,
)
from tools.chembl_tool.tasks.bioavailability_ma import (
    build_normalized_starling_evidence_library as normalized_builder,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_normalized_starling_evidence_library import (
    load_direct_hf_rows,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_policy import (
    endpoint_specific_standardization_of_unit,
    family_assignment,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    SPACING_AND_SPELLING_VERSION,
    spacing_and_spelling_decision,
    validate_endpoint_inventory,
)


def _standardize(endpoint, measurement, unit):
    return endpoint_specific_standardization_of_unit(
        endpoint, normalize_measurement_and_unit(measurement, unit)
    )


def test_auc_standardization_supports_mass_and_molar_dimensions():
    mass = _standardize("auc0_inf", "5", "µg/mL·h")
    assert mass.canonical_measurement == "5000"
    assert mass.canonical_unit == "ng·h/mL"

    molar = _standardize("auc", "845", "nmol L-1 h")
    assert float(molar.canonical_measurement) == pytest.approx(0.845)
    assert molar.canonical_unit == "µmol·h/L"

    reciprocal = _standardize("auc", "14.5", "h nmol^-1 mL^-1")
    assert reciprocal.canonical_measurement == "14.5"
    assert reciprocal.canonical_unit == "h/mL·nmol"
    assert reciprocal.status == "incompatible_endpoint_unit"

    tissue_normalized = _standardize("auc0_t", "11727", "ng h g^-1")
    assert tissue_normalized.canonical_measurement == "11727"
    assert tissue_normalized.canonical_unit == "h·ng/g"
    assert tissue_normalized.status != "incompatible_endpoint_unit"


@pytest.mark.parametrize(
    ("endpoint", "measurement", "unit", "expected_measurement", "expected_unit"),
    [
        ("bioavailability", "0.57", "fraction", "57", "%"),
        ("intestinal_permeability", "4", "nm/s", "0.0000004", "cm/s"),
        ("metabolic_half_life", "120", "min", "2", "h"),
        ("cmax", "2", "µg/mL", "2000", "ng/mL"),
        ("intrinsic_clearance", "8.3", "µL/min·mg", "8.3", "µL/min/mg protein"),
    ],
)
def test_reviewed_endpoint_unit_conversions(
    endpoint, measurement, unit, expected_measurement, expected_unit
):
    result = _standardize(endpoint, measurement, unit)
    assert result.canonical_measurement == expected_measurement
    assert result.canonical_unit == expected_unit
    assert result.status == "endpoint_standardized"


@pytest.mark.parametrize(
    ("measurement", "expected_measurement"),
    [
        ("~0.5", "~50"),
        ("≈0.5", "≈50"),
        ("about 0.5", "about 50"),
        ("~0.5 ± 0.1", "~50 ± 10"),
    ],
)
def test_fraction_to_percent_preserves_approximation_metadata(
    measurement, expected_measurement
):
    result = _standardize("bioavailability", measurement, "fraction")

    assert result.canonical_measurement == expected_measurement
    assert result.canonical_unit == "%"
    assert result.status == "endpoint_standardized"
    assert parse_point_measurement(result.canonical_measurement).approximate is True


def test_unusual_dimension_preserves_mechanical_pair_status_and_scalar():
    profile = NormalizedSourceProfile(
        source_id="oral_exposure",
        source_name="test",
        endpoint_constant="AUC",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
    )
    result = normalize_source_rows(
        [{"smiles": "CCO", "value": "5", "unit": "%"}],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        family_resolver=family_assignment,
    )
    record = result.records[0]
    assert record["canonical_endpoint"] == "auc"
    assert record["canonical_measurement"] == "5"
    assert record["canonical_unit"] == "%"
    assert record["finite_scalar_value"] == 5.0
    assert record["measurement_unit_status"] == "cleaned_pair"


def test_orthographic_corrections_are_explicit_case_preserving_and_audited():
    unchanged = spacing_and_spelling_decision("oral_exposure", "AUC0-T")
    assert unchanged.endpoint_name == "AUC0-T"
    assert unchanged.spacing_and_spelling_endpoint == "AUC0-T"
    assert unchanged.spacing_and_spelling_version == SPACING_AND_SPELLING_VERSION

    corrected = spacing_and_spelling_decision("fa", "solubidity")
    assert corrected.endpoint_name == "solubidity"
    assert corrected.spacing_and_spelling_endpoint == "solubility"
    assert corrected.status == "reviewed_correction"
    assert corrected.reason == "explicit_reviewed_spacing_or_spelling_error"
    assert (
        spacing_and_spelling_decision("fa", "SOLUBIDITY").spacing_and_spelling_endpoint
        == "SOLUBILITY"
    )


@pytest.mark.parametrize(
    ("source", "left", "right"),
    [
        ("fa", "intest inal_absorption", "intestinal_absorption"),
        ("fg", "efflux_or_secretary_transport", "efflux_or_secretory_transport"),
        ("fa", "caco2_mdck_pampa_perability", "caco2_mdck_pampa_permeability"),
        ("fh", "hepatic_first_pass_m etabolism", "hepatic_first_pass_metabolism"),
    ],
)
def test_reviewed_variants_converge(source, left, right):
    assert (
        spacing_and_spelling_decision(source, left).spacing_and_spelling_endpoint
        == spacing_and_spelling_decision(source, right).spacing_and_spelling_endpoint
    )


def test_scientifically_related_endpoints_remain_distinct():
    profile = NormalizedSourceProfile(
        source_id="oral_exposure",
        source_name="test",
        endpoint_field="endpoint",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
        record_id_field="id",
    )
    result = normalize_source_rows(
        [
            {
                "id": "1",
                "smiles": "CCO",
                "endpoint": "absolute_bioavailability",
                "value": "50",
                "unit": "%",
            },
            {
                "id": "2",
                "smiles": "CCN",
                "endpoint": "relative_bioavailability",
                "value": "50",
                "unit": "%",
            },
        ],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        family_resolver=family_assignment,
    )
    assert {row["canonical_endpoint"] for row in result.records} == {
        "absolute_bioavailability",
        "relative_bioavailability",
    }


def test_report_type_and_unit_do_not_rewrite_endpoint_identity():
    profile = NormalizedSourceProfile(
        source_id="direct_hf",
        source_name="test",
        endpoint_constant="oral_bioavailability",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
        record_id_field="id",
        context_fields=("bioavailability_report_type",),
    )
    result = normalize_source_rows(
        [
            {
                "id": "1",
                "smiles": "CCO",
                "value": "57",
                "unit": "%",
                "bioavailability_report_type": "absolute",
            },
            {
                "id": "2",
                "smiles": "CCN",
                "value": "140",
                "unit": "%",
                "bioavailability_report_type": "relative_comparison",
            },
            {
                "id": "3",
                "smiles": "CCC",
                "value": "4",
                "unit": "nm/s",
                "bioavailability_report_type": "apparent",
            },
        ],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        family_resolver=family_assignment,
    )
    assert {row["canonical_endpoint"] for row in result.records} == {
        "oral_bioavailability"
    }
    assert result.records[2]["measurement_unit_status"] == "cleaned_pair"


def test_permeability_unit_does_not_rewrite_broad_fa_endpoint():
    profile = NormalizedSourceProfile(
        source_id="fa",
        source_name="test",
        endpoint_constant="absorption",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
    )
    result = normalize_source_rows(
        [{"smiles": "CCO", "value": "4", "unit": "nm/s"}],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        family_resolver=family_assignment,
    )
    record = result.records[0]
    assert record["canonical_endpoint"] == "absorption"
    assert record["canonical_unit"] == "nm/s"
    assert record["measurement_unit_status"] == "cleaned_pair"


def test_family_mapping_remains_separate_and_stable():
    assert family_assignment("direct_hf", "oral_bioavailability").group_id == (
        "Observed.direct_oral_bioavailability"
    )
    assert family_assignment("oral_exposure", "AUC").group_id == (
        "Observed.oral_auc_cmax_exposure"
    )
    assert family_assignment("fa", "solubility").group_id.startswith("Fa.")
    assert family_assignment("fg", "intestinal_metabolism").group_id.startswith("Fg.")
    assert family_assignment("fh", "intrinsic_clearance").group_id.startswith("Fh.")


def test_frozen_inventory_audits_orthography_without_semantic_registry():
    inventory = validate_endpoint_inventory("direct_hf", ["oral_bioavailability"])
    assert inventory["coverage"] == 1.0
    assert inventory["n_reviewed_corrections"] == 0
    assert (
        inventory["endpoints"][0]["spacing_and_spelling_version"]
        == SPACING_AND_SPELLING_VERSION
    )
    with pytest.raises(ValueError, match="endpoint inventory drift"):
        validate_endpoint_inventory("direct_hf", ["oral_bioavailability", "new_endpoint"])


def test_direct_hf_loader_reconstructs_kept_and_dropped_rows(tmp_path):
    kept = tmp_path / "kept.jsonl"
    dropped = tmp_path / "dropped.jsonl"
    kept.write_text(
        json.dumps(
            {
                "source_index": 1,
                "smiles": "CCO",
                "metadata": {"oral_bioavailability_value": "57%", "smiles": "CCO"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dropped.write_text(
        json.dumps(
            {
                "source_index": 2,
                "drop_reason": "unparseable_or_non_numeric_value",
                "raw_row": {
                    "oral_bioavailability_value": "low",
                    "smiles": "CCN",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_direct_hf_rows(kept, dropped)
    assert [row["source_index"] for row in rows] == [1, 2]
    assert rows[1]["_prepared_drop_reason"] == "unparseable_or_non_numeric_value"


def test_versioned_builder_schema_manifest_and_restart(tmp_path):
    data_dir = tmp_path / "sources"
    source_rows = {
        "Oral_AUC-Cmax_Exposure": {
            "global_identifier": "SMILES:1",
            "extraction_id": "q1",
            "exposure_measure": "AUC",
            "parameter_value": 5.0,
            "parameter_units": "µg/mL·h",
        },
        "Fa": {
            "global_identifier": "SMILES:2",
            "extraction_id": "q2",
            "endpoint_category": "solubidity",
            "reported_value": "2",
            "reported_units": "µM",
        },
        "Fg": {
            "global_identifier": "SMILES:3",
            "extraction_id": "q3",
            "gut_wall_process": "intestinal_metabolism",
            "measured_value": "substrate",
        },
        "Fh": {
            "global_identifier": "SMILES:4",
            "extraction_id": "q4",
            "metric_type": "metabolic_half_life",
            "reported_value": "2",
            "reported_units": "h",
        },
    }
    for directory, row in source_rows.items():
        path = data_dir / directory
        path.mkdir(parents=True)
        pd.DataFrame([row]).to_parquet(path / "extractions.parquet", index=False)

    mapping = tmp_path / "mapping.parquet"
    pd.DataFrame(
        [
            {"global_identifier": f"SMILES:{index}", "smiles": smiles}
            for index, smiles in enumerate(("CCO", "CCN", "CCC", "CCCl"), start=1)
        ]
    ).to_parquet(mapping, index=False)
    kept = tmp_path / "kept.jsonl"
    kept.write_text(
        json.dumps(
            {
                "source_index": 0,
                "smiles": "CCBr",
                "metadata": {
                    "oral_bioavailability_value": "40%",
                    "smiles": "CCBr",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dropped = tmp_path / "dropped.jsonl"
    dropped.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    common_args = [
        "--starling-data-dir",
        str(data_dir),
        "--smiles-mapping",
        str(mapping),
        "--allow-unpinned-smiles-mapping",
        "--direct-source-jsonl",
        str(kept),
        "--direct-dropped-jsonl",
        str(dropped),
        "--out-dir",
        str(out_dir),
        "--no-strict-endpoint-inventory",
        "--max-rows-per-source",
        "1",
        "--max-direct-rows",
        "1",
        "--progress-every",
        "0",
    ]
    assert normalized_builder.main(common_args) == 0

    records = pd.read_parquet(out_dir / normalized_builder.RECORDS_FILENAME)
    normalized = pd.read_parquet(
        out_dir / normalized_builder.NORMALIZED_RECORDS_FILENAME
    )
    endpoint_columns = [
        "endpoint_name",
        "spacing_and_spelling_endpoint",
        "canonical_endpoint",
    ]
    assert all(column in records for column in endpoint_columns)
    assert all(
        column in records
        for column in (
            "spacing_and_spelling_status",
            "spacing_and_spelling_reason",
            "spacing_and_spelling_version",
        )
    )
    assert all(
        column in records
        for column in (
            "source_smiles",
            "normalization_validity_status",
            "canonical_bioavailability_report_type",
            "canonical_dose_key",
            "canonical_assay_system",
            "canonical_species",
        )
    )
    assert len(records) == len(normalized) == 5
    removed = {
        "metric_name",
        "comparison_geometry",
        "comparison_status",
        "comparison_stratum",
        "comparison_value",
        "comparison_unit",
        "comparison_domain_valid",
        "comparison_domain_reason",
        "comparison_metric_type",
        "comparison_policy_version",
        "comparison_policy_key",
        "assay_transfer_status",
    }
    assert not removed & set(records)
    assert not removed & set(normalized)

    manifest = json.loads(
        (out_dir / normalized_builder.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["v65_reconciliation"] == {
        "matching_performed": False,
        "status": "not_performed",
    }
    assert manifest["artifact_version"].endswith(".v5")
    assert manifest["normalization_domain_rules_version"].endswith(".v2")
    assert "comparison_policy_registry_version" not in manifest
    assert "expected_frozen_census" not in manifest
    assert not (out_dir / "pair_buckets").exists()

    preserved_paths = [
        out_dir / normalized_builder.RECORDS_FILENAME,
        out_dir / normalized_builder.EVIDENCE_FILENAME,
        out_dir / normalized_builder.INDEX_FILENAME,
        out_dir / normalized_builder.MANIFEST_FILENAME,
        out_dir / normalized_builder.CANONICALIZATION_POLICY_FILENAME,
    ]
    preserved_bytes = {path: path.read_bytes() for path in preserved_paths}
    failed_args = list(common_args)
    failed_args[failed_args.index("--smiles-mapping") + 1] = str(
        tmp_path / "missing-mapping.parquet"
    )
    failed_args.extend(["--through-stage", "clean"])
    with pytest.raises(FileNotFoundError, match="authoritative SMILES mapping"):
        normalized_builder.main(failed_args)
    assert all(path.read_bytes() == preserved_bytes[path] for path in preserved_paths)

    for directory in ("pair_buckets", "endpoint_policies", "analysis"):
        target = out_dir / directory
        target.mkdir()
        (target / "stale.txt").write_text("stale", encoding="utf-8")
    for filename in normalized_builder.RECORD_DEPENDENT_FILES:
        (out_dir / filename).write_text("stale", encoding="utf-8")

    assert normalized_builder.main(
        [
            "--out-dir",
            str(out_dir),
            "--from-stage",
            "normalize",
            "--through-stage",
            "organize",
            "--progress-every",
            "0",
        ]
    ) == 0
    restarted = json.loads(
        (out_dir / normalized_builder.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert restarted["completed_stages"] == ["clean", "normalize", "organize"]
    assert restarted["rebuild_request"] == {
        "from_stage": "normalize",
        "through_stage": "organize",
    }
    assert {
        "analysis",
        "endpoint_policies",
        "molecule_family_evidence.jsonl",
        "neighbor_index.meta.json",
        "neighbor_index.pkl",
        "pair_buckets",
    } <= set(restarted["invalidated_artifacts"])
    assert not (out_dir / normalized_builder.EVIDENCE_FILENAME).exists()
    assert not (out_dir / normalized_builder.INDEX_FILENAME).exists()
    assert not (out_dir / normalized_builder.INDEX_META_FILENAME).exists()
    assert all(
        not (out_dir / directory).exists()
        for directory in ("pair_buckets", "endpoint_policies", "analysis")
    )
    assert not any("-stage-" in path.name for path in out_dir.iterdir())

    frozen_v1 = out_dir / "endpoint_policies" / "v1"
    frozen_v1.mkdir(parents=True)
    (frozen_v1 / "endpoint_policy_registry.json").write_text(
        "stale", encoding="utf-8"
    )
    assert normalized_builder.main([*common_args, "--through-stage", "clean"]) == 0
    clean_only = json.loads(
        (out_dir / normalized_builder.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert clean_only["completed_stages"] == ["clean"]
    assert not (out_dir / normalized_builder.NORMALIZED_RECORDS_FILENAME).exists()
    assert not (out_dir / normalized_builder.RECORDS_FILENAME).exists()
    assert not (out_dir / "endpoint_policies").exists()
    assert "endpoint_policies" in clean_only["invalidated_artifacts"]


@pytest.mark.parametrize(
    ("stage", "removes_record_dependents"),
    [
        ("clean", True),
        ("normalize", True),
        ("organize", True),
        ("index", False),
    ],
)
def test_stage_invalidation_follows_dependency_order(
    tmp_path, stage, removes_record_dependents
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for filenames in normalized_builder.STAGE_OUTPUT_FILENAMES.values():
        for filename in filenames:
            if filename == normalized_builder.MANIFEST_FILENAME:
                continue
            (out_dir / filename).write_text(filename, encoding="utf-8")
    for directory in normalized_builder.RECORD_DEPENDENT_DIRECTORIES:
        target = out_dir / directory
        target.mkdir()
        (target / "stale.txt").write_text("stale", encoding="utf-8")

    normalized_builder._invalidate_downstream_artifacts(out_dir, stage)

    stage_index = normalized_builder.STAGES.index(stage)
    for candidate in normalized_builder.STAGES:
        for filename in normalized_builder.STAGE_OUTPUT_FILENAMES[candidate]:
            if filename == normalized_builder.MANIFEST_FILENAME:
                continue
            expected = (
                normalized_builder.STAGES.index(candidate) <= stage_index
            )
            assert (out_dir / filename).exists() is expected
    assert all(
        (out_dir / directory).exists() is (not removes_record_dependents)
        for directory in normalized_builder.RECORD_DEPENDENT_DIRECTORIES
    )


def test_stage_resume_rejects_stale_upstream_input(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cleaned_path = out_dir / normalized_builder.CLEANED_FILENAME
    normalized_path = out_dir / normalized_builder.NORMALIZED_RECORDS_FILENAME
    pd.DataFrame([{"cleaned_record_id": "clean-1"}]).to_parquet(
        cleaned_path, index=False
    )
    pd.DataFrame(
        [
            {
                "cleaned_record_id": "clean-1",
                "normalized_record_id": "normalized-1",
            }
        ]
    ).to_parquet(normalized_path, index=False)
    manifest = normalized_builder.stage_manifest(
        stage="normalize",
        version=normalized_builder.NORMALIZATION_STAGE_VERSION,
        inputs={"cleaned_records": cleaned_path},
        output=normalized_path,
        row_counts={"normalized_records": 1},
        validations={"test": True},
    )
    (out_dir / "02_normalization.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    pd.DataFrame([{"cleaned_record_id": "clean-2"}]).to_parquet(
        cleaned_path, index=False
    )

    with pytest.raises(ValueError, match="upstream input hash mismatch"):
        normalized_builder._load_verified_stage(out_dir, "normalize")


def test_builder_rejects_reconciliation_until_comparison_contract_exists():
    with pytest.raises(SystemExit):
        normalized_builder._parse_args(["--v65-reconciliation"])


def test_builder_cli_has_no_expansion_stage():
    with pytest.raises(SystemExit):
        normalized_builder._parse_args(
            ["--from-stage", "expand", "--through-stage", "expand"]
        )
