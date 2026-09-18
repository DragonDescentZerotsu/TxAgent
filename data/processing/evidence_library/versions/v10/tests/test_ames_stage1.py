from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    normalize_endpoint_name,
)
from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import (
    parse_args,
)
from data.processing.evidence_library.versions.v10.tasks.ames import starling_policy
from data.processing.evidence_library.versions.v10.tasks.ames.starling_endpoint_normalization import (
    canonical_endpoint_name,
    endpoint_decision,
    endpoint_orthography,
)
from data.processing.paths import evidence_library_root


def test_ames_v10_is_a_registered_normalized_release() -> None:
    policy = starling_policy.POLICY
    assert Path(policy.default_out_dir) == evidence_library_root("ames", "v10")
    args = parse_args(policy, [], default_through_stage="clean", core_only=True)
    assert args.through_stage == "clean"


def test_raw_quality_audit_joins_sparse_selected_rows_by_uid(tmp_path) -> None:
    source_id = "fixed_mutation"
    source_dir = tmp_path / starling_policy.SOURCE_SPECS[source_id]["directory"]
    source_dir.mkdir()
    rows = [
        {
            "source_row_uid": f"sr_{index:032x}",
            "pmid": str(index),
            "extraction_id": f"extraction-{index}",
            "support_text": f"support {index}",
            "confidence": 0.9,
            "response_value": str(index),
            "response_unit": "count",
        }
        for index in (1, 2)
    ]
    pd.DataFrame(rows).to_parquet(source_dir / "extractions.parquet", index=False)
    selected = [
        {
            "source_row_uid": rows[1]["source_row_uid"],
            "source_record_id": rows[1]["extraction_id"],
        }
    ]

    raw, _, _, _ = starling_policy._raw_audit_inputs(
        tmp_path,
        source_id,
        selected,
        {"schema_spec": []},
    )

    assert raw["extraction_id"].tolist() == ["extraction-2"]


def test_endpoint_mapping_aliases_exclusions_and_unknowns() -> None:
    assert canonical_endpoint_name("fixed_mutation", "microunucleus_assay") == (
        "micronucleus_assay"
    )
    assert canonical_endpoint_name("premutagenic_damage", None) == ""
    assert endpoint_decision("premutagenic_damage", None)["reason"] == (
        "missing_assay_family"
    )
    with pytest.raises(ValueError, match="unmapped Ames endpoint"):
        canonical_endpoint_name("fixed_mutation", "new_unreviewed_family")


def test_endpoint_orthography_preserves_review_provenance() -> None:
    alias = endpoint_orthography("fixed_mutation", "microunucleus_assay")
    assert alias.spacing_and_spelling_endpoint == "micronucleus_assay"
    assert alias.status == "reviewed_normalization"
    assert alias.reason == "reviewed_alias"
    assert alias.spacing_and_spelling_version == "ames_endpoint_normalization.v1"
    excluded = endpoint_orthography("premutagenic_damage", None)
    assert excluded.spacing_and_spelling_endpoint == ""
    assert excluded.status == "reviewed_exclusion"
    assert excluded.reason == "missing_assay_family"


@pytest.mark.parametrize(
    ("source_id", "raw", "status", "canonical", "reason"),
    [
        (
            "mutagenicity_mechanism",
            "consistent_ros_or_redox_activity_placeholder_discarded_antioxidant_assay",
            "alias",
            "antioxidant_or_thiol_assay",
            "reviewed_alias",
        ),
        (
            "mutagenicity_outcomes",
            "microin_vivo_somatic_gene_mutation_placeholder",
            "alias",
            "multiple_mutation_assays",
            "reviewed_alias",
        ),
        (
            "mutagenicity_outcomes",
            "specified",
            "alias",
            "unspecified_mutagenicity",
            "reviewed_alias",
        ),
        (
            "mutagenicity_outcomes",
            "uncategorized",
            "alias",
            "non_mammalian_eukaryote_mutation",
            "reviewed_alias",
        ),
        (
            "mutagenicity_outcomes",
            "human_observational_gene_mutation",
            "alias",
            "in_vivo_somatic_gene_mutation",
            "reviewed_alias",
        ),
        (
            "mutagenicity_mechanism",
            "reactively_trapped_metabolite_detection",
            "alias",
            "metabolite_formation_or_reaction_phenotyping",
            "reviewed_alias",
        ),
        (
            "mutagenicity_mechanism",
            "reactive_metabolite_formation",
            "alias",
            "metabolic_enzyme_activity_or_inactivation",
            "reviewed_alias",
        ),
        (
            "mutagenicity_mechanism",
            "mechanical_electrophoretic_evidence",
            "alias",
            "metabolic_enzyme_activity_or_inactivation",
            "reviewed_alias",
        ),
        (
            "mutagenicity_mechanism",
            "oxididative_damage_assay",
            "alias",
            "ros_or_redox_assay",
            "reviewed_alias",
        ),
        (
            "mutagenicity_mechanism",
            "spindle_or_microtubule_assay",
            "excluded",
            "",
            "no_direct_tubulin_or_microtubule_measurement",
        ),
        (
            "mutagenicity_mechanism",
            "oxidatic_damage_assay",
            "excluded",
            "",
            "insufficient_assay_identity",
        ),
        (
            "premutagenic_damage",
            "other_direct_premutagenic_damage",
            "excluded",
            "",
            "mixed_scope_no_safe_global_mapping",
        ),
        (
            "premutagenic_damage",
            "strand_break_or_other_direct_damage_assay",
            "alias",
            "other_direct_damage_assay",
            "reviewed_alias",
        ),
        (
            "premutagenic_damage",
            "amp:true_placeholder",
            "excluded",
            "",
            "analytical_calibration_only",
        ),
        (
            "mutagenicity_outcomes",
            "ano_nonexistent_placeholder",
            "excluded",
            "",
            "no_mutagenicity_endpoint",
        ),
        (
            "fixed_mutation",
            "heritable_lethal_or_mutation_proxy",
            "excluded",
            "",
            "unsupported_assay_family",
        ),
        ("mutagenicity_outcomes", "unknown", "excluded", "", "ambiguous_assay_family"),
    ],
)
def test_independent_audit_endpoint_decisions(
    source_id: str, raw: str, status: str, canonical: str, reason: str
) -> None:
    decision = endpoint_decision(source_id, raw)
    assert (decision["status"], decision["canonical_endpoint"], decision["reason"]) == (
        status,
        canonical,
        reason,
    )


def test_frozen_source_endpoint_inventory_is_fully_decided() -> None:
    data_root = Path(starling_policy.DEFAULT_DATA_DIR)
    row_effects: Counter[str] = Counter()
    for source_id, profile in zip(
        starling_policy.SOURCE_COLUMNS,
        starling_policy.source_profiles(data_root),
    ):
        values = pq.read_table(profile.source_path, columns=[profile.endpoint_field])[
            profile.endpoint_field
        ].to_pylist()
        frequencies = Counter(normalize_endpoint_name(value) or "" for value in values)
        endpoints = sorted(frequencies)
        inventory = starling_policy.endpoint_inventory(
            source_id, endpoints, strict=True
        )
        assert inventory["mapping_coverage"] == 1.0
        for endpoint, count in frequencies.items():
            row_effects[endpoint_decision(source_id, endpoint)["status"]] += count
    assert row_effects == {"identity": 1_502_805, "alias": 325, "excluded": 7_317}


def test_endpoint_exclusions_are_dropped_before_routing(monkeypatch) -> None:
    records = [
        {
            "task_id": "ames",
            "source_id": "fixed_mutation",
            "cleaned_record_id": "keep",
            "source_row_uid": "source-keep",
            "source_row_number": 1,
            "source_record_id": "keep",
            "endpoint_name": "microunucleus_assay",
        },
        {
            "task_id": "ames",
            "source_id": "fixed_mutation",
            "cleaned_record_id": "drop",
            "source_row_uid": "source-drop",
            "source_row_number": 2,
            "source_record_id": "drop",
            "endpoint_name": None,
        },
    ]
    monkeypatch.setattr(
        starling_policy, "_audit_source_quality", lambda records, data_dir: ([], {}, ())
    )
    args = type("Args", (), {"starling_data_dir": "/unused"})()
    result = starling_policy.clean_source_values(records, args)
    assert [row["cleaned_record_id"] for row in result.records] == ["keep"]
    assert result.records[0]["endpoint_name"] == "microunucleus_assay"
    assert result.records[0]["canonical_endpoint_name"] == "micronucleus_assay"
    orthography = endpoint_orthography(
        result.records[0]["source_id"], result.records[0]["endpoint_name"]
    )
    assert (orthography.status, orthography.reason) == (
        "reviewed_normalization",
        "reviewed_alias",
    )
    assert result.manifest["n_repaired_records"] == 1
    assert result.manifest["n_dropped_records"] == 1
    assert result.audit_rows[0] == {
        "task_id": "ames",
        "source_id": "fixed_mutation",
        "cleaned_record_id": "keep",
        "source_row_uid": "source-keep",
        "source_row_number": 1,
        "source_record_id": "keep",
        "audit_type": "endpoint_alias",
        "field": "endpoint_name",
        "before": "microunucleus_assay",
        "after": "micronucleus_assay",
        "action": "normalized",
    }
    assert result.audit_rows[-1]["after"] == "dropped"
