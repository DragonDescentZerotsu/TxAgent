from tools.chembl_tool.common.evidence_compatibility import (
    COMPATIBILITY_INPUT_VERSION,
    compatibility_inputs_from_row,
)


def test_compatibility_inputs_are_descriptive_and_source_backed():
    row = {
        "minimal_evidence": {
            "contract_version": "minimal_evidence.v1",
            "source": {"name": "Starling/test", "record_id": "r1"},
            "molecule": {"id": "m1", "canonical_smiles": "CC", "names": []},
            "group": {"id": "Observed.direct", "tier": "Observed", "endpoint_group": "direct"},
            "endpoint": {
                "name": "Oral bioavailability",
                "measurement": {"relation": "=", "value": 42, "unit": "%"},
            },
            "text": {"evidence": "measured F", "context": "rat oral"},
            "annotations": {
                "evidence_role": "direct_outcome",
                "scope": {"species": ["rat"], "route": "oral"},
                "transferability": "not_assessed",
                "uncertainty": ["mixed formulation"],
            },
            "quality": {"confidence": 0.8},
            "provenance": {"assay_id": "a1", "source_record_count": 3},
            "examples": [],
        },
        "source_numeric_record_count": 2,
        "source_pmids": ["1", "2"],
    }
    result = compatibility_inputs_from_row(row)
    assert result["contract_version"] == COMPATIBILITY_INPUT_VERSION
    assert result["measurement"]["raw_comparability_key"] == "oral bioavailability|%"
    assert result["support"] == {
        "source_record_count": 3,
        "numeric_record_count": 2,
        "independent_pmid_count": 2,
        "has_multi_record_support": True,
        "has_independent_pmid_support": True,
    }
    assert all(result["readiness"].values())
    assert "score" not in result
    assert "label" not in result


def test_missing_metadata_remains_explicitly_not_ready():
    row = {
        "minimal_evidence": {
            "contract_version": "minimal_evidence.v1",
            "source": {"name": "test", "record_id": ""},
            "molecule": {"id": "m", "canonical_smiles": "CC", "names": []},
            "group": {"id": "g", "tier": "", "endpoint_group": ""},
            "endpoint": {"name": "", "measurement": {"relation": "", "value": "", "unit": ""}},
            "text": {"evidence": "background", "context": ""},
            "annotations": {
                "evidence_role": "unspecified",
                "scope": {},
                "transferability": "not_assessed",
                "uncertainty": [],
            },
            "quality": {"confidence": ""},
            "provenance": {"assay_id": "", "source_record_count": ""},
            "examples": [],
        }
    }
    result = compatibility_inputs_from_row(row)
    assert result["readiness"] == {
        "endpoint_match": False,
        "context_compatibility": False,
        "measurement_available": False,
        "measurement_consistency": False,
        "record_agreement": False,
        "evidence_scope": False,
    }
