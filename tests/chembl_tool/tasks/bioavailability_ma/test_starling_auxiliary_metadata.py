import json

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    AuxiliaryMetadataAttacher,
)


def _write_mapping(path, *, fa_context="caco_2"):
    sections = {
        "fa": {
            "global_context": {
                "source_columns": ["assay_system"],
                "mapping": {'["Caco-2  cells"]': fa_context},
            },
            "global_species_context": {
                "source_columns": ["biological_context", "assay_system"],
                "mapping": {'["rat intestine","Caco-2  cells"]': "rat"},
            },
        },
        "fg": {
            "global_context": {
                "source_columns": ["assay_system"],
                "mapping": {'["Xenopus assay"]': "oocyte assay"},
            },
            "global_species_context": {
                "source_columns": ["assay_system"],
                "mapping": {
                    '["Xenopus assay"]': "african clawed frog host; human gene"
                },
            },
        },
        "fh": {
            "global_context": {
                "source_columns": ["assay_system"],
                "mapping": {'["recombinant CYP"]': "recombinant enzyme"},
            },
            "global_species_context": {
                "source_columns": ["species", "assay_system"],
                "mapping": {
                    '["human","recombinant CYP"]': "insect host; human gene"
                },
            },
        },
    }
    path.write_text(
        json.dumps(
            {
                "mapping_version": "starling_auxiliary.globally_reconciled.v1",
                "sources": sections,
            }
        ),
        encoding="utf-8",
    )


def test_attachment_uses_cleaned_source_tuples_and_preserves_role_labels(tmp_path):
    path = tmp_path / "mapping.json"
    _write_mapping(path)
    attacher = AuxiliaryMetadataAttacher(path)
    assert attacher.attach(
        {
            "source_id": "fa",
            "assay_system": "Caco-2 cells",
            "biological_context": "rat intestine",
        }
    ) == {
        "global_context": "caco_2",
        "global_species_context": "rat",
        "auxiliary_mapping_status": "mapped",
        "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
    }
    assert attacher.attach(
        {
            "source_id": "fg",
            "assay_system": "Xenopus assay",
        }
    )["global_species_context"] == "african clawed frog host; human gene"


def test_null_is_a_valid_mapping_but_absent_tuple_fails(tmp_path):
    path = tmp_path / "mapping.json"
    _write_mapping(path, fa_context=None)
    attacher = AuxiliaryMetadataAttacher(path)
    attached = attacher.attach(
        {
            "source_id": "fa",
            "assay_system": "Caco-2 cells",
            "biological_context": "rat intestine",
        }
    )
    assert attached["global_context"] is None
    with pytest.raises(ValueError, match="mapping lacks"):
        attacher.attach(
            {
                "source_id": "fa",
                "assay_system": "PAMPA",
                "biological_context": "rat intestine",
            }
        )


def test_source_null_sentinels_join_the_frozen_null_tuple(tmp_path):
    path = tmp_path / "mapping.json"
    _write_mapping(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sources"]["fa"]["global_context"]["mapping"]['[null]'] = None
    payload["sources"]["fa"]["global_species_context"]["mapping"][
        '[null,null]'
    ] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    attached = AuxiliaryMetadataAttacher(path).attach(
        {
            "source_id": "fa",
            "assay_system": "unknown",
            "biological_context": "not specified",
        }
    )
    assert attached["global_context"] is None
    assert attached["global_species_context"] is None
    assert attached["auxiliary_mapping_status"] == "mapped"


def test_direct_and_oral_exposure_are_explicitly_not_applicable(tmp_path):
    path = tmp_path / "mapping.json"
    _write_mapping(path)
    attacher = AuxiliaryMetadataAttacher(path)
    for source_id in ("hf_bioavailability", "oral_exposure"):
        attached = attacher.attach({"source_id": source_id})
        assert attached["auxiliary_mapping_status"] == "not_applicable"
        assert attached["global_context"] is None
        assert attached["global_species_context"] is None


def test_conflicting_cleaned_tuple_keys_are_rejected(tmp_path):
    path = tmp_path / "mapping.json"
    _write_mapping(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sources"]["fa"]["global_context"]["mapping"][
        '["Caco-2 cells"]'
    ] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting cleaned tuple"):
        AuxiliaryMetadataAttacher(path)


def test_manifest_and_coverage_are_deterministic(tmp_path):
    path = tmp_path / "mapping.json"
    _write_mapping(path)
    left = AuxiliaryMetadataAttacher(path)
    right = AuxiliaryMetadataAttacher(path)
    assert left.manifest() == right.manifest()
    records = [
        {"source_id": "fa", "auxiliary_mapping_status": "mapped"},
        {"source_id": "hf_bioavailability", "auxiliary_mapping_status": "not_applicable"},
    ]
    audit = left.coverage_audit(records)
    assert audit["validations"]["all_applicable_records_mapped"] is True
