"""Behavioral checks for the BBB V24.1 level-cache contract."""

from predict.harnesses.progressive.runner import (
    _record_level_names,
    _run_levels,
    parse_args,
)
from predict.retrieval.assay_reranking.v24_1_levels import (
    LEVELS,
    MODELS,
    V241PromptRenderer,
    prompt_payload,
)


def _record(**source_fields):
    return {
        "record_id": "record",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "progressive_level": "L2",
        "canonical_smiles": "CCO",
        "measurement_kind": "continuous",
        "source_fields": source_fields,
        "canonical_pair_fields_json": "{}",
        "canonical_measurement_scale_id": "",
        "canonical_category_id": "",
        "finite_scalar_value": 42.0,
        "canonical_measurement_text": "42",
        "canonical_unit_text": "ng/mL",
        "canonical_transporter_identifier": None,
    }


def test_query_keeps_context_but_hides_result_bearing_source_fields():
    prompt = V241PromptRenderer().render(
        _record(
            source_smiles="CCO",
            assay_model="PAMPA",
            species="rat",
            measurement_text="42",
            unit_text="ng/mL",
            support_text="observed value 42 ng/mL",
            extra_details="reported concentration 42 ng/mL",
            bbb_permeability_label="permeable",
        ),
        "CCN",
    )
    known, query = prompt.split("Experiment B (query measurement hidden)")
    assert "observed value 42 ng/mL" in known
    assert "reported concentration 42 ng/mL" in known
    assert "Assay model: PAMPA" in query and "Species: rat" in query
    for hidden in (
        "observed value 42 ng/mL",
        "reported concentration 42 ng/mL",
        "Known source measurement: 42",
        "BBB permeability label: permeable",
    ):
        assert hidden not in query


def test_binary_projection_uses_semantic_display_and_result_tail():
    record = _record(
        source_smiles="CCO",
        assay_model="in vivo",
        bbb_permeability_label="legacy raw label",
    )
    record.update(
        canonical_measurement_scale_id="bbb_permeability_binary.v1",
        canonical_category_id="positive",
        finite_scalar_value=1.0,
    )
    payload = prompt_payload(record, record["source_fields"])
    assert payload["measurement_text"] == "permeable"
    assert payload["unit_text"] == "BBB permeability"
    assert list(payload)[-2:] == ["measurement_text", "unit_text"]
    assert "bbb_permeability_label" not in payload


def test_level_models_and_runner_profile_are_split_specific():
    assert tuple(MODELS) == LEVELS
    assert MODELS["L2"]["revision"] == "88a5f51f7b7903c0fa5ffc1811d8156b3f0f8eac"
    args = parse_args([
        "--tasks", "bbb_martins", "--max-level", "4",
        "--query-prior", "none",
        "--prepare-only",
        "--skip-tool-prefetch",
    ])
    assert _record_level_names(args, "bbb_martins") == LEVELS
    assert [row["level"] for row in _run_levels(args, "bbb_martins")] == [1, 2, 3, 4]
