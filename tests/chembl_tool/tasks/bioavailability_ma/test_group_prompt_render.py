"""Tests for the new text group-prompt formats and their editable field policy."""

import hashlib

import pytest

from tools.chembl_tool.tasks.bioavailability_ma import group_prompt_field_policy as policy
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_render import (
    _assay_transfer_evidence_record,
    build_group_messages,
    group_output_schema_provenance,
    group_output_validation,
    group_system_message,
    group_prompt_provenance,
    instruction_file_provenance,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
    full_record_example,
)
from tools.chembl_tool.common.starling.normalized_evidence import (
    EndpointOrthography,
    FamilyAssignment,
    NormalizedSourceProfile,
    aggregate_molecule_family_records,
    normalize_source_rows,
)


def _neighbor(molecule_id, smiles, similarity, examples, *, transfer=None, winning=None):
    row = {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "minimal_evidence": {
            "contract_version": "minimal_evidence.v1",
            "source": {"name": "starling-labs/bioavailability_ma/Fg", "record_id": "R"},
            "molecule": {"id": molecule_id, "canonical_smiles": smiles, "names": []},
            "group": {"id": "Fg.x", "tier": "Fg", "endpoint_group": "x"},
            "endpoint": {"name": "e", "measurement": {"relation": "", "value": "", "unit": ""}},
            "text": {"evidence": "blob", "context": ""},
            "annotations": {"evidence_role": "mechanistic_factor", "scope": {},
                            "transferability": "not_assessed", "uncertainty": []},
            "quality": {"confidence": 0.9},
            "provenance": {"assay_id": "R", "source_record_count": len(examples)},
            "examples": examples,
        },
    }
    neighbor = {
        "rank": 1,
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "similarity": similarity,
        "similarity_bucket": "weak_analog",
        "evidence_rows": [row],
    }
    if transfer is not None:
        neighbor["transfer_selection_score"] = transfer
    if winning is not None:
        neighbor["transfer_winning_record"] = winning
    return neighbor


def _group(neighbors):
    return {
        "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
        "tier": "Fg",
        "endpoint_group": "gut_wall_efflux_intestinal_metabolism",
        "evidence_source": "starling-labs/bioavailability_ma/Fg",
        "neighbors": neighbors,
    }


QUERY = {"input_smiles": "CCO", "canonical_smiles": "CCO"}
EXAMPLE = {
    "endpoint_type": "efflux_or_secretory_transport",
    "reported_value": "12.4%",
    "reported_units": "",
    "context": {"transporter_or_enzyme": "P-gp/ABCB1", "substrate_status": "substrate"},
    "support_text": "polarized transport observed",
    "source_confidence": 0.96,
}


def test_morgan_is_text_not_json_and_shows_records():
    group = _group([_neighbor("M1", "c1ccccc1", 0.45, [EXAMPLE])])
    _, user = build_group_messages(QUERY, group, prompt_format="morganfingerprint",
                                   options={"prompt_min_similarity": 0.3})
    content = user["content"]
    # Human-readable text sections, not the legacy JSON payload keys.
    assert "NEIGHBOR ANALOGS" in content and "Records (1)" in content
    assert '"evidence_rows"' not in content and '"minimal_evidence"' not in content
    # The record's support text and endpoint appear; the duplicated text.evidence blob does not.
    assert "polarized transport observed" in content
    assert "blob" not in content


def test_normalized_record_shows_the_resolved_extraction_pair():
    rendered = dict(
        _assay_transfer_evidence_record(
            {
                "source_contract": {
                    "contract_version": "source_column_contract.v1",
                    "source_or_simply_cleaned": {"measurement_text": True},
                },
                "source_fields": {"measurement_text": "4.2 × 10^-6"},
                "resolved_measurement_display": {
                    "value": "4.2",
                    "unit": "10^-6 cm/s",
                    "origin": "llm",
                },
            },
            "Starling normalized oral bioavailability",
            _group([]),
        )
    )

    assert rendered["resolved measurement (extracted scale)"] == (
        "value: 4.2; unit: 10^-6 cm/s; origin: llm"
    )
    assert "source record" in rendered


@pytest.mark.parametrize("prompt_format", ["morganfingerprint", "assay_transfer_tool"])
def test_named_text_prompt_version_fingerprints_jinja_and_instructions(prompt_format):
    provenance = group_prompt_provenance(
        prompt_format,
        prompt_version="bioavailability_text_v1",
        output_schema_profile=(
            "assay-transfer" if prompt_format == "assay_transfer_tool" else "legacy"
        ),
    )
    assert provenance["prompt_version"] == "bioavailability_text_v1"
    assert len(provenance["template_sha256"]) == 64
    assert len(provenance["instructions_sha256"]) == 64
    assert "bioavailability_text_v1" in provenance["template_path"]


def test_flat_no_tools_text_prompt_omits_tool_guidance():
    group = _group([_neighbor("M1", "c1ccccc1", 0.45, [EXAMPLE])])
    group["identity_blind"] = True
    messages = build_group_messages(
        QUERY,
        group,
        prompt_format="morganfingerprint",
        options={
            "prompt_version": "bioavailability_text_v1",
            "group_tools_enabled": False,
            "omit_query_tools": True,
        },
    )
    assert "No tools are available for this branch" in messages[0]["content"]
    assert "Do not infer query identity" in messages[0]["content"]
    assert "mmp_structure_compare" not in messages[1]["content"]
    assert "properties_compare" not in messages[1]["content"]


def test_normalized_starling_evidence_row_renders_non_blank_endpoint_value_unit():
    """Normalized Starling rows retain endpoint, value, and unit fields."""
    profile = NormalizedSourceProfile(
        source_id="v7_test",
        source_name="test/starling_v7",
        endpoint_field="kind",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
        record_id_field="record",
    )
    result = normalize_source_rows(
        [{"smiles": "c1ccccc1", "kind": "efflux_or_secretory_transport",
          "value": "12.4", "unit": "%", "record": "1"}],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=lambda source_id, endpoint_name: EndpointOrthography(
            endpoint_name, endpoint_name, "unchanged", "no_reviewed_correction", "test.v1"
        ),
        family_resolver=lambda source_id, endpoint_name, record=None: FamilyAssignment(
            "Fg.gut_wall_efflux_intestinal_metabolism", "Fg",
            "gut_wall_efflux_intestinal_metabolism", "mechanistic_factor", "test endpoint",
        ),
    )
    rows = aggregate_molecule_family_records(result.records)
    assert len(rows) == 1
    row = rows[0]

    neighbor = {
        "rank": 1,
        "molecule_chembl_id": row["molecule_chembl_id"],
        "canonical_smiles": row["canonical_smiles"],
        "similarity": 0.45,
        "similarity_bucket": "weak_analog",
        "evidence_rows": [row],
    }
    group = _group([neighbor])
    _, user = build_group_messages(QUERY, group, prompt_format="morganfingerprint",
                                   options={"prompt_min_similarity": 0.3})
    content = user["content"]
    assert "efflux_or_secretory_transport" in content
    assert "12.4" in content
    assert "%" in content


def test_morgan_similarity_threshold_drops_low_neighbors():
    group = _group([
        _neighbor("KEEPME", "C1CCCCC1", 0.60, [EXAMPLE]),
        _neighbor("DROPME", "C1CCNCC1", 0.20, [EXAMPLE]),
    ])
    group["neighbors"][1]["rank"] = 2
    _, user = build_group_messages(QUERY, group, prompt_format="morganfingerprint",
                                   options={"prompt_min_similarity": 0.3})
    # neighbors are identified by SMILES in the header (molecule id is not shown)
    assert "C1CCCCC1" in user["content"]
    assert "C1CCNCC1" not in user["content"]
    assert "NEIGHBOR ANALOGS (1)" in user["content"]


def test_assay_transfer_shows_score_and_one_record_per_ranked_entry():
    winning = {
        "original_smiles": "c1ccccc1",
        "canonical_smiles": "c1ccccc1-canon",
        "canonical_endpoint_key": "Fg.efflux",
        "measurement_label": "efflux",
        "value_display": "12.4%",
        "unit_basis": "percent",
        "metric_type": "positive_scalar",
        "threshold_display": "2-fold",
        "endpoint_subtype": "efflux_or_secretory_transport",
        "context": {"transporter_or_enzyme": "P-gp/ABCB1"},
        "support_text": "polarized transport observed",
    }
    group = _group([_neighbor("M1", "c1ccccc1", 0.45, [EXAMPLE], transfer=0.294, winning=winning)])
    _, user = build_group_messages(QUERY, group, prompt_format="assay_transfer_tool")
    content = user["content"]
    assert "transfer likelihood (0-1): 0.29" in content
    assert "SELECTED MOLECULES (1)" in content
    assert "[Molecule 1]" in content
    assert "[Assay record 1.1]" in content
    assert "[Neighbor 1]" not in content
    assert "Records (1):" not in content
    assert "endpoint: Fg.efflux" in content
    assert "value: 12.4%" in content
    assert "unit: percent" in content
    assert "evidence: polarized transport observed" in content
    assert "Winning transfer record" not in content  # internal detail must not reach the prompt
    # SMILES shown once (neighbor header); the in-block SMILES variants are excluded.
    assert "c1ccccc1" in content
    assert "c1ccccc1-canon" not in content
    # This ranked entry presents exactly one selected record without a redundant wrapper.
    assert content.count("endpoint: Fg.efflux") == 1


def test_assay_transfer_groups_multiple_endpoint_records_under_one_molecule():
    first = {
        "canonical_endpoint_key": "Fg.efflux",
        "value_display": "12.4%",
        "unit_basis": "percent",
        "support_text": "efflux evidence",
    }
    second = {
        "canonical_endpoint_key": "Fg.substrate_status",
        "value_display": "substrate",
        "support_text": "substrate evidence",
    }
    neighbor = _neighbor(
        "M1", "c1ccccc1", 0.45, [EXAMPLE], transfer=0.876, winning=first
    )
    neighbor["transfer_selected_records"] = [
        {
            "record_rank": 1,
            "transfer_selection_score": 0.876,
            "transfer_winning_record_id": "hidden-one",
            "canonical_endpoint_key": "fg_efflux",
            "transfer_winning_record": first,
        },
        {
            "record_rank": 2,
            "transfer_selection_score": 0.754,
            "transfer_winning_record_id": "hidden-two",
            "canonical_endpoint_key": "fg_substrate_status",
            "transfer_winning_record": second,
        },
    ]

    _, user = build_group_messages(
        QUERY, _group([neighbor]), prompt_format="assay_transfer_tool"
    )
    content = user["content"]

    assert "SELECTED MOLECULES (1)" in content
    assert "[Molecule 1]" in content
    assert "[Assay record 1.1]" in content
    assert "[Assay record 1.2]" in content
    assert "transfer likelihood (0-1): 0.88" in content
    assert "transfer likelihood (0-1): 0.75" in content
    assert "endpoint: Fg.efflux" in content
    assert "endpoint: Fg.substrate_status" in content
    assert "hidden-one" not in content and "hidden-two" not in content


def test_assay_transfer_output_schema_is_evidence_centric_and_identity_free():
    _, user = build_group_messages(
        QUERY,
        _group([]),
        prompt_format="assay_transfer_tool",
        options={"output_schema_profile": "assay-transfer"},
    )
    content = user["content"]

    assert '"assay_transfer_assessment": "string"' in content
    assert '"bioavailability_implications": [' in content
    assert '"record_rank": "integer or null"' in content
    assert '"assay_endpoint": "string"' in content
    assert '"transfer_likelihood": "number or null"' in content
    assert '"molecule_chembl_id"' not in content
    assert '"similarity_bucket"' not in content
    assert '"tool_summary"' not in content
    assert '"evidence_direction"' not in content
    assert '"transferability"' not in content
    assert "Return a single JSON object matching the required schema shown below." in content

    validation = group_output_validation("assay-transfer")
    assert "assay_transfer_assessment" in validation["required_fields"]
    assert validation["forbidden_field_names"] == ("molecule_chembl_id",)
    assert group_output_schema_provenance("assay-transfer")["contract_version"] == (
        "bioavailability_group_output.assay_transfer.v1"
    )


def test_legacy_output_schema_remains_the_default():
    _, user = build_group_messages(
        QUERY,
        _group([]),
        prompt_format="assay_transfer_tool",
    )
    content = user["content"]

    assert '"transferability": "high | moderate | low | not_applicable"' in content
    assert '"evidence_direction":' in content
    assert '"molecule_chembl_id": "string"' in content
    assert '"assay_transfer_assessment"' not in content


def test_record_field_policy_is_shared_by_both_retrievers():
    winning = {
        "original_smiles": "c1ccccc1",
        "canonical_endpoint_key": "Fg.efflux",
        "endpoint_subtype": "efflux_or_secretory_transport",
        "value_display": "12.4%",
        "unit_basis": "percent",
        "context": {},
    }
    group = _group([_neighbor("M1", "c1ccccc1", 0.45, [EXAMPLE], transfer=0.5, winning=winning)])

    _, morgan_before = build_group_messages(
        QUERY, group, prompt_format="morganfingerprint", options={"prompt_min_similarity": 0.0}
    )
    _, transfer_before = build_group_messages(QUERY, group, prompt_format="assay_transfer_tool")
    assert "endpoint: efflux_or_secretory_transport" in morgan_before["content"]
    assert "endpoint: Fg.efflux" in transfer_before["content"]

    specs = policy.DEFAULT_POLICY["morganfingerprint.record"]
    idx = next(i for i, s in enumerate(specs) if s.key == "endpoint_type")
    original = specs[idx]
    specs[idx] = policy.FieldSpec("endpoint_type", "endpoint", include=False)
    try:
        _, morgan_after = build_group_messages(
            QUERY, group, prompt_format="morganfingerprint", options={"prompt_min_similarity": 0.0}
        )
        _, transfer_after = build_group_messages(QUERY, group, prompt_format="assay_transfer_tool")
        assert "endpoint: efflux_or_secretory_transport" not in morgan_after["content"]
        assert "endpoint: Fg.efflux" not in transfer_after["content"]
    finally:
        specs[idx] = original


def test_system_message_matches_legacy_string():
    open_group = {"group_id": "g"}
    assert group_system_message(open_group) == (
        "You are a medicinal chemistry oral bioavailability analog evidence analyst. "
        "Reason about whether analog evidence for one aspect of oral bioavailability is transferable to the query molecule. "
        "You may call the provided molecule comparison tools when structural or property differences matter. "
        "Return only valid JSON."
    )
    blind = {"group_id": "g", "identity_blind": True}
    assert group_system_message(blind) == (
        "You are a medicinal chemistry oral bioavailability analog evidence analyst. "
        "Reason about whether analog evidence for one aspect of oral bioavailability is transferable to the query molecule. "
        "Use the harness-prefetched comparison results; do not call tools. Do not infer query identity. "
        "Return only valid JSON."
    )


def test_tool_free_assay_transfer_system_message_uses_supplied_likelihoods():
    system, _ = build_group_messages(
        QUERY,
        _group([]),
        prompt_format="assay_transfer_tool",
        options={"group_tools_enabled": False},
    )

    assert "No tools are available for this branch." in system["content"]
    assert (
        "Use the supplied assay-transfer likelihoods as the best available transfer estimates."
        in system["content"]
    )
    assert "You may call" not in system["content"]


def test_assay_transfer_instruction_file_override_is_verbatim_and_fingerprinted(tmp_path):
    instructions_path = tmp_path / "ignore.txt"
    raw = (
        "# ignored comment\n"
        "Put aside structural-similarity judgments.\n"
        "\n"
        "Completely trust the supplied likelihoods.\n"
    )
    instructions_path.write_text(raw, encoding="utf-8")

    _, user = build_group_messages(
        QUERY,
        _group([]),
        prompt_format="assay_transfer_tool",
        options={"instructions_file": str(instructions_path)},
    )
    provenance = instruction_file_provenance(
        "assay_transfer_tool", instructions_path
    )

    assert "1. Put aside structural-similarity judgments." in user["content"]
    assert "2. Completely trust the supplied likelihoods." in user["content"]
    assert provenance == {
        "path": str(instructions_path.resolve()),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "instruction_count": 2,
        "instructions": [
            "Put aside structural-similarity judgments.",
            "Completely trust the supplied likelihoods.",
        ],
    }


def test_empty_group_instruction_file_is_rejected(tmp_path):
    instructions_path = tmp_path / "empty.txt"
    instructions_path.write_text("# comments only\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no instruction lines"):
        instruction_file_provenance("assay_transfer_tool", instructions_path)


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        build_group_messages(QUERY, _group([]), prompt_format="nope")


# --- Per-source `full` presentation style ----------------------------------------

# In-distribution normalized winning/catalog record with the full scientific payload.
INDIST_WINNING = {
    "original_smiles": "c1ccccc1",
    "canonical_smiles": "c1ccccc1",
    "canonical_endpoint_key": "q3.intestinal_transport.efflux_ratio.dimensionless_ratio.secretory_over_absorptive",
    "endpoint_family": "intestinal_transport",
    "endpoint_subtype": "efflux_ratio",
    "measurement_label": "efflux ratio",
    "value": 1.5,
    "value_display": "1.5",
    "unit_basis": "dimensionless_ratio",
    "unit_normalized": "ratio",
    "metric_type": "dimensionless_ratio",
    "threshold_display": "within 2-fold / at least 5-fold apart",
    "direction": "higher_is_more_efflux",
    "variation_type": "sd",
    "variation_value": "0.2",
    "statistic_type": "mean",
    "assay_concept": "gut_wall_efflux",
    "context": {"study_or_assay_system": "Caco-2 bidirectional transport"},
    "support_text": "efflux ratio 1.5 indicates limited active efflux",
    "source_contract": {
        "contract_version": "source_column_contract.v2",
        "source_id": "fg",
        "source_or_simply_cleaned": {"measured_value": True, "pmid": True},
    },
    "source_fields": {
        "measured_value": "efflux ratio 1.5",
        "pmid": "12345",
    },
}


def _indist_group(neighbors):
    group = _group(neighbors)
    group["evidence_source"] = "starling-in-distribution/Fg"
    return group


def _indist_example(winning):
    """Build the source-contracted example emitted by the library."""
    return {
        "source_contract": winning["source_contract"],
        "source_fields": winning["source_fields"],
    }


# Scientific fields that only the `full` view should surface (label prefixes).
_SCI_LINES = [
    "endpoint (canonical): q3.intestinal_transport.efflux_ratio",
    "metric type: dimensionless_ratio",
    "threshold: within 2-fold",
    "direction: higher_is_more_efflux",
    "variation type: sd",
    "statistic type: mean",
    "unit (normalized): ratio",
]


def test_all_styles_surface_only_source_fields_for_in_distribution():
    group = _indist_group([_neighbor("M1", "c1ccccc1", 0.45, [_indist_example(INDIST_WINNING)])])

    _, legacy = build_group_messages(
        QUERY, group, prompt_format="morganfingerprint",
        options={"prompt_min_similarity": 0.0},
    )
    _, full = build_group_messages(
        QUERY, group, prompt_format="morganfingerprint",
        options={"prompt_min_similarity": 0.0, "presentation_style": "full"},
    )
    # Both styles use the same source-faithful projection.
    for line in _SCI_LINES:
        assert line not in legacy["content"]
        assert line not in full["content"]
    assert "measured_value: efflux ratio 1.5" in legacy["content"]
    assert "pmid: 12345" in legacy["content"]
    assert "measured_value: efflux ratio 1.5" in full["content"]


def test_full_style_is_per_source_txagent_gets_no_scoring_fields():
    # TxAgent-library source: report/prose fields, NOT the normalized scoring fields.
    example = {
        "endpoint_type": "oral bioavailability",
        "dose": "10 mg/kg",
        "species_or_population": "rat",
        "support_text": "oral bioavailability reported",
        "metric_type": "dimensionless_ratio",
    }
    group = _group([_neighbor("M1", "c1ccccc1", 0.45, [example])])  # starling-labs source
    _, full = build_group_messages(
        QUERY, group, prompt_format="morganfingerprint",
        options={"prompt_min_similarity": 0.0, "presentation_style": "full"},
    )
    content = full["content"]
    assert "dose: 10 mg/kg" in content
    assert "species/population: rat" in content
    # the txagent `full` spec does not list metric_type, so it must not appear
    assert "metric type: dimensionless_ratio" not in content


def test_full_style_keeps_contracted_provenance_and_omits_uncontracted_ids():
    example = _indist_example(INDIST_WINNING)
    example.update({"pmid": "12345678", "source_id": "SRC1", "record_id": "REC1"})
    group = _indist_group([_neighbor("M1", "c1ccccc1", 0.45, [example])])
    _, full = build_group_messages(
        QUERY, group, prompt_format="morganfingerprint",
        options={"prompt_min_similarity": 0.0, "presentation_style": "full"},
    )
    content = full["content"]
    assert "pmid: 12345" in content
    assert "12345678" not in content
    assert "SRC1" not in content and "REC1" not in content


def test_full_is_invariant_across_retrievers():
    """The `full` record view must be identical for Morgan and assay-transfer retrieval of
    the same in-distribution source (retriever-invariance)."""
    morgan_group = _indist_group(
        [_neighbor("M1", "c1ccccc1", 0.45, [_indist_example(INDIST_WINNING)])]
    )
    transfer_group = _indist_group(
        [_neighbor("M1", "c1ccccc1", 0.45, [EXAMPLE], transfer=0.5, winning=INDIST_WINNING)]
    )
    _, morgan_full = build_group_messages(
        QUERY, morgan_group, prompt_format="morganfingerprint",
        options={"prompt_min_similarity": 0.0, "presentation_style": "full"},
    )
    _, transfer_full = build_group_messages(
        QUERY, transfer_group, prompt_format="assay_transfer_tool",
        options={"presentation_style": "full"},
    )
    assert "measured_value: efflux ratio 1.5" in morgan_full["content"]
    assert "measured_value: efflux ratio 1.5" in transfer_full["content"]
    for line in _SCI_LINES:
        assert line not in morgan_full["content"], line
        assert line not in transfer_full["content"], line


def test_full_resolves_evidence_source_from_evidence_rows():
    """In the real retrieval structure, evidence_source lives on each evidence row (not the
    group/neighbor). The `full` policy must still resolve the per-source spec from there."""
    ex = _indist_example(INDIST_WINNING)
    row = {"evidence_source": "starling-in-distribution/Fg", "source_record_examples": [ex]}
    neighbor = {
        "rank": 1, "molecule_chembl_id": "M1", "canonical_smiles": "c1ccccc1",
        "similarity": 0.45, "similarity_bucket": "weak_analog", "evidence_rows": [row],
    }
    # No group-level or neighbor-level evidence_source on purpose.
    group = {"group_id": "Fg.x", "tier": "Fg", "endpoint_group": "x", "neighbors": [neighbor]}
    _, full = build_group_messages(
        QUERY, group, prompt_format="morganfingerprint",
        options={"prompt_min_similarity": 0.0, "presentation_style": "full"},
    )
    assert "measured_value: efflux ratio 1.5" in full["content"]
    for line in _SCI_LINES:
        assert line not in full["content"], line


def test_included_fields_style_and_prefix_matching():
    indist = "starling-in-distribution/Fg"
    full_specs = policy.included_fields("morganfingerprint.record", indist, "full")
    keys = [k for k, _ in full_specs]
    assert keys == [
        "resolved_measurement_display",
        "source_contract",
        "source_fields",
    ]
    # legacy uses the same contract and cannot expose canonical scoring fields.
    legacy_specs = policy.included_fields("morganfingerprint.record", indist, "legacy")
    assert [k for k, _ in legacy_specs] == keys
    # unknown source under `full` falls back to the legacy policy
    unknown = policy.included_fields("morganfingerprint.record", "some-other-source/Fg", "full")
    assert [k for k, _ in unknown] == [
        "endpoint_type", "reported_value", "reported_units", "context", "support_text",
    ]
