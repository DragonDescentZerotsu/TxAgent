"""Tests for the new text group-prompt formats and their editable field policy."""

import pytest

from tools.chembl_tool.tasks.bioavailability_ma import group_prompt_field_policy as policy
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_render import (
    build_group_messages,
    group_system_message,
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


def test_assay_transfer_shows_score_and_single_winning_record():
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
    }
    group = _group([_neighbor("M1", "c1ccccc1", 0.45, [EXAMPLE], transfer=0.294, winning=winning)])
    _, user = build_group_messages(QUERY, group, prompt_format="assay_transfer_tool")
    content = user["content"]
    assert "transfer likelihood (0-1): 0.29" in content
    assert content.count("Assay measurement:") == 1
    assert "Winning transfer record" not in content  # internal detail must not reach the prompt
    # SMILES shown once (neighbor header); the in-block SMILES variants are excluded.
    assert "c1ccccc1" in content
    assert "c1ccccc1-canon" not in content
    # only the winning record, not the 5-example dump semantics
    assert "Records (" not in content


def test_field_policy_toggle_changes_output():
    winning = {"original_smiles": "c1ccccc1", "canonical_smiles": "CANONVAL",
               "canonical_endpoint_key": "Fg.efflux", "context": {}}
    group = _group([_neighbor("M1", "c1ccccc1", 0.45, [EXAMPLE], transfer=0.5, winning=winning)])

    _, before = build_group_messages(QUERY, group, prompt_format="assay_transfer_tool")
    assert "CANONVAL" not in before["content"]

    # Flip canonical_smiles include on, then restore.
    specs = policy.DEFAULT_POLICY["assay_transfer_tool.record"]
    idx = next(i for i, s in enumerate(specs) if s.key == "canonical_smiles")
    specs[idx] = policy.FieldSpec("canonical_smiles", "canonical SMILES", include=True)
    try:
        _, after = build_group_messages(QUERY, group, prompt_format="assay_transfer_tool")
        assert "CANONVAL" in after["content"]
    finally:
        specs[idx] = policy.FieldSpec("canonical_smiles", "canonical SMILES", include=False)


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


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        build_group_messages(QUERY, _group([]), prompt_format="nope")
