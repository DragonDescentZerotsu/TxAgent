from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from predict.harnesses.progressive import runner
from predict.harnesses.progressive._records import (
    build_tianang_aligned_messages,
    tianang_aligned_levels,
)
from predict.harnesses.progressive.inference import (
    _full_flat_errors,
    _full_flat_state,
    _normalize_full_flat_prediction,
)
from predict.harnesses.progressive.prompt import (
    prompt_asset_manifest,
    prompt_assets,
    prompt_directory,
)
from predict.harnesses.progressive.state import ProgressiveTaskContract
from predict.utils.json import sha256_file


CONTRACT = ProgressiveTaskContract(
    task="bbb_martins",
    endpoint_name="meaningful CNS access",
    label_scope="meaningful_cns_access.v1",
    prediction_field="final_prediction",
    positive_prediction="pass",
    negative_prediction="fail",
    system_role="Reason carefully.",
    task_instructions=("Choose pass or fail.",),
)


def _active() -> dict:
    return {
        "l1-parent": {
            "analog_id": "l1-parent",
            "canonical_smiles": "CC",
            "morgan_similarity": 0.8,
            "transfer_likelihood": 0.9,
            "first_seen_level": 1,
            "group_kind": "l1_context",
            "_selection_rank": 0,
            "cards": {
                "stable-l1": {
                    "card_id": "stable-l1",
                    "first_seen_level": 1,
                    "endpoint": "brain exposure",
                    "reported_value": "present",
                    "retrieved_by": "assay_transfer_contrastive",
                }
            },
        },
        "l5-parent": {
            "analog_id": "l5-parent",
            "canonical_smiles": "CO",
            "morgan_similarity": 0.6,
            "first_seen_level": 5,
            "group_kind": "parent_molecule",
            "_selection_rank": 1,
            "cards": {
                "stable-l5": {
                    "card_id": "stable-l5",
                    "first_seen_level": 5,
                    "endpoint": "uptake",
                    "reported_value": "weak",
                    "morgan_similarity": 0.6,
                    "retrieved_by": "morgan",
                }
            },
        },
    }


@pytest.mark.parametrize(
    "version",
    ["full_flat_progressive_v1", "full_flat_progressive_v2", "full_flat_progressive_v3"],
)
def test_bundle_is_self_contained_and_renders_one_combined_update(version):
    assets = prompt_assets(version)
    provenance = json.loads(
        (prompt_directory(version) / "provenance.json").read_text(encoding="utf-8")
    )
    assert "asset_parent_version" not in provenance
    assert str(assets["settings"]["output_contract"]).startswith(
        "full_flat_progressive."
    )

    all_levels = tianang_aligned_levels("bbb_martins", prompt_version=version)
    messages, references = build_tianang_aligned_messages(
        contract=CONTRACT,
        levels=[all_levels[0], all_levels[-1]],
        current_level=5,
        query_smiles="CN",
        condition_sentence="",
        query_prior={"passive_bbb_plausibility": "moderate"},
        query_tool_summary={},
        active=_active(),
        prior_state={
            "level": 1,
            "claims": [{
                "claim": "Direct evidence supports access.",
                "molecule_ids": ["Molecule 1"],
                "record_ids": ["C01"],
                "evidence_role": "supportive",
            }],
            "summary": "L1 supports access.",
            "final_prediction": "pass",
        },
        prompt_version=version,
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=10,
        retrieval_policy={
            "L1": "assay_transfer_contrastive",
            "L2": "assay_transfer",
            "L3": "assay_transfer",
            "L4": "assay_transfer",
            "L5": "morgan",
        },
        molecule_limit=10,
        return_reference_index=True,
    )
    user = messages[1]["content"]
    assert "# L1 — direct_brain_exposure" not in user
    assert "# L5 — influx_transport" not in user
    assert "The direct and indirect records are grouped by parent molecule." in user
    assert "# L1 prior decision" in user
    assert user.count("## Molecule 1") == 1
    assert user.count("## Molecule 2") == 1
    assert [row["visible_id"] for row in references] == [
        "Molecule 1", "C01", "Molecule 2", "C02"
    ]
    assert [row["is_new"] for row in references] == [False, False, True, True]


def test_v2_restores_pass_fail_language_without_asset_inheritance():
    version = "full_flat_progressive_v2"
    assets = prompt_assets(version)
    provenance = json.loads(
        (prompt_directory(version) / "provenance.json").read_text(encoding="utf-8")
    )
    assert "asset_parent_version" not in provenance
    oral = assets["tasks"]["bioavailability_ma"]
    assert oral["positive_prediction"] == "pass"
    assert oral["negative_prediction"] == "fail"


def test_v3_uses_relaxed_versioned_output_contract():
    version = "full_flat_progressive_v3"
    assets = prompt_assets(version)
    provenance = json.loads(
        (prompt_directory(version) / "provenance.json").read_text(encoding="utf-8")
    )
    assert "asset_parent_version" not in provenance
    assert assets["settings"]["output_contract"] == "full_flat_progressive.v2"
    assert runner.FULL_FLAT_PROGRESSIVE_HARNESSES["full-flat-progressive-v3"] == version


def test_v3_keeps_the_v2_model_visible_prompt_surface():
    levels = [tianang_aligned_levels(
        "bbb_martins", prompt_version="full_flat_progressive_v2"
    )[0]]
    kwargs = dict(
        contract=CONTRACT,
        levels=levels,
        current_level=1,
        query_smiles="CN",
        condition_sentence="",
        query_prior={"passive_bbb_plausibility": "moderate"},
        query_tool_summary={},
        active={"l1-parent": _active()["l1-parent"]},
        prior_state=None,
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=10,
        retrieval_policy={"L1": "assay_transfer_contrastive"},
        molecule_limit=10,
    )
    v2 = build_tianang_aligned_messages(
        **kwargs, prompt_version="full_flat_progressive_v2"
    )
    v3 = build_tianang_aligned_messages(
        **kwargs, prompt_version="full_flat_progressive_v3"
    )
    assert v3 == v2


def test_v3_relaxes_nested_metadata_but_keeps_top_level_types():
    references = [
        {"visible_id": "C01", "stable_id": "record-1", "unit_kind": "record", "is_new": False},
    ]
    content = {
        "claims": [
            {"claim": "Old evidence.", "record_ids": ["C01"], "extra": "kept"},
            "unstructured claim",
        ],
        "summary": "The decision changed.",
        "final_prediction": "fail",
        "revision_action": "reconsidered",
        "new_evidence_assessment": [{
            "record_ids": ["C01"],
            "applicability": "partly useful",
        }],
        "extra_top_level": "accepted",
    }
    assert _full_flat_errors(
        content,
        contract=CONTRACT,
        reference_index=references,
        prior_state={"final_prediction": "pass"},
        relaxed=True,
    ) == []
    state = _full_flat_state(content, reference_index=references, level=5)
    assert state["claims"] == content["claims"]
    assert state["supportive_card_ids"] == []
    assert state["contradictory_card_ids"] == []

    content["claims"] = "not an array"
    assert "claims must be an array" in _full_flat_errors(
        content,
        contract=CONTRACT,
        reference_index=references,
        prior_state={"final_prediction": "pass"},
        relaxed=True,
    )


@pytest.mark.parametrize(
    ("task", "value", "expected"),
    [
        ("bbb_martins", "BBB+", "pass"),
        ("bbb_martins", "negative", "fail"),
        ("bioavailability_ma", "HIGH", "pass"),
        ("bioavailability_ma", 0, "fail"),
        ("bbb_martins", "uncertain", "uncertain"),
    ],
)
def test_v3_normalizes_only_reviewed_prediction_aliases(task, value, expected):
    assert _normalize_full_flat_prediction(task, CONTRACT, value) == expected


def test_full_flat_claims_allow_multi_citation_and_role_overlap():
    references = [
        {"visible_id": "Molecule 1", "stable_id": "parent@L1", "unit_kind": "molecule", "is_new": False},
        {"visible_id": "C01", "stable_id": "record-1", "unit_kind": "record", "is_new": False},
        {"visible_id": "Molecule 2", "stable_id": "parent@L2", "unit_kind": "molecule", "is_new": True},
        {"visible_id": "C02", "stable_id": "record-2", "unit_kind": "record", "is_new": True},
    ]
    content = {
        "claims": [
            {"claim": "Supports.", "molecule_ids": ["Molecule 1", "Molecule 2"], "record_ids": ["C01", "C02"], "evidence_role": "supportive"},
            {"claim": "Also conflicts.", "molecule_ids": ["Molecule 2"], "record_ids": ["C02"], "evidence_role": "contradictory"},
        ],
        "summary": "The indirect evidence strengthens the original decision.",
        "final_prediction": "pass",
        "revision_action": "strengthen",
        "new_evidence_assessment": [{
            "molecule_ids": ["Molecule 2"],
            "record_ids": ["C02"],
            "applicability": "moderate",
            "direction": "supportive",
            "decision_effect": "strengthened",
        }],
    }
    prior = {"final_prediction": "pass"}
    assert _full_flat_errors(
        content, contract=CONTRACT, reference_index=references, prior_state=prior
    ) == []
    state = _full_flat_state(content, reference_index=references, level=5)
    assert state["supportive_card_ids"] == ["record-1", "record-2"]
    assert state["contradictory_card_ids"] == ["record-2"]


def test_full_flat_l1_carries_progressive_handoff_state():
    references = [
        {"visible_id": "C01", "stable_id": "record-1", "unit_kind": "record", "is_new": False},
    ]
    content = {
        "claims": [{
            "claim": "Direct evidence supports access.",
            "molecule_ids": [],
            "record_ids": ["C01"],
            "evidence_role": "supportive",
        }],
        "summary": "Direct evidence supports access with an unresolved route gap.",
        "final_prediction": "pass",
        "confidence": "moderate",
        "evidence_gaps": ["No matched in-vivo systemic exposure record."],
    }
    assert _full_flat_errors(
        content, contract=CONTRACT, reference_index=references, prior_state=None
    ) == []


def test_partial_l1_prior_reuses_finished_queries_by_default(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(
        runner.PROGRESSIVE_TASKS,
        "bbb_martins",
        SimpleNamespace(input_jsonl=input_path),
    )
    root = tmp_path / "prior"
    prompt = prompt_asset_manifest("full_flat_progressive_v2")
    manifest = {
        "harness_version": "full-flat-progressive-v2",
        "reasoning_phase": "l1",
        "prompt_profile": "full_flat_progressive_v2",
        "model": "test-model",
        "inputs": {"bbb_martins": {"input_sha256": sha256_file(input_path)}},
        "prompt_assets": {"bbb_martins": prompt},
    }
    root.mkdir()
    (root / "experiment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "run.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    level = root / "bbb_martins/queries/query_idx0000/levels/level_1"
    level.mkdir(parents=True)
    (level / "prepared.json").write_text(
        json.dumps({"benchmark_row_id": "q0"}), encoding="utf-8"
    )
    (level / "output.json").write_text(
        json.dumps({"status": "ok", "state": {}}), encoding="utf-8"
    )
    (level / "request.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        l1_prior_run=root,
        harness_version="full-flat-progressive-v2",
        reasoning_phase="indirect-update",
        prompt_version="full_flat_progressive_v2",
        model="test-model",
        tasks=["bbb_martins"],
        require_complete_l1_prior=False,
    )
    records = {"bbb_martins": [
        {"benchmark_row_id": "q0"}, {"benchmark_row_id": "q1"}
    ]}
    sources, audit = runner._validate_full_flat_l1_source(
        args, records, {"bbb_martins": [0, 1]}
    )
    assert set(sources["bbb_martins"]) == {"q0"}
    assert audit["reused_counts_by_task"] == {"bbb_martins": 1}
    assert audit["missing_counts_by_task"] == {"bbb_martins": 1}
    assert audit["source_status"] == "complete"

    args.require_complete_l1_prior = True
    with pytest.raises(ValueError, match="is not complete"):
        runner._validate_full_flat_l1_source(
            args, records, {"bbb_martins": [0, 1]}
        )

    args.require_complete_l1_prior = False
    args.harness_version = "full-flat-progressive-v3"
    args.prompt_version = "full_flat_progressive_v3"
    sources, audit = runner._validate_full_flat_l1_source(
        args, records, {"bbb_martins": [0, 1]}
    )
    assert set(sources["bbb_martins"]) == {"q0"}
    assert audit["reused_counts_by_task"] == {"bbb_martins": 1}


def test_full_flat_indirect_chains_from_scratch_without_prior():
    args = runner.parse_args([
        "--harness-version", "full-flat-progressive-v2",
        "--reranking", "assay-transfer-contrastive",
        "--reasoning-phase", "indirect-update",
    ])
    assert args.l1_prior_run is None
    assert args.require_complete_l1_prior is False


def test_full_flat_flip_requires_new_evidence_citation():
    references = [
        {"visible_id": "C01", "stable_id": "record-1", "unit_kind": "record", "is_new": False},
        {"visible_id": "C02", "stable_id": "record-2", "unit_kind": "record", "is_new": True},
    ]
    content = {
        "claims": [{
            "claim": "Old evidence only.", "molecule_ids": [],
            "record_ids": ["C01"], "evidence_role": "supportive",
        }],
        "summary": "Changed.",
        "final_prediction": "fail",
        "revision_action": "flip",
        "new_evidence_assessment": [],
    }
    errors = _full_flat_errors(
        content,
        contract=CONTRACT,
        reference_index=references,
        prior_state={"final_prediction": "pass"},
    )
    assert "a flip must cite at least one indirect molecule or record" in errors
