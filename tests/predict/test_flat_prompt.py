from __future__ import annotations

import importlib
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from predict.utils.response import response_validation_errors
from predict.harnesses.branches.runner import (
    _parse_args,
    _validate_assay_transfer_scores,
    _validate_flat_prompt,
    _validated_flat_selection_manifest,
    _validate_group_prompt_contract,
    prediction_to_label,
)
from predict.harnesses.branches.tasks.dili.contract import CONFIG as DILI_CONFIG
from predict.harnesses.branches.flat import (
    ASSAY_TRANSFER_VARIANT,
    CONTEXT_V4_HARNESS_VERSION,
    CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_V5_HARNESS_VERSION,
    CONTEXT_V5_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
    CONTEXT_V6_PROMPT_VERSION,
    CONTEXT_V7_HARNESS_VERSION,
    CONTEXT_V7_PROMPT_VERSION,
    CONTEXT_V8_HARNESS_VERSION,
    CONTEXT_V8_PROMPT_VERSION,
    EVIDENCE_PROJECTION,
    EXTRA_DETAILS_POLICY,
    JOINT_VARIANT,
    JOSEPH_PROMPT_VERSION,
    JOSEPH_V1_HARNESS_VERSION,
    JOSEPH_V1_PROMPT_VERSION,
    MORGAN_VARIANT,
    PROMPT_VERSION,
    PUBLIC_HARNESS_VERSION,
    apply_flat_group_prompt,
    build_flat_context_request,
    cache_matched_flat_retrieval,
    derive_flat_claim_evidence,
    flat_group_validation,
    flat_context_validation,
    native_flat_prediction,
    flat_prompt_provenance,
    flat_prompt_variant,
    prompt_asset_manifest,
    prompt_directory,
    render_flat_group_messages,
)
from predict.harnesses.branches import flat
from predict.harnesses.branches import inference as calls
from predict.harnesses.branches.tasks.bbb_martins.contract import CONFIG as BBB_CONFIG
from predict.harnesses.branches.tasks.bioavailability_ma.contract import (
    CONFIG as BIOAVAILABILITY_CONFIG,
)
from predict.harnesses.branches.tasks.skin_reaction.contract import CONFIG as SKIN_CONFIG
from predict.api_client.pool import OpenAIProviderPool


FIXTURE_PATH = Path("tests/predict/fixtures/tianang_flat_v1_rendered.json")
JOSEPH_FIXTURE_PATH = Path("tests/predict/fixtures/joseph_flat_v2_rendered.json")














def test_flat_prompt_variant_is_selected_by_existing_retrieval_strategy() -> None:
    assert flat_prompt_variant("morgan_fingerprint") == MORGAN_VARIANT
    assert flat_prompt_variant("assay_transfer_tool") == ASSAY_TRANSFER_VARIANT
    with pytest.raises(ValueError, match="requires morgan_fingerprint"):
        flat_prompt_variant("unsupported")
    assert flat_prompt_variant(
        "morgan_fingerprint",
        prompt_version=JOSEPH_PROMPT_VERSION,
        flat_reranking="joint",
    ) == JOINT_VARIANT




def test_flat_prompt_bundle_is_complete_and_version_owned() -> None:
    manifest = prompt_asset_manifest()
    assert set(manifest["files_sha256"]) == {
        "modes/assay-transfer/user.yaml",
        "modes/morgan/user.yaml",
        "provenance.json",
        "system.jinja",
        "tasks.yaml",
    }
    assert manifest["assembly_files_sha256"].keys() == {"flat.py", "prompts/__init__.py"}
    assert len(manifest["sha256"]) == 64
    with pytest.raises(ValueError, match="Unknown flat prompt version"):
        prompt_directory("../tianang_flat_v1")
    joseph = prompt_asset_manifest(JOSEPH_PROMPT_VERSION)
    assert set(joseph["files_sha256"]) == {
        "modes/assay-transfer/user.yaml",
        "modes/joint/user.yaml",
        "modes/morgan/user.yaml",
        "provenance.json",
        "system.jinja",
        "tasks.yaml",
    }
    context = prompt_asset_manifest(CONTEXT_V4_PROMPT_VERSION)
    assert set(context["files_sha256"]) == {
        "modes/assay-transfer/user.yaml",
        "modes/assay-transfer-contrastive/user.yaml",
        "modes/assay-transfer-within-morgan/user.yaml",
        "modes/morgan/user.yaml",
        "modes/morgan-contrastive/user.yaml",
        "provenance.json",
        "system.jinja",
        "tasks.yaml",
        "user.jinja",
    }
    context_v5 = prompt_asset_manifest(CONTEXT_V5_PROMPT_VERSION)
    assert set(context_v5["files_sha256"]) == {
        "levels/L1.yaml",
        "levels/L2.yaml",
        "levels/L3.yaml",
        "levels/L4.yaml",
        "levels/L5.yaml",
        "levels/L6.yaml",
        "provenance.json",
        "system.jinja",
        "tasks.yaml",
        "user.jinja",
    }
    context_v6 = prompt_asset_manifest(CONTEXT_V6_PROMPT_VERSION)
    assert set(context_v6["files_sha256"]) == set(context_v5["files_sha256"])
    context_v7 = prompt_asset_manifest(CONTEXT_V7_PROMPT_VERSION)
    assert set(context_v7["files_sha256"]) == set(context_v6["files_sha256"])
    context_v8 = prompt_asset_manifest(CONTEXT_V8_PROMPT_VERSION)
    assert set(context_v8["files_sha256"]) == set(context_v5["files_sha256"])


def _flat_context_retrieval() -> dict:
    return {
        "query": {"canonical_smiles": "Q"},
        "groups": [{
            "group_id": "Flat.all_evidence",
            "neighbors": [{
                "standard_inchi_key": "PARENT",
                "canonical_smiles": "N",
                "evidence_rows": [
                    {
                        "evidence_id": "uid-1",
                        "prompt_evidence": {"card_id": "C01", "endpoint": "direct"},
                        "selection_provenance": {
                            "level": "L1", "evidence_family": "Direct.one",
                            "morgan_similarity": 0.8,
                        },
                    },
                    {
                        "evidence_id": "uid-2",
                        "prompt_evidence": {"card_id": "C02", "endpoint": "indirect"},
                        "selection_provenance": {
                            "level": "L2", "evidence_family": "Mechanism.two",
                            "morgan_similarity": 0.8,
                        },
                    },
                ],
            }],
        }],
    }


@pytest.mark.parametrize("prompt_version", [
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
])
def test_upstream_prompt_shows_neighbor_morgan_with_transfer_likelihood(
    prompt_version: str,
) -> None:
    retrieval = _flat_context_retrieval()
    neighbor = retrieval["groups"][0]["neighbors"][0]
    neighbor.update(similarity=0.73, similarity_metric="Morgan radius=2 bits=2048 Tanimoto")
    for row in neighbor["evidence_rows"]:
        row["selection_provenance"].pop("morgan_similarity")
        row["selection_provenance"]["assay_transfer_score"] = 0.88
    messages, _ = build_flat_context_request(
        retrieval, task_id="bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        layout="global", reranking="assay-transfer-contrastive",
        query_prior=None, prompt_version=prompt_version,
    )
    assert "Morgan similarity: 0.73" in messages[1]["content"]
    assert "Transfer likelihood: 0.88" in messages[1]["content"]


@pytest.mark.parametrize("benchmark", ["gold_v1", "tdc"])
def test_final_full_flat_renders_available_scores_and_valid_references(benchmark: str) -> None:
    retrieval = _flat_context_retrieval()
    retrieval["benchmark"] = benchmark
    neighbor = retrieval["groups"][0]["neighbors"][0]
    neighbor.update(similarity=0.73, similarity_metric="Morgan radius=2 bits=2048 Tanimoto")
    for row in neighbor["evidence_rows"]:
        row["selection_provenance"].pop("morgan_similarity", None)
        row["selection_provenance"]["assay_transfer_score"] = 0.88
    messages, metadata = build_flat_context_request(
        retrieval, task_id="bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        layout="global", reranking="assay-transfer-contrastive",
        query_prior=None, prompt_version=flat.FINAL_FULL_FLAT_PROMPT_VERSION,
    )
    assert "Morgan similarity: 0.73" in messages[1]["content"]
    assert "Transfer likelihood: 0.88" in messages[1]["content"]
    assert metadata["reasoning_reference_contract"]["schema_version"] == "progressive_reasoning_references.v1"
    assert flat_context_validation(
        "bioavailability_ma", task_prompt_profile="f20_evidence_calibrated_v2",
        prompt_version=flat.FINAL_FULL_FLAT_PROMPT_VERSION,
        reference_index=metadata["reasoning_reference_index"],
    )["allowed_values"]["final_prediction"] == {"pass", "fail"}
    retrieval["groups"][0]["neighbors"][0]["evidence_rows"][0]["selection_provenance"].pop(
        "assay_transfer_score"
    )
    messages, _ = build_flat_context_request(
        retrieval, task_id="bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        layout="global", reranking="assay-transfer-contrastive",
        query_prior=None, prompt_version=flat.FINAL_FULL_FLAT_PROMPT_VERSION,
    )
    assert "Morgan similarity: 0.73" in messages[1]["content"]
    assert messages[1]["content"].count("Transfer likelihood: 0.88") == 1


@pytest.mark.parametrize("band,visible", [
    ("high", True), ("middle", False), ("low", False),
])
def test_gold_score_band_hides_likelihood_below_seventy_percent(
    band: str, visible: bool,
) -> None:
    retrieval = _flat_context_retrieval()
    neighbor = retrieval["groups"][0]["neighbors"][0]
    neighbor.update(similarity=0.73, similarity_metric="Morgan radius=2 bits=2048 Tanimoto")
    for row in neighbor["evidence_rows"]:
        row["selection_provenance"]["assay_transfer_score"] = 0.88
    flat._apply_gold_score_visibility(retrieval, band)
    messages, _ = build_flat_context_request(
        retrieval, task_id="bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        layout="global", reranking="assay-transfer-contrastive",
        query_prior=None,
        prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    )
    assert "Morgan similarity: 0.8" in messages[1]["content"]
    assert ("Transfer likelihood: 0.88" in messages[1]["content"]) is visible
    assert ("The assay-transfer score" in messages[0]["content"]) is visible


def test_flat_context_layouts_keep_global_ids_and_change_only_grouping() -> None:
    common = {
        "task_id": "bbb_martins",
        "task_prompt_profile": "meaningful_cns_access_v1",
        "reranking": "morgan",
        "query_prior": None,
    }
    _, global_metadata = build_flat_context_request(
        _flat_context_retrieval(), layout="global", **common
    )
    _, level_metadata = build_flat_context_request(
        _flat_context_retrieval(), layout="level-grouped", **common
    )
    assert global_metadata["card_alias_map"] == level_metadata["card_alias_map"] == {
        "C01": "uid-1", "C02": "uid-2"
    }
    assert [
        row["visible_id"] for row in global_metadata["reasoning_reference_index"]
        if row["unit_kind"] == "molecule"
    ] == ["Molecule 1"]
    assert [
        row["visible_id"] for row in level_metadata["reasoning_reference_index"]
        if row["unit_kind"] == "molecule"
    ] == ["Molecule 1", "Molecule 2"]


def test_upstream_v5_renders_l7_safety_records() -> None:
    retrieval = _flat_context_retrieval()
    retrieval["groups"][0]["neighbors"][0]["evidence_rows"].append({
        "evidence_id": "uid-7",
        "prompt_evidence": {"card_id": "C07", "endpoint": "indirect"},
        "selection_provenance": {
            "level": "L7", "evidence_family": "Mechanism.seven",
            "morgan_similarity": 0.8,
        },
    })
    messages, metadata = build_flat_context_request(
        retrieval, task_id="dili",
        task_prompt_profile="dili_conditioned_or_source_molecule_outcome.v1",
        layout="level-grouped", reranking="morgan",
        prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        query_prior=None,
    )
    assert "uid-7" in metadata["card_alias_map"].values()
    assert "L7" in messages[1]["content"]


def test_flat_context_schema_has_only_summary_and_final_prediction() -> None:
    validation = flat_context_validation(
        "bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
    )
    assert validation["required_fields"] == ("summary", "final_prediction")
    assert validation["allowed_values"]["final_prediction"] == {"pass", "fail"}
    validator = validation["content_validator"]
    assert validator({"summary": "x", "final_prediction": "pass"}) == []
    assert validator({
        "summary": "x", "final_prediction": "pass", "confidence": "high"
    }) == ["invalid_output_fields"]




def test_v5_uses_molecule_scoped_record_ids_and_claim_schema() -> None:
    messages, metadata = build_flat_context_request(
        _flat_context_retrieval(),
        task_id="bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
        layout="level-grouped",
        reranking="assay-transfer-contrastive",
        query_prior=None,
        prompt_version=CONTEXT_V5_PROMPT_VERSION,
    )
    record_ids = [
        row["visible_id"] for row in metadata["reasoning_reference_index"]
        if row["unit_kind"] == "record"
    ]
    assert record_ids == ["Record 1-1", "Record 2-1"]
    assert metadata["card_alias_map"] == {
        "Record 1-1": "uid-1", "Record 2-1": "uid-2",
    }
    assert "# Level L2" in messages[1]["content"]

    validation = flat_context_validation(
        "bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
        prompt_version=CONTEXT_V5_PROMPT_VERSION,
        reference_index=metadata["reasoning_reference_index"],
    )
    content = {
        "claims": [{
            "claim": "The direct evidence supports the decision.",
            "molecule_ids": ["Molecule 1", "Molecule 2"],
            "record_ids": ["Record 1-1", "Record 2-1"],
            "evidence_role": "supportive",
        }],
        "final_prediction": "pass",
    }
    assert validation["content_validator"](content) == []
    content["claims"].append({
        "claim": "The same molecule and record also contain opposing evidence.",
        "molecule_ids": ["Molecule 1"],
        "record_ids": ["Record 1-1"],
        "evidence_role": "contradictory",
    })
    assert validation["content_validator"](content) == []


@pytest.mark.parametrize("task,profile,prediction", [
    ("bbb_martins", "meaningful_cns_access_v1", "pass"),
    ("bioavailability_ma", "f20_evidence_calibrated_v2", "high"),
    ("skin_reaction", "sensitization_aligned_v2", "risk"),
    ("ames", "ames_gold_v1", "positive"),
    ("dili", "dili_gold_v1", "dili_risk"),
    ("carcinogens", "carcinogens_gold_v1", "positive"),
])
def test_six_task_successor_renders_each_task(
    task: str, profile: str, prediction: str,
) -> None:
    messages, metadata = build_flat_context_request(
        _flat_context_retrieval(),
        task_id=task,
        task_prompt_profile=profile,
        layout="level-grouped",
        reranking="assay-transfer-contrastive",
        query_prior={"endpoint_prior": "mixed_or_unclear"},
        prompt_version=CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    )
    validation = flat_context_validation(
        task,
        task_prompt_profile=profile,
        prompt_version=CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
        reference_index=metadata["reasoning_reference_index"],
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert prediction in validation["allowed_values"]["final_prediction"]


@pytest.mark.parametrize("task,profile,native", [
    ("bbb_martins", "meaningful_cns_access_v1", "pass"),
    ("bioavailability_ma", "f20_evidence_calibrated_v2", "high"),
    ("skin_reaction", "skin_sensitization_contact_allergy.v2", "risk"),
    ("ames", "ames_bacterial_reverse_mutation.v1", "positive"),
    ("dili", "dili_conditioned_or_source_molecule_outcome.v1", "dili_risk"),
    ("carcinogens", "carcinogens_starling_only_five_organism_groups.v1", "positive"),
])
def test_upstream_components_render_portable_schema_and_map_native_label(
    task: str, profile: str, native: str,
) -> None:
    messages, metadata = build_flat_context_request(
        _flat_context_retrieval(),
        task_id=task,
        task_prompt_profile=profile,
        layout="level-grouped",
        reranking="assay-transfer-contrastive",
        query_prior={
            "endpoint_prior": "mixed_or_unclear",
            "property_drivers": ["size", "charge"],
            "exact_chembl_evidence_assessment": "omit",
        },
        prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    )
    validation = flat_context_validation(
        task,
        task_prompt_profile=profile,
        prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        reference_index=metadata["reasoning_reference_index"],
    )
    mapped, digest = native_flat_prediction(
        task, "pass", prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION
    )

    assert validation["allowed_values"]["final_prediction"] == {"pass", "fail"}
    assert mapped == native
    assert len(digest) == 64
    assert "property drivers: [\"size\", \"charge\"]" in messages[1]["content"]
    assert "exact chembl evidence assessment" not in messages[1]["content"]
    assert all("\n\n\n" not in message["content"] for message in messages)
    assert "\n\n- " not in messages[0]["content"]


@pytest.mark.parametrize("task,profile,positive,negative,native_positive,native_negative", [
    ("dili", "dili_conditioned_or_source_molecule_outcome.v1", "dili_risk", "no_dili_risk", "dili_risk", "no_dili_risk"),
    ("ames", "ames_bacterial_reverse_mutation.v1", "positive", "negative", "positive", "negative"),
    ("skin_reaction", "skin_sensitization_contact_allergy.v2", "risk", "no_risk", "risk", "no_risk"),
    ("carcinogens", "carcinogens_starling_only_five_organism_groups.v1", "positive", "negative", "positive", "negative"),
])
def test_safety_v3_uses_endpoint_prediction_labels(
    task: str, profile: str, positive: str, negative: str,
    native_positive: str, native_negative: str,
) -> None:
    version = CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION
    messages, metadata = build_flat_context_request(
        _flat_context_retrieval(), task_id=task, task_prompt_profile=profile,
        layout="level-grouped", reranking="assay-transfer-contrastive",
        query_prior=None, prompt_version=version,
    )
    validation = flat_context_validation(
        task, task_prompt_profile=profile, prompt_version=version,
        reference_index=metadata["reasoning_reference_index"],
    )

    assert validation["allowed_values"]["final_prediction"] == {positive, negative}
    assert f'"final_prediction": "{positive} | {negative}"' in messages[1]["content"]
    assert native_flat_prediction(task, positive, prompt_version=version)[0] == native_positive
    assert native_flat_prediction(task, negative, prompt_version=version)[0] == native_negative
    assert flat_context_validation(
        task, task_prompt_profile=profile,
        prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    )["allowed_values"]["final_prediction"] == {"pass", "fail"}


def test_dili_v3_labels_reach_binary_scorer() -> None:
    version = CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION
    allowed = flat_context_validation(
        "dili", task_prompt_profile="dili_conditioned_or_source_molecule_outcome.v1",
        prompt_version=version,
    )["allowed_values"]
    assert response_validation_errors(
        {"content": {"final_prediction": "pass"}}, allowed_values=allowed,
    ) == ["invalid_value:final_prediction"]
    for prediction, expected in (("dili_risk", 1), ("no_dili_risk", 0)):
        assert response_validation_errors(
            {"content": {"final_prediction": prediction}}, allowed_values=allowed,
        ) == []
        native, _ = native_flat_prediction("dili", prediction, prompt_version=version)
        assert prediction_to_label(DILI_CONFIG, native) == expected


def test_v5_claims_allow_unbounded_molecule_or_record_citations() -> None:
    references = [
        {"visible_id": f"Molecule {index}", "unit_kind": "molecule"}
        for index in range(1, 13)
    ] + [
        {"visible_id": f"Record 1-{index}", "unit_kind": "record"}
        for index in range(1, 13)
    ]
    validation = flat_context_validation(
        "bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
        prompt_version=CONTEXT_V5_PROMPT_VERSION,
        reference_index=references,
    )
    content = {
        "claims": [
            {
                "claim": "Many records support the decision.",
                "molecule_ids": [],
                "record_ids": [f"Record 1-{index}" for index in range(1, 13)],
                "evidence_role": "supportive",
            },
            {
                "claim": "A molecule-level signal is contradictory.",
                "molecule_ids": ["Molecule 12"],
                "record_ids": [],
                "evidence_role": "contradictory",
            },
        ],
        "final_prediction": "fail",
    }
    assert validation["content_validator"](content) == []
    assert derive_flat_claim_evidence(content) == {
        "supportive_molecule_ids": [],
        "supportive_record_ids": [f"Record 1-{index}" for index in range(1, 13)],
        "contradictory_molecule_ids": ["Molecule 12"],
        "contradictory_record_ids": [],
    }


def test_v6_oral_uses_high_low_without_changing_v5() -> None:
    v6 = flat_context_validation(
        "bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        prompt_version=CONTEXT_V6_PROMPT_VERSION,
    )
    v5 = flat_context_validation(
        "bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        prompt_version=CONTEXT_V5_PROMPT_VERSION,
    )
    assert v6["allowed_values"]["final_prediction"] == {"high", "low"}
    assert v5["allowed_values"]["final_prediction"] == {"pass", "fail"}


def test_v8_keeps_transfer_likelihood_on_its_context_record() -> None:
    retrieval = _flat_context_retrieval()
    rows = retrieval["groups"][0]["neighbors"][0]["evidence_rows"]
    rows[0]["selection_provenance"].update(
        assay_transfer_score=0.9819,
        selected_condition="release_profile=modified_release",
    )
    rows[1]["selection_provenance"].update(
        level="L1",
        assay_transfer_score=0.9766,
        selected_condition="no_reported_external_condition",
    )

    with pytest.raises(ValueError, match="Conflicting molecule-scoped transfer_likelihood"):
        build_flat_context_request(
            retrieval,
            task_id="bioavailability_ma",
            task_prompt_profile="f20_evidence_calibrated_v2",
            layout="global",
            reranking="assay-transfer-contrastive",
            query_prior=None,
            prompt_version=CONTEXT_V5_PROMPT_VERSION,
        )

    messages, _ = build_flat_context_request(
        retrieval,
        task_id="bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        layout="global",
        reranking="assay-transfer-contrastive",
        query_prior=None,
        prompt_version=CONTEXT_V8_PROMPT_VERSION,
    )
    visible = messages[1]["content"]
    assert "Evidence condition: release_profile=modified_release\nTransfer likelihood: 0.9819" in visible
    assert "Evidence condition: no_reported_external_condition\nTransfer likelihood: 0.9766" in visible
    assert "SMILES: N\nTransfer likelihood:" not in visible


def test_v7_oral_requires_only_high_low_final_prediction() -> None:
    messages, metadata = build_flat_context_request(
        _flat_context_retrieval(),
        task_id="bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        layout="level-grouped",
        reranking="assay-transfer-contrastive",
        query_prior=None,
        prompt_version=CONTEXT_V7_PROMPT_VERSION,
    )
    validation = flat_context_validation(
        "bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        prompt_version=CONTEXT_V7_PROMPT_VERSION,
        reference_index=metadata["reasoning_reference_index"],
    )
    assert validation["required_fields"] == ("final_prediction",)
    assert validation["allowed_values"]["final_prediction"] == {"high", "low"}
    assert validation["content_validator"]({"final_prediction": "high"}) == []
    assert validation["content_validator"]({
        "claims": [], "final_prediction": "high"
    }) == ["invalid_output_fields"]
    assert '"final_prediction": "high | low"' in messages[1]["content"]
    assert '"claims"' not in messages[1]["content"]
    assert "Return only `final_prediction` and no other fields." in messages[0]["content"]

    bbb = flat_context_validation(
        "bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
        prompt_version=CONTEXT_V7_PROMPT_VERSION,
    )
    assert bbb["required_fields"] == ("final_prediction",)
    assert bbb["allowed_values"]["final_prediction"] == {"pass", "fail"}










def test_archived_assay_reranker_requires_legacy_opt_in() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            BBB_CONFIG,
            [
                "--retrieval-strategy",
                "assay_transfer_tool",
                "--retrieval-source",
                "starling",
            ],
        )


def test_flat_prompt_rejects_a_different_task_profile() -> None:
    args = _parse_args(
        BBB_CONFIG,
        [
            "--experiment-mode",
            "full_flat",
            "--flat-prompt-version",
            PROMPT_VERSION,
            "--bbb-prompt-profile",
            "meaningful_cns_adjudication_v2",
        ],
    )
    with pytest.raises(SystemExit, match="Retired prompt requires frozen request replay"):
        _validate_flat_prompt(BBB_CONFIG, args)


def test_flat_prompt_provenance_binds_assets_task_profile_and_variant() -> None:
    common = {
        "task_id": "bioavailability_ma",
        "task_prompt_profile": "f20_evidence_calibrated_v2",
    }
    morgan = flat_prompt_provenance(
        **common,
        retrieval_strategy="morgan_fingerprint",
    )
    assay = flat_prompt_provenance(
        **common,
        retrieval_strategy="assay_transfer_tool",
    )

    assert morgan["prompt_version"] == PROMPT_VERSION
    assert morgan["upstream_commit"] == "e56a79f371d1029201eff29e348d30e7beb94d5c"
    assert morgan["task_prompt_profile"] == common["task_prompt_profile"]
    assert morgan["variant"] == MORGAN_VARIANT
    assert assay["variant"] == ASSAY_TRANSFER_VARIANT
    assert morgan["asset_manifest_sha256"] == assay["asset_manifest_sha256"]
    assert morgan["contract_sha256"] != assay["contract_sha256"]
    assert len(morgan["upstream_sources"]) == 2


def test_flat_group_reuse_requires_the_same_prompt_contract(tmp_path: Path) -> None:
    target = flat_prompt_provenance(
        "bbb_martins",
        retrieval_strategy="morgan_fingerprint",
        task_prompt_profile="meaningful_cns_access_v1",
    )
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps({"group_prompt_provenance": {"prompt_version": "other"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reuse source provenance mismatch"):
        _validate_group_prompt_contract(
            tmp_path / "target",
            target,
            source_batches=(str(source),),
        )

    manifest_path.write_text(
        json.dumps({"group_prompt_provenance": target}),
        encoding="utf-8",
    )
    _validate_group_prompt_contract(
        tmp_path / "target",
        target,
        source_batches=(str(source),),
    )




def _selected_payload(record_id: str, level: str, family: str) -> dict:
    return {
        "record_id": record_id,
        "source_row_uid": f"uid-{record_id}",
        "task_id": "bbb_martins",
        "source_id": "toy",
        "progressive_level": level,
        "family_key": family,
        "canonical_smiles": "CCN",
        "source_canonical_smiles": "CCN",
        "source_fields": {
            "endpoint_name": "brain exposure",
            "measurement_text": "2.0",
            "extra_details": "internal-only duplicate text",
        },
        "source_contract": {
            "contract_version": "source_column_contract.v1",
            "source_or_simply_cleaned": {
                "endpoint_name": True,
                "measurement_text": True,
                "extra_details": True,
            },
        },
        "source_projection": "library_source_contract.v2",
        "measurement_kind": "continuous",
    }


def test_cache_matched_flat_adapter_groups_without_reselecting_or_leaking_pool() -> None:
    l1 = {
        "reference_molecule_id": "REF",
        "canonical_smiles": "CCN",
        "ranking_method": "joint",
        "morgan_similarity": 0.7,
        "transfer_likelihood": 0.8,
        "morgan_top5_rank": 1,
        "assay_transfer_top5_rank": "not_selected_in_top5",
        "l1_records": [{
            "record_id": "r1", "reference_molecule_id": "REF",
            "morgan_similarity": 0.7, "ranking_method": "joint",
            "payload": _selected_payload("r1", "L1", "direct"),
        }],
    }
    later = {
        "L2": {"records": [
            {
                "record_id": "r1", "reference_molecule_id": "REF",
                "morgan_similarity": 0.7, "ranking_method": "assay_transfer",
                "transfer_likelihood": 0.9, "payload": _selected_payload("r1", "L1", "direct"),
            },
            {
                "record_id": "r2", "reference_molecule_id": "REF",
                "morgan_similarity": 0.7, "ranking_method": "assay_transfer",
                "transfer_likelihood": 0.9, "payload": _selected_payload("r2", "L2", "passive"),
            },
        ]}
    }
    retrieval = cache_matched_flat_retrieval(
        "q", "CCO", [l1], later, task="bbb_martins", reranking="joint",
        query_audit={"L1": {"selected_records": 1}, "L2": {"selected_records": 1}},
    )
    neighbors = retrieval["groups"][0]["neighbors"]
    assert len(neighbors) == 1
    rows = [row["minimal_evidence"] for row in neighbors[0]["evidence_rows"]]
    assert [row["provenance"]["retrieval"]["record_id"] for row in rows] == ["r1", "r2"]
    assert rows[0]["provenance"]["retrieval"] == {
        "record_id": "r1", "level": "L1", "evidence_family": "direct",
        "ranking_method": "joint", "morgan_similarity": 0.7,
        "assay_transfer_score": 0.8, "morgan_panel_rank": 1,
        "assay_transfer_panel_rank": "not_selected_in_top5",
    }
    assert rows[1]["provenance"]["retrieval"]["ranking_method"] == "assay-transfer"
    assert "extra_details" not in json.dumps(rows)
    assert "record_pool" not in json.dumps(retrieval)


def test_cached_query_identity_allows_missing_inchi_key() -> None:
    molecule = {
        "reference_molecule_id": "REF",
        "ranking_method": "assay_transfer",
        "transfer_likelihood": 0.8,
        "l1_records": [{
            "record_id": "r1", "reference_molecule_id": "REF",
            "morgan_similarity": 0.7, "assay_rank": 1,
            "ranking_method": "assay_transfer", "transfer_likelihood": 0.8,
            "payload": _selected_payload("r1", "L1", "direct"),
        }],
    }

    retrieval = cache_matched_flat_retrieval(
        "q", "[*]CC", [molecule], {}, task="bbb_martins",
        reranking="assay-transfer", query_audit={"L1": {}},
        query_identity={"parent_smiles": "[*]CC", "parent_id": ""},
    )

    assert retrieval["query"]["canonical_smiles"] == "[*]CC"
    assert retrieval["query"]["standard_inchi_key"] == ""


def test_flat_shared_parent_selection_preserves_distinct_physical_uids() -> None:
    first = _selected_payload("duplicate", "L2", "passive")
    second = _selected_payload("duplicate", "L2", "passive")
    first["source_row_uid"] = "uid-one"
    second["source_row_uid"] = "uid-two"
    rows = [
        {
            "record_id": "duplicate", "reference_molecule_id": "REF",
            "morgan_similarity": 0.7, "ranking_method": "assay_transfer",
            "transfer_likelihood": 0.8, "payload": payload,
        }
        for payload in (first, second)
    ]

    retrieval = cache_matched_flat_retrieval(
        "q", "CCO", [], {"L2": {"records": rows}}, task="bbb_martins",
        reranking="assay-transfer", query_audit={},
        preserve_physical_records=True,
    )

    assert [row["evidence_id"] for row in retrieval["groups"][0]["neighbors"][0]["evidence_rows"]] == [
        "uid-one", "uid-two",
    ]


def test_flat_shared_parent_policy_does_not_apply_level_record_limit() -> None:
    rows = [
        {"record_id": f"r{i}", "reference_molecule_id": "REF", "payload": _selected_payload(f"r{i}", "L2", "passive")}
        for i in range(3)
    ]
    args = SimpleNamespace(
        assay_transfer_cache=Path("cache.yaml"), task="bbb_martins", benchmark="gold",
        evaluation_subset="valid", reranking="assay-transfer", max_level=2,
        prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        l1_molecules=1, l1_records_per_molecule=1,
        record_limits_by_level={"L2": 1}, ranking_tie_seed=0,
        l1_min_contrast=0, morgan_primary_parent_width=100,
    )
    policy = {"stages": {"L1": "morgan", "L2": "assay_transfer"}, "all_later_records": True}
    audit = {
        "query_audits": {"q": {"L1": {"selected_records": 0}}},
        "query_identities": {}, "neighbor_identity_policy": "parent_disjoint",
        "neighbor_identity_policy_by_level": {"L1": "scaffold_disjoint", "L2": "parent_disjoint"},
        "selection_policy": "ranked_uid_retrieval.v1", "cache_content_ids": {},
    }

    _, later, receipt = flat._load_flat_context_candidates(
        args, {"q": "CCO"}, load_cache_policy=lambda *args, **kwargs: policy,
        load_candidates=lambda *args, **kwargs: ({"q": []}, {"q": {"L2": {"records": rows}}}, audit),
    )

    assert len(later["q"]["L2"]["records"]) == 3
    assert receipt["contract"]["all_later_records"] is True
    assert receipt["neighbor_identity_policy_by_level"]["L1"] == "scaffold_disjoint"


@pytest.mark.parametrize("prompt_version", [
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
])
def test_flat_partial_snapshot_accepts_empty_assay_level(prompt_version: str) -> None:
    args = SimpleNamespace(
        assay_transfer_cache=Path("cache.yaml"), task="ames", benchmark="gold",
        evaluation_subset="valid", reranking="assay-transfer", max_level=2,
        prompt_version=prompt_version,
        l1_molecules=1, l1_records_per_molecule=1,
        record_limits_by_level={"L2": 3}, ranking_tie_seed=0,
        l1_min_contrast=0, morgan_primary_parent_width=100,
    )
    policy = {"stages": {"L1": "morgan", "L2": "assay_transfer"}}
    audit = {
        "query_audits": {"q": {"L1": {"selected_records": 0}}},
        "query_identities": {}, "neighbor_identity_policy": "parent_disjoint",
        "neighbor_identity_policy_by_level": {"L2": "parent_disjoint"},
        "selection_policy": "ranked_uid_retrieval.v1", "cache_content_ids": {},
    }
    _, later, _ = flat._load_flat_context_candidates(
        args, {"q": "CCO"}, load_cache_policy=lambda *args, **kwargs: policy,
        load_candidates=lambda *args, **kwargs: (
            {"q": []}, {"q": {"L2": {"records": [], "allow_shortfall": True}}}, audit
        ),
    )
    assert later["q"]["L2"]["records"] == []
    assert later["q"]["L2"]["shortfall"] == 3


def test_cache_metadata_keeps_safety_l7_visible() -> None:
    from predict.retrieval.assay_reranking.artifact_bundle import DEFAULT_CACHE_SET, cache_level_limit
    assert cache_level_limit(DEFAULT_CACHE_SET, "carcinogens", "gold") == 7


def test_l1_context_cards_with_one_parent_remain_separate() -> None:
    molecules = []
    for context_id, record_id, score in (("c1", "r1", 0.8), ("c2", "r2", 0.2)):
        molecules.append({
            "reference_molecule_id": "REF",
            "canonical_smiles": "CCN",
            "context_card_id": context_id,
            "ranking_method": "assay_transfer",
            "morgan_similarity": 0.7,
            "transfer_likelihood": score,
            "l1_records": [{
                "record_id": record_id,
                "reference_molecule_id": "REF",
                "morgan_similarity": 0.7,
                "transfer_likelihood": score,
                "ranking_method": "assay_transfer",
                "payload": _selected_payload(record_id, "L1", "direct"),
            }],
        })

    retrieval = cache_matched_flat_retrieval(
        "q", "CCO", molecules, {}, task="bbb_martins",
        reranking="assay-transfer", query_audit={},
        prompt_version=CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    )

    assert len(retrieval["groups"][0]["neighbors"]) == 2


def test_flat_hybrid_partial_level_hides_assay_score() -> None:
    later = {"L2": {"records": [{
        "record_id": "r2", "reference_molecule_id": "REF",
        "morgan_similarity": 0.712345,
        "ranking_method": "assay_transfer_partial_hidden",
        "payload": _selected_payload("r2", "L2", "passive"),
    }]}}

    retrieval = cache_matched_flat_retrieval(
        "q", "CCO", [], later, task="bbb_martins",
        reranking="assay-transfer", query_audit={},
        prompt_version=CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    )

    row = retrieval["groups"][0]["neighbors"][0]["evidence_rows"][0]
    assert row["selection_provenance"]["ranking_method"] == "assay-transfer-partial-hidden"
    assert row["selection_provenance"]["morgan_similarity"] == 0.7123
    assert "assay_transfer_score" not in row["selection_provenance"]
    assert "transfer_likelihood" not in row["prompt_evidence"]




def test_flat_v2_rounds_similarity_after_assigning_its_bucket() -> None:
    payload = _selected_payload("r1", "L1", "direct")
    molecule = {
        "reference_molecule_id": "REF",
        "canonical_smiles": "CCN",
        "ranking_method": "morgan",
        "morgan_similarity": 0.79996,
        "l1_records": [{
            "record_id": "r1",
            "reference_molecule_id": "REF",
            "morgan_similarity": 0.79996,
            "ranking_method": "morgan",
            "payload": payload,
        }],
    }
    retrieval = cache_matched_flat_retrieval(
        "q", "CCO", [molecule], {}, task="bbb_martins", reranking="morgan",
        query_audit={}, prompt_version=JOSEPH_PROMPT_VERSION,
    )
    neighbor = retrieval["groups"][0]["neighbors"][0]
    assert neighbor["similarity"] == 0.8
    assert neighbor["similarity_bucket"] == "moderate_analog"
    assert neighbor["_prompt_scores"]["morgan_similarity"] == 0.8


def test_flat_v2_uses_cached_query_identity_without_normalizing(monkeypatch) -> None:
    payload = _selected_payload("r1", "L1", "direct")
    molecule = {
        "reference_molecule_id": "REF",
        "canonical_smiles": "CCN",
        "ranking_method": "morgan",
        "morgan_similarity": 0.5,
        "l1_records": [{
            "record_id": "r1",
            "reference_molecule_id": "REF",
            "morgan_similarity": 0.5,
            "ranking_method": "morgan",
            "payload": payload,
        }],
    }
    monkeypatch.setattr(
        flat,
        "normalize_molecule_identity",
        lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )
    retrieval = cache_matched_flat_retrieval(
        "q",
        "raw-query",
        [molecule],
        {},
        task="bbb_martins",
        reranking="morgan",
        query_audit={},
        query_identity={"parent_id": "QUERY", "parent_smiles": "cached-query"},
        prompt_version=JOSEPH_PROMPT_VERSION,
    )
    assert retrieval["query"]["canonical_smiles"] == "cached-query"
    assert retrieval["query"]["standard_inchi_key"] == "QUERY"


def test_flat_v2_claims_allow_empty_and_reject_unknown_cards() -> None:
    group = {
        "neighbors": [{
            "evidence_rows": [
                {"prompt_evidence": {"card_id": "C01"}},
                {"prompt_evidence": {"card_id": "C02"}},
            ]
        }]
    }
    validation = flat_group_validation(
        "bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
        prompt_version=JOSEPH_PROMPT_VERSION,
        group=group,
    )
    validator = validation["content_validator"]
    assert validator({"claims": []}) == []
    assert validator({"claims": [{"claim": "Measured access", "card_ids": ["C01"]}]}) == []
    assert validator({"claims": [{"claim": "Unknown", "card_ids": ["C99"]}]}) == [
        "unknown_card_ids:0:C99"
    ]
    assert validation["forbidden_field_names"] == ("key_evidence", "tool_summary")


def test_public_flat_defaults_to_final_and_rejects_retired_harnesses(monkeypatch, tmp_path) -> None:
    observed = {}
    def prepare(args):
        observed.update(prompt=args.prompt_version, cache=args.assay_transfer_cache)
        return tmp_path, {"indices": [0]}
    monkeypatch.setattr(flat, "_materialize_cache_matched_retrievals", prepare)
    assert flat.run(["--task", "bbb_martins", "--query-prior", "none", "--prepare-only",
                     "--run-name", "default-prompt", "--output-root", str(tmp_path)]) == 0
    assert observed["prompt"] == flat.FINAL_FULL_FLAT_PROMPT_VERSION
    assert observed["cache"].name == "catalog.json"
    with pytest.raises(SystemExit):
        flat.run(["--harness-version", "tianang-flat-v1", "--task", "bbb_martins"])


def test_joseph_flat_prompt_rejects_skin() -> None:
    args = _parse_args(
        SKIN_CONFIG,
        [
            "--experiment-mode", "full_flat",
            "--flat-prompt-version", JOSEPH_PROMPT_VERSION,
            "--flat-reranking", "morgan",
            "--retrieval-replay-source-batch", "/tmp/frozen",
        ],
    )
    with pytest.raises(SystemExit, match="Retired prompt requires frozen request replay"):
        _validate_flat_prompt(SKIN_CONFIG, args)


def test_score_only_upstream_flat_prompt_accepts_skin() -> None:
    args = _parse_args(
        SKIN_CONFIG,
        [
            "--experiment-mode", "full_flat",
            "--flat-prompt-version", flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
            "--flat-reranking", "assay-transfer-contrastive",
            "--retrieval-replay-source-batch", "/tmp/frozen",
            "--flat-selection-manifest", "/tmp/selection.json",
            "--disable-flat-tools",
            "--reasoning-effort", "high",
            "--flat-query-prior", "none",
            "--flat-request-replay-root", "/tmp/frozen/requests",
            SKIN_CONFIG.prompt_profile_option, "skin_sensitization_contact_allergy.v2",
        ],
    )
    _validate_flat_prompt(SKIN_CONFIG, args)


@pytest.mark.parametrize(
    ("flag", "public", "stored"),
    [
        ("--record_pool", "all", "all"),
    ],
)
def test_joseph_cli_maps_public_pool_names_only(
    monkeypatch, tmp_path: Path, flag: str, public: str, stored: str
) -> None:
    observed = {}

    def materialize(args):
        observed.update(public=args.record_pool, stored=flat.RECORD_POOLS[args.record_pool])
        return tmp_path, {"status": "prepared"}

    monkeypatch.setattr(flat, "_materialize_cache_matched_retrievals", materialize)
    assert flat.run([
            "--prompt", flat.FINAL_FULL_FLAT_PROMPT_VERSION,
            "--task", "bioavailability_ma", "--query-prior", "none", "--prepare-only",
            "--run-name", "pool-check", "--output-root", str(tmp_path)
    ]) == 0
    assert observed == {"public": public, "stored": stored}


@pytest.mark.parametrize("explicit", [False, True])
def test_flat_launcher_forwards_the_selected_provider_config(monkeypatch, tmp_path, explicit):
    selected = flat.DEFAULT_PROVIDER_CONFIG.with_name("current_endpoints.json") if explicit else flat.DEFAULT_PROVIDER_CONFIG
    observed = {}
    monkeypatch.setattr(flat, "_materialize_cache_matched_retrievals", lambda args: (tmp_path, {}))
    monkeypatch.setattr(flat, "mode_main", lambda mode, argv, **kwargs: observed.update(argv=argv, **kwargs) or 0)
    argv = ["--task", "bbb_martins", "--query-prior", "none", "--max-tokens", "262144",
            "--run-name", "provider-check", "--output-root", str(tmp_path)]
    if explicit:
        argv.extend(["--provider-pool-config", str(selected)])
    assert flat.run(argv) == 0
    assert observed["launch_command"][3:] == argv
    forwarded = observed["argv"]
    assert forwarded[forwarded.index("--provider-pool-config") + 1] == str(selected)


@pytest.mark.parametrize("pool", ["assay-transfer-trained", "all_transfer_eligible"])
def test_active_joseph_cli_rejects_legacy_pools(pool: str) -> None:
    with pytest.raises(SystemExit):
        flat.run([
            "--task", "bioavailability_ma", "--record_pool", pool,
            "--prepare-only",
        ])


def test_joseph_cli_rejects_legacy_retrieval_flags() -> None:
    with pytest.raises(SystemExit):
        flat.run([
            "--task", "bbb_martins", "--retrieval-strategy", "morgan_fingerprint"
        ])


def test_joseph_batch_binds_selection_manifest(tmp_path: Path) -> None:
    input_path = Path(
        "data/gold_labels/BBB_Martins/v1/scaffold/valid_molecule_condition_labels.jsonl"
    ).resolve()
    source = tmp_path / "cache_matched_retrieval"
    source.mkdir()
    selection = {
        "harness_version": "joseph-flat-v2",
        "prompt_version": JOSEPH_PROMPT_VERSION,
        "task": "bbb_martins",
        "evaluation_subset": "valid",
        "reranking": "joint",
        "record_pool": "assay-transfer-trained",
        "cache_pool": "tool-accepted",
        "indices": [0],
        "input_jsonl": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "selection_contract_sha256": "contract-hash",
        "evidence_projection": EVIDENCE_PROJECTION,
        "extra_details_policy": EXTRA_DETAILS_POLICY,
        "molecule_name_visible": False,
        "group_tools_enabled": False,
    }
    from predict.harnesses.branches.manifests import artifact_name
    selection["artifacts"] = {"input_jsonl": artifact_name(input_path)}
    selection["queries"] = {"0": json.loads(input_path.read_text().splitlines()[0])["benchmark_row_id"]}
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(selection), encoding="utf-8")
    args = _parse_args(
        BBB_CONFIG,
        [
            "--input-jsonl", str(input_path),
            "--indices", "0",
            "--experiment-mode", "full_flat",
            "--flat-prompt-version", JOSEPH_PROMPT_VERSION,
            "--flat-reranking", "joint",
            "--flat-selection-manifest", str(manifest_path),
            "--retrieval-replay-source-batch", str(source),
            "--disable-flat-tools",
        ],
    )
    assert _validated_flat_selection_manifest(BBB_CONFIG, args, [0]) == selection
    selection["reranking"] = "morgan"
    manifest_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(SystemExit, match="manifest mismatch"):
        _validated_flat_selection_manifest(BBB_CONFIG, args, [0])


def test_no_tool_provider_pool_uses_plain_group_call(monkeypatch) -> None:
    class Client:
        enable_group_tools = OpenAIProviderPool.enable_group_tools

        def chat_json(self, messages):
            return {"messages": messages}

    monkeypatch.setattr(
        calls,
        "call_with_json_validation",
        lambda call, messages, **kwargs: call(messages),
    )
    messages = [{"role": "user", "content": "{}"}]
    result = calls.call_group_branch(
        Client(), messages, group={"neighbors": []}, tools=[]
    )
    assert result["messages"] == messages


@pytest.mark.parametrize("prompt_version", [
    flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
    flat.TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
])
def test_score_only_successor_exposes_cached_morgan_similarity(prompt_version: str) -> None:
    group = {"neighbors": [{
        "canonical_smiles": "CCN", "standard_inchi_key": "parent",
        "similarity": 0.712345,
        "similarity_metric": "Morgan radius=2 bits=2048 Tanimoto",
        "evidence_rows": [{
            "evidence_id": "uid", "prompt_evidence": {"card_id": "C01"},
            "selection_provenance": {
                "level": "L1", "evidence_family": "direct_brain_exposure",
                "assay_transfer_score": 0.8235,
            },
        }],
    }]}
    old, _, _ = flat._flat_context_sections(
        group, layout="level-grouped",
        prompt_version=flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    )
    new, _, _ = flat._flat_context_sections(
        group, layout="level-grouped",
        prompt_version=prompt_version,
    )
    assert "morgan_similarity" not in old[0]["molecules"][0]
    assert new[0]["molecules"][0]["morgan_similarity"] == 0.7123
    assert new[0]["molecules"][0]["transfer_likelihood"] == 0.8235
    del group["neighbors"][0]["evidence_rows"][0]["selection_provenance"]["assay_transfer_score"]
    morgan_only, _, _ = flat._flat_context_sections(
        group, layout="level-grouped", prompt_version=prompt_version,
    )
    assert morgan_only[0]["molecules"][0]["morgan_similarity"] == 0.7123
    assert "transfer_likelihood" not in morgan_only[0]["molecules"][0]
    group["neighbors"][0]["similarity_metric"] = "unknown"
    with pytest.raises(ValueError, match="requires Morgan"):
        flat._flat_context_sections(
            group, layout="level-grouped",
            prompt_version=prompt_version,
        )


def test_l1_member_shared_by_two_contexts_keeps_both_memberships():
    payload = _selected_payload("external-123", "L1", "direct")
    payload["source_row_uid"] = "physical-one"
    molecules = [dict(context_card_id=context, _diagnostic_context_id=context,
                      reference_molecule_id="REF", morgan_similarity=0.7,
                      ranking_method="morgan", l1_records=[dict(
                          record_id="external-123", reference_molecule_id="REF",
                          morgan_similarity=0.7, ranking_method="morgan", payload=payload)] * 2)
                 for context in ("null-card", "conditioned-card")]
    result = cache_matched_flat_retrieval(
        "q", "CCO", molecules, {}, task="bbb_martins", reranking="morgan",
        query_audit={}, preserve_physical_records=True)
    neighbors = result["groups"][0]["neighbors"]
    assert [len(n["evidence_rows"]) for n in neighbors] == [1, 1]
    assert [n["evidence_rows"][0]["minimal_evidence"]["provenance"]["retrieval"]["diagnostic_context_id"]
            for n in neighbors] == ["null-card", "conditioned-card"]


def test_retired_assay_contrastive_execution_requires_frozen_request_replay(capsys):
    with pytest.raises(SystemExit):
        _parse_args(SKIN_CONFIG, [
            '--experiment-mode', 'full_flat', '--flat-prompt-version', 'final_full_flat',
            '--flat-reranking', 'assay-transfer-contrastive',
        ])
    assert 'contrastive selection is retired' in capsys.readouterr().err
