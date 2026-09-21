from __future__ import annotations

import importlib
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from predict.harnesses.branches.runner import (
    _parse_args,
    _validate_assay_transfer_scores,
    _validate_flat_prompt,
    _validated_flat_selection_manifest,
    _validate_group_prompt_contract,
)
from predict.harnesses.branches.flat import (
    ASSAY_TRANSFER_VARIANT,
    CONTEXT_V4_HARNESS_VERSION,
    CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_V5_HARNESS_VERSION,
    CONTEXT_V5_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
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


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _payload() -> dict:
    return deepcopy(_fixture()["input_payload"])


def _render(case: dict) -> list[dict[str, str]]:
    fixture = _fixture()
    return render_flat_group_messages(
        deepcopy(fixture["input_payload"]),
        group=deepcopy(fixture["group_context"]),
        task_id=case["task_id"],
        task_prompt_profile=case["task_prompt_profile"],
        group_tools_enabled=False,
        retrieval_strategy=case["retrieval_strategy"],
    )


def test_reviewed_flat_prompt_messages() -> None:
    fixture = _fixture()
    assert fixture["prompt_version"] == PROMPT_VERSION
    for case in fixture["cases"]:
        messages = _render(case)
        rendered = [
            messages[0],
            {"role": "user", "content": json.loads(messages[1]["content"])},
        ]
        assert rendered == case["expected_messages"]


def test_reviewed_joseph_flat_prompt_messages() -> None:
    fixture = json.loads(JOSEPH_FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        messages = render_flat_group_messages(
            deepcopy(fixture["input_payload"]),
            group=deepcopy(fixture["group_context"]),
            task_id=case["task_id"],
            task_prompt_profile=case["task_prompt_profile"],
            group_tools_enabled=False,
            prompt_version=fixture["prompt_version"],
            retrieval_strategy="morgan_fingerprint",
            flat_reranking=case["reranking"],
        )
        observed = hashlib.sha256(
            json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        assert observed == case["messages_sha256"]


def test_flat_variants_project_only_their_ranking_surface() -> None:
    source = _payload()
    common = {
        "task_id": "bbb_martins",
        "task_prompt_profile": "meaningful_cns_access_v1",
        "include_query_tool_guidance": True,
    }
    morgan = apply_flat_group_prompt(
        source,
        retrieval_strategy="morgan_fingerprint",
        **common,
    )
    assay = apply_flat_group_prompt(
        source,
        retrieval_strategy="assay_transfer_tool",
        **common,
    )

    assert source == _payload()
    assert morgan["neighbors"][0]["similarity"] == 0.72
    assert "assay_transfer_score" not in str(morgan)
    assert "similarity" not in assay["neighbors"][0]
    assert assay["neighbors"][0]["assay_transfer_score"] == 0.91


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


def test_flat_prompt_rejects_a_nonflat_group() -> None:
    payload = _payload()
    payload["group"]["group_id"] = "Tier 1.direct"
    with pytest.raises(ValueError, match="Flat.all_evidence"):
        apply_flat_group_prompt(
            payload,
            task_id="bbb_martins",
            task_prompt_profile="meaningful_cns_access_v1",
            include_query_tool_guidance=True,
            retrieval_strategy="morgan_fingerprint",
        )


def test_flat_prompt_bundle_is_complete_and_version_owned() -> None:
    manifest = prompt_asset_manifest()
    assert set(manifest["files_sha256"]) == {
        "modes/assay-transfer/user.yaml",
        "modes/morgan/user.yaml",
        "provenance.json",
        "system.jinja",
        "tasks.yaml",
    }
    assert manifest["assembly_files_sha256"].keys() == {"flat.py"}
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


def test_flat_context_harness_is_publicly_selectable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(flat, "_joseph_main", lambda argv: calls.append(argv) or 0)
    assert flat.run([
        "--harness-version", CONTEXT_V4_HARNESS_VERSION,
        "--task", "bbb_martins", "--query-prior", "none", "--prepare-only",
    ]) == 0
    assert CONTEXT_V4_HARNESS_VERSION in calls[0]


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


def test_v7_harness_is_publicly_selectable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(flat, "_joseph_main", lambda argv: calls.append(argv) or 0)
    assert flat.run([
        "--harness-version", CONTEXT_V7_HARNESS_VERSION,
        "--task", "bioavailability_ma", "--query-prior", "none", "--prepare-only",
    ]) == 0
    assert CONTEXT_V7_HARNESS_VERSION in calls[0]


def test_v8_harness_is_publicly_selectable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(flat, "_joseph_main", lambda argv: calls.append(argv) or 0)
    assert flat.run([
        "--harness-version", CONTEXT_V8_HARNESS_VERSION,
        "--task", "bbb_martins", "--query-prior", "none", "--prepare-only",
    ]) == 0
    assert CONTEXT_V8_HARNESS_VERSION in calls[0]


def test_v5_harness_is_publicly_selectable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(flat, "_joseph_main", lambda argv: calls.append(argv) or 0)
    assert flat.run([
        "--harness-version", CONTEXT_V5_HARNESS_VERSION,
        "--task", "bbb_martins", "--query-prior", "none", "--prepare-only",
    ]) == 0
    assert CONTEXT_V5_HARNESS_VERSION in calls[0]


@pytest.mark.parametrize(
    ("config", "strategy", "extra"),
    [
        (BBB_CONFIG, "morgan_fingerprint", []),
        (BIOAVAILABILITY_CONFIG, "morgan_fingerprint", []),
        (SKIN_CONFIG, "morgan_fingerprint", []),
        (
            BBB_CONFIG,
            "assay_transfer_tool",
            ["--retrieval-source", "starling", "--enable-assay-transfer-scores", "--legacy"],
        ),
        (
            BIOAVAILABILITY_CONFIG,
            "assay_transfer_tool",
            ["--retrieval-source", "starling", "--enable-assay-transfer-scores", "--legacy"],
        ),
        (
            SKIN_CONFIG,
            "assay_transfer_tool",
            ["--retrieval-source", "starling", "--enable-assay-transfer-scores", "--legacy"],
        ),
    ],
)
def test_flat_prompt_cli_uses_frozen_task_contract(
    config, strategy: str, extra: list[str]
) -> None:
    args = _parse_args(
        config,
        [
            "--experiment-mode",
            "full_flat",
            "--flat-prompt-version",
            PROMPT_VERSION,
            "--retrieval-strategy",
            strategy,
            *extra,
        ],
    )

    _validate_flat_prompt(config, args)
    _validate_assay_transfer_scores(config, args)
    assert args.group_prompt_format == "legacy"
    assert args.group_output_schema in {"", "legacy"}


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
    with pytest.raises(SystemExit, match="requires task prompt profile"):
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


@pytest.mark.parametrize(
    ("task_id", "task_prompt_profile"),
    [
        ("bbb_martins", "meaningful_cns_access_v1"),
        ("bioavailability_ma", "f20_evidence_calibrated_v2"),
        ("skin_reaction", "sensitization_aligned_v2"),
    ],
)
def test_task_pipelines_delegate_flat_rendering(
    monkeypatch, task_id: str, task_prompt_profile: str
) -> None:
    module = importlib.import_module(
        f"predict.harnesses.branches.tasks.{task_id}.pipeline"
    )
    group = {
        "group_id": "Flat.all_evidence",
        "tier": "Flat",
        "endpoint_group": "all_evidence",
        "neighbors": [],
        "identity_blind": True,
        "tools_prefetched": [],
    }
    payload = module.build_group_prompt_payload(
        {},
        group,
        prompt_profile=task_prompt_profile,
        include_query_tool_guidance=False,
    )
    expected = render_flat_group_messages(
        payload,
        group=group,
        task_id=task_id,
        task_prompt_profile=task_prompt_profile,
        group_tools_enabled=False,
        retrieval_strategy="morgan_fingerprint",
    )
    captured = {}

    def fake_call(_client, messages, **kwargs):
        captured.update(messages=messages, kwargs=kwargs)
        return {"structured_output_validation": {"valid": True}}

    monkeypatch.setattr(module, "call_group_branch", fake_call)
    module._reason_one_group(
        SimpleNamespace(enable_group_tools=False),
        {},
        group,
        prompt_profile=task_prompt_profile,
        flat_prompt_version=PROMPT_VERSION,
        retrieval_strategy="morgan_fingerprint",
    )

    assert captured["messages"] == expected
    assert captured["kwargs"]["required_fields"] == (
        "transferability",
        "confidence",
        "reasoning_summary",
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


def test_flat_v2_renders_complete_source_semantics_without_molecule_name() -> None:
    source = _selected_payload("r1", "L1", "direct_oral")
    source["task_id"] = "bioavailability_ma"
    source["source_fields"] = {
        "assay_system": "Caco-2 monolayer",
        "endpoint_name": "intestinal permeability",
        "measurement_text": "12.3",
        "unit_text": "cm/s",
        "species": "human",
        "qualifying_conditions": "60 minutes",
        "dose": 0,
        "substrate_status": False,
        "biological_context": ["intestinal", {"region": "jejunum"}],
        "extra_details": {"fed": False, "temperature_c": 37},
        "support_text": "Measured permeability was 12.3 cm/s.",
        "molecule_name": "IDENTITY_SENTINEL",
        "pmid": "12345678",
        "confidence": 0.99,
        "smiles": "CCN",
    }
    source["source_contract"]["source_or_simply_cleaned"] = {
        name: True for name in source["source_fields"]
    }
    molecule = {
        "reference_molecule_id": "REF",
        "canonical_smiles": "CCN",
        "ranking_method": "joint",
        "morgan_similarity": 0.712345,
        "transfer_likelihood": 0.823456,
        "morgan_top5_rank": 1,
        "assay_transfer_top5_rank": "not_selected_in_top5",
        "l1_records": [{
            "record_id": "r1",
            "reference_molecule_id": "REF",
            "morgan_similarity": 0.712345,
            "ranking_method": "joint",
            "payload": source,
        }],
    }
    retrieval = cache_matched_flat_retrieval(
        "q",
        "CCO",
        [molecule],
        {},
        task="bioavailability_ma",
        reranking="joint",
        query_audit={"L1": {"selected_records": 1}},
        prompt_version=JOSEPH_PROMPT_VERSION,
    )
    group = retrieval["groups"][0]
    row = group["neighbors"][0]["evidence_rows"][0]
    accounting = row["field_accounting"]
    assert set(accounting["approved_nonempty_fields"]) == set(source["source_fields"])
    assert accounting["excluded_fields"] == {
        "molecule_name": "evidence_molecule_name",
        "pmid": "nonsemantic_metadata_or_duplicate_structure",
        "confidence": "nonsemantic_metadata_or_duplicate_structure",
        "smiles": "nonsemantic_metadata_or_duplicate_structure",
    }

    module = importlib.import_module(
        "predict.harnesses.branches.tasks.bioavailability_ma.pipeline"
    )
    payload = module.build_group_prompt_payload(
        {"canonical_smiles": "CCO"},
        group,
        prompt_profile="f20_evidence_calibrated_v2",
        include_query_tool_guidance=False,
    )
    messages = render_flat_group_messages(
        payload,
        group=group,
        task_id="bioavailability_ma",
        task_prompt_profile="f20_evidence_calibrated_v2",
        group_tools_enabled=False,
        prompt_version=JOSEPH_PROMPT_VERSION,
        retrieval_strategy="morgan_fingerprint",
        flat_reranking="joint",
    )
    visible = json.loads(messages[1]["content"])
    assert set(visible) == {
        "task", "query", "molecules", "instructions", "required_json_schema"
    }
    assert len(visible["molecules"]) == 1
    rendered_molecule = visible["molecules"][0]
    assert rendered_molecule["canonical_smiles"] == "CCN"
    assert rendered_molecule["morgan_similarity"] == 0.7123
    assert rendered_molecule["transfer_likelihood"] == 0.8235
    assert rendered_molecule["morgan_top5_rank"] == 1
    assert rendered_molecule["assay_transfer_top5_rank"] == "not_selected_in_top5"
    card = rendered_molecule["evidence_cards"][0]
    assert card == {
        "card_id": "C01",
        "assay_context": "Caco-2 monolayer",
        "endpoint": "intestinal permeability",
        "reported_value": "12.3",
        "reported_unit": "cm/s",
        "species": "human",
        "qualifying_conditions": "60 minutes",
        "experimental_details": {
            "dose": 0,
            "substrate_status": False,
            "biological_context": ["intestinal", {"region": "jejunum"}],
        },
        "extra_details": {"fed": False, "temperature_c": 37},
        "support_text": "Measured permeability was 12.3 cm/s.",
    }
    assert "IDENTITY_SENTINEL" not in messages[1]["content"]


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


def test_flat_v2_places_later_assay_score_on_its_card() -> None:
    l1_payload = _selected_payload("r1", "L1", "direct")
    later_payload = _selected_payload("r2", "L2", "passive")
    molecule = {
        "reference_molecule_id": "REF",
        "canonical_smiles": "CCN",
        "ranking_method": "assay_transfer",
        "morgan_similarity": 0.712345,
        "transfer_likelihood": 0.823456,
        "l1_records": [{
            "record_id": "r1", "reference_molecule_id": "REF",
            "morgan_similarity": 0.712345, "ranking_method": "assay_transfer",
            "payload": l1_payload,
        }],
    }
    later = {"L2": {"records": [{
        "record_id": "r2", "reference_molecule_id": "REF",
        "morgan_similarity": 0.712345, "ranking_method": "assay_transfer",
        "transfer_likelihood": 0.934567, "payload": later_payload,
    }]}}
    retrieval = cache_matched_flat_retrieval(
        "q", "CCO", [molecule], later, task="bbb_martins",
        reranking="assay-transfer", query_audit={},
        prompt_version=JOSEPH_PROMPT_VERSION,
    )
    group = retrieval["groups"][0]
    module = importlib.import_module(
        "predict.harnesses.branches.tasks.bbb_martins.pipeline"
    )
    payload = module.build_group_prompt_payload(
        {"canonical_smiles": "CCO"}, group,
        prompt_profile="meaningful_cns_access_v1",
        include_query_tool_guidance=False,
    )
    visible = apply_flat_group_prompt(
        payload,
        task_id="bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
        include_query_tool_guidance=False,
        prompt_version=JOSEPH_PROMPT_VERSION,
        retrieval_strategy="morgan_fingerprint",
        flat_reranking="assay-transfer",
    )
    rendered = visible["molecules"][0]
    assert rendered["transfer_likelihood"] == 0.8235
    assert "morgan_similarity" not in rendered
    assert "transfer_likelihood" not in rendered["evidence_cards"][0]
    assert rendered["evidence_cards"][1]["transfer_likelihood"] == 0.9346
    assert [card["card_id"] for card in rendered["evidence_cards"]] == ["C01", "C02"]


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


def test_public_flat_defaults_to_joseph_and_legacy_is_explicit(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(flat, "_joseph_main", lambda argv: calls.append((PUBLIC_HARNESS_VERSION, argv)) or 0)
    monkeypatch.setattr(flat, "mode_main", lambda mode, argv: calls.append((mode, argv)) or 0)
    assert flat.run(["--task", "bbb_martins", "--prepare-only"]) == 0
    assert calls[0][0] == PUBLIC_HARNESS_VERSION
    assert flat.run([
        "--harness-version", "tianang-flat-v1", "--task", "bbb_martins"
    ]) == 0
    assert "--flat-prompt-version" in calls[1][1]


def test_flat_v2_disables_only_group_comparison_tools(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(
        flat,
        "_materialize_cache_matched_retrievals",
        lambda _args: (tmp_path, {"status": "prepared"}),
    )
    monkeypatch.setattr(
        flat, "mode_main", lambda mode, argv: calls.append((mode, argv)) or 0
    )
    assert flat._joseph_main([
        "--harness-version", PUBLIC_HARNESS_VERSION,
        "--task", "bbb_martins",
        "--limit", "1",
    ]) == 0
    forwarded = calls[-1][1]
    assert JOSEPH_PROMPT_VERSION in forwarded
    assert "--disable-flat-tools" in forwarded

    assert flat._joseph_main([
        "--harness-version", JOSEPH_V1_HARNESS_VERSION,
        "--task", "bbb_martins",
        "--limit", "1",
    ]) == 0
    forwarded = calls[-1][1]
    assert JOSEPH_V1_PROMPT_VERSION in forwarded
    assert "--disable-flat-tools" not in forwarded


def test_full_flat_context_v5_accepts_the_published_test_cache(
    monkeypatch, tmp_path: Path
) -> None:
    observed = {}

    def materialize(args):
        observed["subset"] = args.evaluation_subset
        return tmp_path, {"status": "prepared"}

    monkeypatch.setattr(flat, "_materialize_cache_matched_retrievals", materialize)
    assert flat._joseph_main([
        "--harness-version", CONTEXT_V5_HARNESS_VERSION,
        "--task", "bbb_martins",
        "--evaluation-subset", "test",
        "--query-prior", "none",
        "--prepare-only",
    ]) == 0
    assert observed == {"subset": "test"}


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
    with pytest.raises(SystemExit, match="BBB and oral"):
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
    assert flat._joseph_main([
        "--task", "bioavailability_ma", flag, public, "--prepare-only"
    ]) == 0
    assert observed == {"public": public, "stored": stored}


@pytest.mark.parametrize("pool", ["assay-transfer-trained", "all_transfer_eligible"])
def test_active_joseph_cli_rejects_legacy_pools(pool: str) -> None:
    with pytest.raises(SystemExit):
        flat._joseph_main([
            "--task", "bioavailability_ma", "--record_pool", pool,
            "--prepare-only",
        ])


def test_joseph_cli_rejects_legacy_retrieval_flags() -> None:
    with pytest.raises(SystemExit):
        flat._joseph_main([
            "--task", "bbb_martins", "--retrieval-strategy", "morgan_fingerprint"
        ])


def test_joseph_batch_binds_selection_manifest(tmp_path: Path) -> None:
    input_path = Path(
        "data/gold_labels/BBB_Martins/v1/scaffold/valid_molecule_condition_labels.jsonl"
    ).resolve()
    source = tmp_path / "cache_matched_retrieval"
    source.mkdir()
    selection = {
        "harness_version": PUBLIC_HARNESS_VERSION,
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
