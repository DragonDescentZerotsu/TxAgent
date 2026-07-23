import copy
from pathlib import Path

import pytest

from tools.chembl_tool.common.retrieval_ablation import retrieval_prompt_hash
from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    _parse_args as _parse_batch_args,
    _single_run_command,
    _validate_assay_transfer_scores,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_prompt_policy import (
    SCORED_NEIGHBORS_POLICY_NAME,
    enable_scored_neighbors_prompt_policy,
    prepare_assay_transfer_selected_neighbors,
    validate_scored_neighbors_configuration,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch import CONFIG
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import _group_prompt_payload


def _neighbor(index: int, score: float) -> dict:
    return {
        "rank": index,
        "structural_rank": index + 4,
        "transfer_selection_rank": index,
        "transfer_selection_score": score,
        "transfer_winning_record_id": f"audit-record-{index}",
        "molecule_chembl_id": f"STARLING_{index}",
        "canonical_smiles": "CCN",
        "similarity": 0.8,
        "similarity_bucket": "close_analog",
        "evidence_rows": [
            {
                "molecule_chembl_id": f"STARLING_{index}",
                "canonical_smiles": "CCN",
                "group_id": "Observed.direct_oral_bioavailability",
                "standard_type": "Oral bioavailability",
                "standard_value": 50,
                "standard_units": "%",
            }
        ],
    }


def _retrieval(top_k: int = 5) -> dict:
    return {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "experiment": {
            "mode": "full_mechanism",
            "source": "starling",
            "retrieval_reranker": {"name": "assay_transfer"},
        },
        "coverage": {"top_k_per_group": top_k},
        "groups": [
            {
                "group_id": "Observed.direct_oral_bioavailability",
                "tier": "Observed",
                "endpoint_group": "direct_oral_bioavailability",
                "transfer_neighbor_selection": {"selection_metadata_is_llm_hidden": True},
                "neighbors": [_neighbor(index, 0.901 + index / 1000) for index in range(1, top_k + 1)],
            }
        ],
    }


@pytest.mark.parametrize("top_k", [3, 5, 7, 10])
def test_score_visibility_preserves_requested_top_k(top_k):
    validate_scored_neighbors_configuration(
        enabled=True,
        experiment_mode="full_mechanism",
        retrieval_source="starling",
        retrieval_reranker="assay_transfer",
    )
    retrieval = _retrieval(top_k)
    enable_scored_neighbors_prompt_policy(retrieval)

    assert retrieval["coverage"]["top_k_per_group"] == top_k
    assert len(retrieval["groups"][0]["neighbors"]) == top_k
    assert retrieval["experiment"]["llm_neighbor_score_policy"]["effective_top_k_per_group"] == top_k


def test_configuration_requires_full_mechanism_starling_reranking():
    with pytest.raises(ValueError, match="full_mechanism"):
        validate_scored_neighbors_configuration(
            enabled=True,
            experiment_mode="full_flat",
            retrieval_source="starling",
            retrieval_reranker="assay_transfer",
        )


def test_enabled_prompt_exposes_only_rounded_public_score_and_semantics():
    retrieval = _retrieval()
    enable_scored_neighbors_prompt_policy(retrieval)
    group = retrieval["groups"][0]
    payload = _group_prompt_payload(retrieval["query"], group, include_assay_transfer_score=True)
    serialized = str(payload)

    assert retrieval["experiment"]["llm_neighbor_score_policy"]["name"] == SCORED_NEIGHBORS_POLICY_NAME
    assert payload["neighbors"][0]["assay_transfer_score"] == 0.9
    assert "best compatible source measurement" in serialized
    assert "transfer_selection_score" not in serialized
    assert "transfer_winning_record_id" not in serialized
    assert "structural_rank" not in serialized
    assert "audit-record" not in serialized


def test_disabled_prompt_remains_score_hidden():
    retrieval = _retrieval()
    payload = _group_prompt_payload(retrieval["query"], retrieval["groups"][0])

    assert "assay_transfer_score" not in str(payload)
    assert "assay_transfer_score_policy" not in payload["group"]


def test_hidden_policy_filters_identically_without_exposing_scores():
    visible = _retrieval()
    hidden = copy.deepcopy(visible)
    for retrieval in (visible, hidden):
        neighbor = retrieval["groups"][0]["neighbors"][-1]
        neighbor["transfer_selection_score"] = -1.0
        neighbor["transfer_scored_record_count"] = 0

    prepare_assay_transfer_selected_neighbors(visible, expose_scores=True)
    prepare_assay_transfer_selected_neighbors(hidden, expose_scores=False)

    visible_group = visible["groups"][0]
    hidden_group = hidden["groups"][0]
    assert [row["molecule_chembl_id"] for row in visible_group["neighbors"]] == [
        row["molecule_chembl_id"] for row in hidden_group["neighbors"]
    ]
    assert visible["coverage"] == hidden["coverage"]
    assert "llm_neighbor_score_policy" in visible["experiment"]
    assert "llm_neighbor_score_policy" not in hidden["experiment"]
    assert hidden_group["transfer_neighbor_selection"]["selection_metadata_is_llm_hidden"] is True
    payload = _group_prompt_payload(hidden["query"], hidden_group)
    assert "assay_transfer_score" not in str(payload)
    assert "best compatible source measurement" not in str(payload)


def test_policy_rejects_missing_score_without_fabricating_a_value():
    retrieval = _retrieval()
    del retrieval["groups"][0]["neighbors"][2]["transfer_selection_score"]

    with pytest.raises(ValueError, match="Missing cached transfer_selection_score"):
        enable_scored_neighbors_prompt_policy(retrieval)


def test_policy_drops_unscoreable_audit_sentinel_idempotently():
    retrieval = _retrieval()
    neighbor = retrieval["groups"][0]["neighbors"][-1]
    neighbor["transfer_selection_score"] = -1.0
    neighbor["transfer_scored_record_count"] = 0

    enable_scored_neighbors_prompt_policy(retrieval)
    assert len(retrieval["groups"][0]["neighbors"]) == 4
    assert retrieval["coverage"]["n_unscoreable_selected_dropped"] == 1

    enable_scored_neighbors_prompt_policy(retrieval)
    assert retrieval["coverage"]["n_unscoreable_selected_dropped"] == 1


def test_prompt_hash_ignores_hidden_score_but_tracks_visible_rounded_score():
    baseline = _retrieval()
    changed = copy.deepcopy(baseline)
    changed["groups"][0]["neighbors"][0]["transfer_selection_score"] = 0.5
    assert retrieval_prompt_hash(baseline) == retrieval_prompt_hash(changed)

    enable_scored_neighbors_prompt_policy(baseline)
    enable_scored_neighbors_prompt_policy(changed)
    assert retrieval_prompt_hash(baseline) != retrieval_prompt_hash(changed)


@pytest.mark.parametrize("top_k", [3, 5, 7, 10])
def test_batch_switch_preserves_k_and_is_forwarded_to_pipeline(top_k):
    args = _parse_batch_args(
        CONFIG,
        [
            "--experiment-mode", "full_mechanism",
            "--retrieval-source", "starling",
            "--retrieval-reranker", "assay_transfer",
            "--top-k-per-group", str(top_k),
            "--enable-assay-transfer-scores",
        ],
    )
    _validate_assay_transfer_scores(CONFIG, args)
    command = _single_run_command(CONFIG, args, 0, "run", Path("runs"))

    assert args.top_k_per_group == top_k
    assert command[command.index("--top-k-per-group") + 1] == str(top_k)
    assert "--enable-assay-transfer-scores" in command


def test_removed_top5_flag_is_rejected():
    with pytest.raises(SystemExit):
        _parse_batch_args(CONFIG, ["--enable-assay-transfer-scored-top5"])


def test_v6_5_template_profile_propagates_from_batch_to_molecule_runner():
    args = _parse_batch_args(
        CONFIG,
        [
            "--retrieval-reranker", "assay_transfer",
            "--assay-transfer-template-profile", "v6_5_query_context_copy",
        ],
    )
    command = _single_run_command(CONFIG, args, 0, "run", Path("runs"))

    assert args.assay_transfer_template_profile == "v6_5_query_context_copy"
    assert command[command.index("--assay-transfer-template-profile") + 1] == (
        "v6_5_query_context_copy"
    )


def test_in_distribution_threshold_and_prompt_format_propagate_to_pipeline():
    args = _parse_batch_args(
        CONFIG,
        [
            "--experiment-mode", "full_mechanism",
            "--retrieval-source", "starling_in_distribution",
            "--retrieval-reranker", "assay_transfer",
            "--assay-transfer-min-score", "0.5",
            "--enable-assay-transfer-scores",
            "--group-prompt-format", "assay_transfer_tool",
        ],
    )
    _validate_assay_transfer_scores(CONFIG, args)
    command = _single_run_command(CONFIG, args, 0, "run", Path("runs"))

    assert command[command.index("--assay-transfer-min-score") + 1] == "0.5"
    assert command[command.index("--group-prompt-format") + 1] == "assay_transfer_tool"


@pytest.mark.parametrize("threshold", ["-0.01", "1.01"])
def test_batch_rejects_out_of_range_assay_transfer_threshold(threshold):
    args = _parse_batch_args(
        CONFIG,
        [
            "--retrieval-reranker", "assay_transfer",
            "--assay-transfer-min-score", threshold,
        ],
    )

    with pytest.raises(SystemExit, match="between 0 and 1"):
        _validate_assay_transfer_scores(CONFIG, args)
