import copy
from pathlib import Path

import pytest

from tools.chembl_tool.common.retrieval_ablation import retrieval_prompt_hash
from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    _apply_assay_transfer_scored_top5,
    _parse_args as _parse_batch_args,
    _single_run_command,
)
from tools.chembl_tool.tasks.bioavailability_ma.assay_transfer_prompt_policy import (
    SCORED_TOP5_POLICY_NAME,
    enable_scored_top5_prompt_policy,
    resolve_scored_top5_top_k,
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


def _retrieval() -> dict:
    return {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "experiment": {
            "mode": "full_mechanism",
            "source": "starling",
            "retrieval_reranker": {"name": "assay_transfer"},
        },
        "coverage": {"top_k_per_group": 5},
        "groups": [
            {
                "group_id": "Observed.direct_oral_bioavailability",
                "tier": "Observed",
                "endpoint_group": "direct_oral_bioavailability",
                "transfer_neighbor_selection": {"selection_metadata_is_llm_hidden": True},
                "neighbors": [_neighbor(index, 0.901 + index / 1000) for index in range(1, 6)],
            }
        ],
    }


def test_atomic_switch_preserves_requested_top_k_when_disabled_and_forces_five_when_enabled():
    assert resolve_scored_top5_top_k(
        enabled=False,
        requested_top_k=3,
        experiment_mode="direct",
        retrieval_source="chembl",
        retrieval_reranker="none",
    ) == 3
    assert resolve_scored_top5_top_k(
        enabled=True,
        requested_top_k=17,
        experiment_mode="full_mechanism",
        retrieval_source="starling",
        retrieval_reranker="assay_transfer",
    ) == 5

    with pytest.raises(ValueError, match="full_mechanism"):
        resolve_scored_top5_top_k(
            enabled=True,
            requested_top_k=3,
            experiment_mode="full_flat",
            retrieval_source="starling",
            retrieval_reranker="assay_transfer",
        )


def test_enabled_prompt_exposes_only_rounded_public_score_and_semantics():
    retrieval = _retrieval()
    enable_scored_top5_prompt_policy(retrieval)
    group = retrieval["groups"][0]

    payload = _group_prompt_payload(
        retrieval["query"],
        group,
        include_assay_transfer_score=True,
    )
    serialized = str(payload)

    assert retrieval["experiment"]["llm_neighbor_score_policy"]["name"] == SCORED_TOP5_POLICY_NAME
    assert len(payload["neighbors"]) == 5
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


def test_policy_rejects_missing_score_without_fabricating_a_value():
    retrieval = _retrieval()
    del retrieval["groups"][0]["neighbors"][2]["transfer_selection_score"]

    with pytest.raises(ValueError, match="Missing cached transfer_selection_score"):
        enable_scored_top5_prompt_policy(retrieval)


def test_policy_drops_audit_sentinel_and_returns_fewer_than_five():
    retrieval = _retrieval()
    neighbor = retrieval["groups"][0]["neighbors"][-1]
    neighbor["transfer_selection_score"] = -1.0
    neighbor["transfer_scored_record_count"] = 0

    enable_scored_top5_prompt_policy(retrieval)

    assert len(retrieval["groups"][0]["neighbors"]) == 4
    assert retrieval["coverage"]["n_unscoreable_selected_dropped"] == 1
    assert retrieval["coverage"]["n_neighbors_total"] == 4

    enable_scored_top5_prompt_policy(retrieval)
    assert retrieval["coverage"]["n_unscoreable_selected_dropped"] == 1


def test_prompt_hash_ignores_hidden_score_but_tracks_visible_rounded_score():
    baseline = _retrieval()
    changed = copy.deepcopy(baseline)
    changed["groups"][0]["neighbors"][0]["transfer_selection_score"] = 0.5
    assert retrieval_prompt_hash(baseline) == retrieval_prompt_hash(changed)

    enable_scored_top5_prompt_policy(baseline)
    enable_scored_top5_prompt_policy(changed)
    assert retrieval_prompt_hash(baseline) != retrieval_prompt_hash(changed)


def test_batch_switch_forces_effective_five_and_is_forwarded_to_pipeline():
    args = _parse_batch_args(
        CONFIG,
        [
            "--experiment-mode",
            "full_mechanism",
            "--retrieval-source",
            "starling",
            "--retrieval-reranker",
            "assay_transfer",
            "--enable-assay-transfer-scored-top5",
        ],
    )
    _apply_assay_transfer_scored_top5(CONFIG, args)
    command = _single_run_command(CONFIG, args, 0, "run", Path("runs"))

    assert args.top_k_per_group == 5
    assert command[command.index("--top-k-per-group") + 1] == "5"
    assert "--enable-assay-transfer-scored-top5" in command
