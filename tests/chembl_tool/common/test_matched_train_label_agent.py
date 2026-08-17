import json
from dataclasses import replace
from pathlib import Path

from tools.chembl_tool.paper_experiments.matched_train_label_agent.contract import (
    TASK_SPECS,
    TaskSpec,
    replay_batch,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.compare_full_pool import (
    _retrieval_summary,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.materialize import (
    materialize_task,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.audit import (
    binary_diagnostics,
    stratum_diagnostics,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.bio_trace_diagnosis import (
    outcome_cohort,
    similarity_bin,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.bbb_property_compatible_report import (
    _selector_response_audit,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.run import (
    _fresh_single_required,
    _prompt_profile_args,
    run_experiment,
)
from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import (
    MEANINGFUL_CNS_ACCESS_V1,
    MEANINGFUL_CNS_ADJUDICATION_V2,
    MEANINGFUL_CNS_ADJUDICATION_V3,
)
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    F20_EVIDENCE_CALIBRATED_V2,
    LEGACY_BIOAVAILABILITY_V1,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_bio_historical_matched_runner_pins_legacy_prompt_profile() -> None:
    assert TASK_SPECS["bioavailability_ma"].prompt_profile_args == (
        "--bioavailability-prompt-profile",
        LEGACY_BIOAVAILABILITY_V1,
    )


def test_matched_runner_selects_task_specific_prompt_profiles() -> None:
    assert _prompt_profile_args(
        "bioavailability_ma",
        TASK_SPECS["bioavailability_ma"].prompt_profile_args,
        F20_EVIDENCE_CALIBRATED_V2,
    ) == (
        "--bioavailability-prompt-profile",
        F20_EVIDENCE_CALIBRATED_V2,
    )
    assert _prompt_profile_args(
        "bbb_martins",
        (),
        F20_EVIDENCE_CALIBRATED_V2,
        bbb_prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V2,
    ) == (
        "--bbb-prompt-profile",
        MEANINGFUL_CNS_ADJUDICATION_V2,
    )


def test_prompt_change_gets_same_profile_single_dependency() -> None:
    assert not _fresh_single_required(
        "bbb_martins",
        bbb_prompt_profile=MEANINGFUL_CNS_ACCESS_V1,
        bioavailability_prompt_profile=LEGACY_BIOAVAILABILITY_V1,
    )


def test_runner_accepts_frozen_selector_and_single_overrides(tmp_path: Path) -> None:
    spec = replace(
        TASK_SPECS["bbb_martins"],
        condition="bbb_martins__selector_test",
        input_jsonl=tmp_path / "valid.jsonl",
    )
    output_root = tmp_path / "candidate"
    replay = replay_batch(output_root, spec)
    replay.mkdir(parents=True)
    (replay / "manifest.json").write_text("{}\n", encoding="utf-8")
    single_root = tmp_path / "frozen_single"

    manifest = run_experiment(
        output_root=output_root,
        manifest_only=True,
        specs_override=[spec],
        single_analysis_root_overrides={"bbb_martins": single_root},
        retrieval_contract="frozen selector test",
        bbb_prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V3,
    )

    assert manifest["dependency_batches"] == []
    assert manifest["retrieval_contract"] == "frozen selector test"
    assert manifest["batches"][0]["single_analysis_source_batch"] == str(
        single_root / "bbb_martins" / "bbb_martins__none"
    )


def test_selector_response_audit_separates_rerun_control() -> None:
    rows = [
        {
            "Y": 1,
            "baseline_morgan_vote": 1,
            "property_compatible_vote": 1,
            "baseline_v3_agent": 1,
            "property_compatible_v3_agent": 0,
            "baseline_train_indices": [1, 2, 3],
            "selected_train_indices": [1, 2, 3],
            "neighbor_overlap": 3,
        },
        {
            "Y": 1,
            "baseline_morgan_vote": 0,
            "property_compatible_vote": 1,
            "baseline_v3_agent": 0,
            "property_compatible_v3_agent": 1,
            "baseline_train_indices": [1, 2, 3],
            "selected_train_indices": [2, 3, 4],
            "neighbor_overlap": 2,
        },
    ]
    transferability = {
        "baseline": ["low", "low"],
        "candidate": ["low", "moderate"],
    }

    audit = _selector_response_audit(rows, transferability)

    assert audit["same_neighbor_set"] == {
        "n": 1,
        "agent_prediction_flips": 1,
        "agent_rescues": 0,
        "agent_harms": 1,
    }
    assert audit["changed_vote"]["agent_rescues"] == 1
    assert audit["same_ordered_neighbors"]["n"] == 1
    assert audit["reordered_same_neighbor_set"]["n"] == 0
    assert audit["group_transferability_changed"] == 1
    assert _fresh_single_required(
        "bbb_martins",
        bbb_prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V2,
        bioavailability_prompt_profile=LEGACY_BIOAVAILABILITY_V1,
    )
    assert _fresh_single_required(
        "bbb_martins",
        bbb_prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V3,
        bioavailability_prompt_profile=LEGACY_BIOAVAILABILITY_V1,
    )


def test_materialize_replays_exact_knn_neighbors_and_labels(tmp_path: Path) -> None:
    labels = tmp_path / "valid.jsonl"
    predictions = tmp_path / "valid_predictions.jsonl"
    _write_jsonl(labels, [{"drug": "CCO", "Y": 1}])
    neighbors = [
        {"train_index": 2, "drug": "CCCO", "Y": 1, "similarity": 0.7},
        {"train_index": 4, "drug": "CCN", "Y": 0, "similarity": 0.6},
        {"train_index": 8, "drug": "CCCCO", "Y": 1, "similarity": 0.5},
    ]
    _write_jsonl(
        predictions,
        [
            {
                "query_index": 0,
                "drug": "CCO",
                "Y": 1,
                "prediction": 1,
                "status": "ok",
                "n_eligible_neighbors": 9,
                "neighbors": neighbors,
            }
        ],
    )
    spec = TaskSpec(
        task="example",
        data_name="Example",
        batch_module="unused",
        condition="example__matched_train_label_direct",
        input_jsonl=labels,
        knn_predictions=predictions,
        retrieval_index=tmp_path / "unused.pkl",
        single_source_batch=tmp_path / "none",
        group_id="Direct.example",
        tier="Direct",
        endpoint_group="direct_example",
        source_group_id="TrainLabel.example",
        endpoint_name="frozen example label",
        negative_label_text="Y=0: negative",
        positive_label_text="Y=1: positive",
    )

    manifest = materialize_task(spec, output_root=tmp_path / "out")
    replay = Path(manifest["batch"])
    retrieval = json.loads(
        (replay / "runs" / f"{replay.name}_idx00000" / "retrieval.json").read_text()
    )

    assert manifest["external_starling_records_visible"] is False
    assert manifest["relation_counts"] == {"structural_analog": 3}
    assert retrieval["groups"][0]["n_candidate_molecules"] == 9
    observed = [
        (
            row["train_index"],
            row["train_label"],
            neighbor["similarity"],
        )
        for neighbor in retrieval["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    ]
    assert observed == [(2, 1, 0.7), (4, 0, 0.6), (8, 1, 0.5)]
    assert "Y=1: positive" in retrieval["groups"][0]["neighbors"][0]["evidence_rows"][0][
        "minimal_evidence"
    ]["text"]["evidence"]


def test_materialize_can_describe_exact_minimol_knn_neighbors(tmp_path: Path) -> None:
    labels = tmp_path / "valid.jsonl"
    predictions = tmp_path / "valid_predictions.jsonl"
    _write_jsonl(labels, [{"drug": "CCO", "Y": 1}])
    _write_jsonl(
        predictions,
        [
            {
                "query_index": 0,
                "drug": "CCO",
                "Y": 1,
                "prediction": 1,
                "status": "ok",
                "neighbors": [
                    {"train_index": 2, "drug": "CCCO", "Y": 1, "similarity": 0.91},
                    {"train_index": 4, "drug": "CCN", "Y": 0, "similarity": 0.89},
                    {"train_index": 8, "drug": "CCCCO", "Y": 1, "similarity": 0.87},
                ],
            }
        ],
    )
    spec = replace(
        TASK_SPECS["skin_reaction"],
        condition="skin_reaction__matched_minimol_train_label_direct",
        input_jsonl=labels,
        knn_predictions=predictions,
        retrieval_feature="minimol_embedding",
        retrieval_similarity="cosine",
        retrieval_similarity_metric="MiniMol cosine (L2-normalized, 512 dimensions)",
        retrieval_neighbor_set_contract="exactly_the_formal_minimol_cosine_knn_top3",
    )

    manifest = materialize_task(spec, output_root=tmp_path / "out")
    replay = Path(manifest["batch"])
    retrieval = json.loads(
        (replay / "runs" / f"{replay.name}_idx00000" / "retrieval.json").read_text()
    )

    assert manifest["retrieval_feature"] == "minimol_embedding"
    assert manifest["neighbor_set_contract"] == "exactly_the_formal_minimol_cosine_knn_top3"
    assert retrieval["experiment"]["retrieval_feature"] == {
        "feature": "minimol_embedding",
        "model": "MiniMol",
        "model_version": "minimol_v1",
        "dimension": 512,
        "normalization": "L2",
        "similarity": "cosine",
    }
    assert "fingerprint" not in retrieval["query"]
    assert retrieval["query"]["retrieval_embedding"]["feature"] == "minimol_embedding"
    assert retrieval["groups"][0]["neighbors"][0]["similarity_metric"].startswith(
        "MiniMol cosine"
    )


def test_materialize_rejects_non_majority_knn_prediction(tmp_path: Path) -> None:
    labels = tmp_path / "valid.jsonl"
    predictions = tmp_path / "valid_predictions.jsonl"
    _write_jsonl(labels, [{"drug": "CCO", "Y": 1}])
    _write_jsonl(
        predictions,
        [
            {
                "query_index": 0,
                "drug": "CCO",
                "Y": 1,
                "prediction": 0,
                "status": "ok",
                "neighbors": [
                    {"train_index": 1, "drug": "CCCO", "Y": 1, "similarity": 0.7},
                    {"train_index": 2, "drug": "CCN", "Y": 1, "similarity": 0.6},
                    {"train_index": 3, "drug": "CCCC", "Y": 0, "similarity": 0.5},
                ],
            }
        ],
    )
    spec = TaskSpec(
        "example",
        "Example",
        "unused",
        "condition",
        labels,
        predictions,
        tmp_path / "unused.pkl",
        tmp_path / "none",
        "Direct.example",
        "Direct",
        "direct_example",
        "TrainLabel.example",
        "label",
        "Y=0: negative",
        "Y=1: positive",
    )

    try:
        materialize_task(spec, output_root=tmp_path / "out")
    except ValueError as exc:
        assert "not majority vote" in str(exc)
    else:
        raise AssertionError("Expected mismatched KNN prediction to fail")


def test_paired_diagnostics_separate_vote_strength_and_class_recall() -> None:
    labels = [0, 0, 1, 1]
    knn = [0, 1, 1, 0]
    agent = [0, 0, 0, 0]
    patterns = ["unanimous", "split_2_to_1", "unanimous", "split_2_to_1"]

    assert binary_diagnostics(labels, agent) == {
        "confusion_matrix": [[2, 0], [2, 0]],
        "recall_y0": 1.0,
        "recall_y1": 0.0,
        "predicted_y0": 4,
        "predicted_y1": 0,
    }
    assert stratum_diagnostics(labels, knn, agent, patterns, "unanimous") == {
        "n": 2,
        "prediction_flips": 1,
        "knn_y0_to_agent_y1": 0,
        "knn_y1_to_agent_y0": 1,
        "agent_only_correct": 0,
        "knn_only_correct": 1,
    }
    assert stratum_diagnostics(labels, knn, agent, patterns, "split_2_to_1") == {
        "n": 2,
        "prediction_flips": 1,
        "knn_y0_to_agent_y1": 0,
        "knn_y1_to_agent_y0": 1,
        "agent_only_correct": 1,
        "knn_only_correct": 0,
    }


def test_full_pool_retrieval_summary_preserves_missing_neighbor_coverage() -> None:
    retrievals = [
        {"groups": [{"neighbors": []}]},
        {
            "groups": [
                {"neighbors": [{"similarity": 0.8}, {"similarity": 0.4}]}
            ]
        },
    ]

    assert _retrieval_summary(retrievals) == {
        "queries": 2,
        "neighbor_count_distribution": {"0": 1, "2": 1},
        "coverage": 0.5,
        "mean_neighbors": 1,
        "median_top1_similarity": 0.8,
        "median_all_neighbor_similarity": 0.6000000000000001,
    }


def test_bio_unanimous_positive_diagnostic_bins_and_outcomes() -> None:
    assert similarity_bin(0.299) == "<0.30"
    assert similarity_bin(0.3) == "0.30-0.39"
    assert similarity_bin(0.4) == ">=0.40"
    assert outcome_cohort(1, 0) == "harm_flip_to_low"
    assert outcome_cohort(0, 0) == "rescue_flip_to_low"
    assert outcome_cohort(1, 1) == "correct_keep_high"
    assert outcome_cohort(0, 1) == "wrong_keep_high"
