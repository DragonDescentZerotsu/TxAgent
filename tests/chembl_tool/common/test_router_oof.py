from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.paper_experiments.router_oof.contract import (
    ROUTER_ARTIFACT_DIR,
    ROUTER_FEATURE_STEM,
    TASK_SPECS,
    task_root,
)
from tools.chembl_tool.paper_experiments.router_oof.features import (
    FEATURE_COLUMNS,
    build_task_features,
    paired_outcome,
    summarize_retrieval,
)
from tools.chembl_tool.paper_experiments.router_oof.folds import assign_grouped_folds
from tools.chembl_tool.paper_experiments.router_oof.knn import run_task_oof_knn
from tools.chembl_tool.paper_experiments.router_oof.progress import summarize_agent_progress
from tools.chembl_tool.paper_experiments.router_oof.post_contract import POST_FEATURE_STEM
from tools.chembl_tool.paper_experiments.router_oof.post_features import (
    POST_FEATURE_COLUMNS,
    build_post_selector_features,
)
from tools.chembl_tool.paper_experiments.router_oof.post_train import (
    choose_direction_thresholds,
    route_with_direction_thresholds,
)
from tools.chembl_tool.paper_experiments.router_oof.post_v31_common import (
    PostSelectorArrays,
)
from tools.chembl_tool.paper_experiments.router_oof.post_v31_learning_curve import (
    direction_only_prediction,
    evaluate_learning_curve_continuation_gate,
    stratified_subsample_mask,
    summarize_learning_curve_jobs,
)
from tools.chembl_tool.paper_experiments.router_oof.post_v31_transfer import (
    TransferData,
    cross_task_training_mask,
    task_balanced_weights,
)
from tools.chembl_tool.paper_experiments.router_oof.post_v31_train import (
    choose_safe_direction_policy,
    fit_direction_calibrated_ensemble,
    wilson_lower_bound,
)
from tools.chembl_tool.paper_experiments.router_oof.train import (
    choose_threshold,
    evaluate_promotion_gate,
    train_task_routers,
)
from tools.chembl_tool.paper_experiments.router_oof.valid import evaluate_task_valid


def test_grouped_folds_are_deterministic_and_keep_groups_intact() -> None:
    rows = [
        {"fold_group_key": f"group_{index // 2}", "Y": index % 2}
        for index in range(30)
    ]
    first = assign_grouped_folds(rows, n_folds=5, seed=11)
    second = assign_grouped_folds(reversed(rows), n_folds=5, seed=11)
    assert first == second
    assert set(first.values()) == set(range(5))


def test_retrieval_summary_uses_common_evidence_contract() -> None:
    payload = {
        "groups": [
            {
                "neighbors": [
                    {
                        "standard_inchi_key": "AAA",
                        "similarity": 0.8,
                        "evidence_rows": [
                            {
                                "group_id": "Direct.endpoint",
                                "molecule_chembl_id": "M1",
                                "evidence_source": "source",
                                "standard_type": "endpoint",
                                "evidence_role": "direct_outcome",
                                "evidence_scope": {"species": "human volunteers"},
                                "confidence_score": 9,
                                "uncertainty": ["small cohort"],
                                "source_record_count": 3,
                            }
                        ],
                    },
                    {
                        "standard_inchi_key": "AAA",
                        "similarity": 0.4,
                        "evidence_rows": [
                            {
                                "group_id": "Mechanism.context",
                                "molecule_chembl_id": "M1",
                                "evidence_source": "source",
                                "standard_type": "context",
                                "evidence_role": "context_modifier",
                                "confidence_score": "",
                                "source_record_count": 2,
                            }
                        ],
                    },
                ]
            }
        ]
    }
    summary = summarize_retrieval(payload)
    assert summary["evidence_group_count"] == 1
    assert summary["evidence_neighbor_count"] == 2
    assert summary["evidence_unique_molecule_count"] == 1
    assert summary["evidence_duplicate_fraction"] == 0.5
    assert summary["evidence_similarity_mean"] == pytest.approx(0.6)
    assert summary["role_direct_outcome_fraction"] == 0.5
    assert summary["role_context_modifier_fraction"] == 0.5
    assert summary["quality_confidence_mean"] == 0.9
    assert summary["quality_confidence_missing_fraction"] == 0.5
    assert summary["direct_human_scope_fraction"] == 0.5
    assert summary["uncertainty_row_fraction"] == 0.5


def test_oof_knn_covers_every_training_row(tmp_path: Path) -> None:
    spec = TASK_SPECS["bbb_martins"]
    smiles = [
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCCO",
        "CCCCO",
        "CCN",
        "CCCN",
        "CCCCN",
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "OC1CCCCC1",
    ]
    rows = [
        {
            "task": spec.task,
            "train_index": index,
            "fold": index % 5,
            "drug": drug,
            "Y": index % 2,
        }
        for index, drug in enumerate(smiles)
    ]
    write_jsonl_atomic(task_root(tmp_path, spec.task) / "folds.jsonl", rows)
    metrics = run_task_oof_knn(spec, output_root=tmp_path, k=3)
    predictions = _read_jsonl(task_root(tmp_path, spec.task) / "oof_knn_predictions.jsonl")
    assert metrics["n_train"] == len(rows)
    assert len(predictions) == len(rows)
    assert [row["train_index"] for row in predictions] == list(range(len(rows)))
    assert all(0 <= row["knn_weighted_p_positive"] <= 1 for row in predictions)


def test_build_task_features_joins_fold_artifacts_without_agent_label_features(
    tmp_path: Path,
) -> None:
    spec = TASK_SPECS["bbb_martins"]
    current_root = tmp_path / spec.task / "fold_00"
    write_jsonl_atomic(
        tmp_path / spec.task / "folds.jsonl",
        [{"train_index": 0, "fold": 0, "drug": "CCO", "Y": 1}],
    )
    write_jsonl_atomic(
        current_root / "agent_input" / "valid.jsonl",
        [{"drug": "CCO", "Y": 1, "oof_train_index": 0, "oof_fold": 0}],
    )
    knn = {
        "fold": 0,
        "local_query_index": 0,
        "train_index": 0,
        "Y": 1,
        "prediction": 0,
        "knn_p_positive": 1 / 3,
        "knn_margin": 1 / 3,
        "knn_label_entropy": 0.9,
        "knn_weighted_p_positive": 0.4,
        "knn_weighted_margin": 0.2,
        "knn_weighted_label_entropy": 0.97,
        "knn_similarity_max": 0.8,
        "knn_similarity_mean": 0.7,
        "knn_similarity_kth": 0.6,
        "knn_top1_top2_gap": 0.1,
        "knn_reference_size": 10,
    }
    write_jsonl_atomic(current_root / "knn_predictions.jsonl", [knn])
    batch_root = (
        current_root
        / "agent"
        / "runs_identity_blind_parent_disjoint"
        / spec.task
        / spec.direct_condition
    )
    run_dir = batch_root / "runs" / f"{spec.direct_condition}_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "retrieval.json").write_text(
        json.dumps({"groups": [{"neighbors": []}]}), encoding="utf-8"
    )
    write_jsonl_atomic(
        batch_root / "predictions.jsonl",
        [{"query_index": 0, "label": 1, "pred_label": 1, "run_dir": str(run_dir)}],
    )
    summary = build_task_features(spec, output_root=tmp_path, folds=[0])
    row = _read_jsonl(
        tmp_path / spec.task / f"{ROUTER_FEATURE_STEM}.jsonl"
    )[0]
    assert summary["paired_outcome_counts"]["agent_only_correct"] == 1
    assert row["agent_preferred"] == 1
    assert row["features"]["query_molecular_weight"] > 0
    assert "agent_prediction" not in FEATURE_COLUMNS
    assert "agent_confidence" not in FEATURE_COLUMNS
    assert "knn_reference_size_log1p" not in FEATURE_COLUMNS
    assert "knn_margin" not in FEATURE_COLUMNS


def test_task_local_router_training_writes_nested_oof_artifacts(tmp_path: Path) -> None:
    spec = TASK_SPECS["skin_reaction"]
    rows = []
    for index in range(100):
        label = index % 2
        agent_only = index % 7 == 0
        knn_prediction = 1 - label if agent_only else label
        agent_prediction = label if agent_only else (1 - label if index % 11 == 0 else label)
        outcome = paired_outcome(label, knn_prediction, agent_prediction)
        features = {column: 0.1 + (index % 3) * 0.01 for column in FEATURE_COLUMNS}
        features["knn_weighted_margin"] = 0.0 if agent_only else 0.8
        features["evidence_similarity_max"] = 0.9 if agent_only else 0.3
        rows.append(
            {
                "schema_version": "router_oof_features.direct.v1",
                "task": spec.task,
                "fold": index % 5,
                "train_index": index,
                "Y": label,
                "knn_prediction": knn_prediction,
                "agent_prediction": agent_prediction,
                "paired_outcome": outcome,
                "agent_preferred": int(outcome == "agent_only_correct"),
                "features": features,
            }
        )
    write_jsonl_atomic(
        task_root(tmp_path, spec.task) / f"{ROUTER_FEATURE_STEM}.jsonl", rows
    )
    metrics = train_task_routers(spec, output_root=tmp_path)
    router_dir = task_root(tmp_path, spec.task) / ROUTER_ARTIFACT_DIR
    assert metrics["task"] == spec.task
    assert metrics["n_rows"] == len(rows)
    assert metrics["nested_deployed_router"]["router"]["accuracy"] >= metrics[
        "nested_deployed_router"
    ]["always_knn"]["accuracy"]
    assert (router_dir / "model.joblib").exists()
    assert (router_dir / "nested_oof_predictions.jsonl").exists()
    assert (router_dir / "report_zh.md").exists()


def test_frozen_full_train_router_evaluates_valid_without_retuning(tmp_path: Path) -> None:
    spec = TASK_SPECS["skin_reaction"]
    output_root = tmp_path / "router_oof"
    data_root = tmp_path / "data"
    knn_root = tmp_path / "knn"
    agent_root = tmp_path / "agent"
    query_path = data_root / spec.data_name / "scaffold" / "valid.jsonl"
    write_jsonl_atomic(query_path, [{"drug": "CCO", "Y": 1}])

    knn_dir = knn_root / spec.data_name / "scaffold"
    write_json_atomic(
        knn_dir / "manifest.json",
        {"k": 3, "evaluation_split": "valid", "test_path": str(query_path)},
    )
    write_jsonl_atomic(
        knn_dir / "valid_predictions.jsonl",
        [
            {
                "query_index": 0,
                "drug": "CCO",
                "Y": 1,
                "prediction": 0,
                "score": 1 / 3,
                "status": "ok",
                "n_eligible_neighbors": 10,
                "neighbors": [
                    {"Y": 0, "similarity": 0.8},
                    {"Y": 0, "similarity": 0.7},
                    {"Y": 1, "similarity": 0.6},
                ],
            }
        ],
    )

    batch_root = agent_root / spec.task / spec.direct_condition
    run_dir = batch_root / "runs" / f"{spec.direct_condition}_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "retrieval.json").write_text(
        json.dumps({"groups": [{"neighbors": []}]}), encoding="utf-8"
    )
    write_jsonl_atomic(
        batch_root / "predictions.jsonl",
        [
            {
                "query_index": 0,
                "run_id": f"{spec.direct_condition}_idx00000",
                "run_dir": str(run_dir),
                "smiles": "CCO",
                "label": 1,
                "pred_label": 1,
                "status": "ok",
            }
        ],
    )
    write_json_atomic(
        batch_root / "manifest.json",
        {
            "batch_id": spec.direct_condition,
            "input_jsonl": str(query_path),
            "n_items": 1,
            "model": "gpt-oss-120b",
            "visibility_mode": "identity_blind",
            "neighbor_identity_policy": "parent_disjoint",
            "experiment_mode": "direct",
            "retrieval_source": "starling",
            "neighbor_selector": "similarity",
        },
    )
    write_json_atomic(batch_root / "metrics.json", {"n_failed_runs": 0})

    router_dir = task_root(output_root, spec.task) / ROUTER_ARTIFACT_DIR
    router_dir.mkdir(parents=True)
    x = np.zeros((2, len(FEATURE_COLUMNS)), dtype=float)
    estimator = DummyClassifier(strategy="constant", constant=1).fit(x, [1, 1])
    calibrator = DummyClassifier(strategy="constant", constant=1).fit(
        np.ones((2, 1)), [1, 1]
    )
    joblib.dump(
        {
            "agent_only_estimator": estimator,
            "knn_only_estimator": DummyClassifier(
                strategy="constant", constant=0
            ).fit(x, [0, 0]),
            "agent_only_calibrator": calibrator,
            "knn_only_calibrator": DummyClassifier(
                strategy="constant", constant=0
            ).fit(np.zeros((2, 1)), [0, 0]),
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_profile": "query_knn_evidence",
            "candidate_threshold": 0.5,
            "threshold": 0.5,
            "promoted": True,
        },
        router_dir / "model.joblib",
    )
    write_json_atomic(
        router_dir / "model_manifest.json",
        {
            "task": spec.task,
            "model_family": "dummy",
            "feature_profile": "query_knn_evidence",
            "candidate_threshold": 0.5,
            "threshold": 0.5,
            "feature_columns": list(FEATURE_COLUMNS),
            "available_feature_columns": list(FEATURE_COLUMNS),
            "promoted": True,
            "training_rows": 2,
        },
    )
    write_jsonl_atomic(
        task_root(output_root, spec.task) / f"{ROUTER_FEATURE_STEM}.jsonl",
        [{"row": 0}, {"row": 1}],
    )

    result = evaluate_task_valid(
        spec,
        output_root=output_root,
        data_root=data_root,
        knn_root=knn_root,
        agent_root=agent_root,
    )

    assert result["valid_policy"].startswith("frozen v2 evaluation")
    assert result["metrics"]["always_knn"]["accuracy"] == 0.0
    assert result["metrics"]["router"]["accuracy"] == 1.0
    assert result["metrics"]["router"]["switch_count"] == 1
    assert (router_dir / "valid" / "manifest.json").exists()


def test_dual_risk_threshold_avoids_high_knn_only_risk() -> None:
    labels = np.asarray([0, 1, 0, 1])
    knn = np.asarray([1, 1, 0, 1])
    agent = np.asarray([0, 0, 1, 1])
    # Only row 0 is an agent rescue; row 1 is a KNN-only harm. The net score
    # must rank the rescue above the harm.
    scores = np.asarray([0.8, -0.7, -0.2, 0.0])
    threshold, metrics = choose_threshold(labels, knn, agent, scores)
    assert threshold > 0
    assert metrics["router"]["accuracy"] >= metrics["always_knn"]["accuracy"]


def test_post_selector_features_keep_full_bank_and_add_decision_trace(
    tmp_path: Path,
) -> None:
    spec = TASK_SPECS["skin_reaction"]
    root = task_root(tmp_path, spec.task)
    run_dir = root / "run_0"
    run_dir.mkdir(parents=True)
    (run_dir / "retrieval.json").write_text("{}", encoding="utf-8")
    write_json_atomic(
        run_dir / "single_molecule_reasoning_output.json",
        {
            "status": "ok",
            "llm": {
                "content": {
                    "skin_reaction_prior": "concerning",
                    "confidence": "moderate",
                }
            },
        },
    )
    write_jsonl_atomic(
        run_dir / "group_reasoning_outputs.jsonl",
        [
            {
                "status": "ok",
                "llm": {
                    "content": {
                        "useful_for_skin_reaction_reasoning": True,
                        "transferability": "high",
                        "evidence_direction": "supports_skin_reaction_risk",
                        "confidence": "moderate",
                    }
                },
            }
        ],
    )
    write_json_atomic(
        run_dir / "final_reasoning_output.json",
        {
            "status": "ok",
            "llm": {
                "content": {
                    "skin_reaction_prediction": "risk",
                    "confidence": "high",
                    "conflicting_evidence": ["one"],
                    "evidence_gaps": ["one", "two"],
                    "final_summary": "summary",
                },
                "reasoning_content": "reasoning",
            },
        },
    )
    prediction_path = root / "agent_predictions.jsonl"
    write_jsonl_atomic(
        prediction_path,
        [
            {
                "query_index": 0,
                "pred_label": 1,
                "confidence": "high",
                "final_summary": "summary",
            }
        ],
    )
    write_jsonl_atomic(
        root / f"{ROUTER_FEATURE_STEM}.jsonl",
        [
            {
                "schema_version": "router_oof_features.direct.v2",
                "task": spec.task,
                "fold": 0,
                "local_query_index": 0,
                "train_index": 0,
                "Y": 1,
                "knn_prediction": 0,
                "agent_prediction": 1,
                "paired_outcome": "agent_only_correct",
                "features": {column: 0.25 for column in FEATURE_COLUMNS},
                "provenance": {
                    "retrieval": str(run_dir / "retrieval.json"),
                    "agent_prediction": str(prediction_path),
                },
            }
        ],
    )
    summary = build_post_selector_features(spec, output_root=tmp_path)
    row = _read_jsonl(root / f"{POST_FEATURE_STEM}.jsonl")[0]
    assert summary["n_disagreements"] == 1
    assert set(row["features"]) == set(POST_FEATURE_COLUMNS)
    assert row["features"]["direction_knn0_agent1"] == 1
    assert row["features"]["decision_supports_agent_weight"] > 0
    assert row["features"]["decision_supports_knn_weight"] == 0
    assert row["features"]["trace_single_final_agreement"] == 1
    assert row["features"]["trace_final_confidence_high"] == 1


def test_post_selector_uses_separate_disagreement_direction_thresholds() -> None:
    labels = np.asarray([1, 0, 0, 1, 1, 0])
    knn = np.asarray([0, 1, 0, 1, 0, 1])
    agent = np.asarray([1, 0, 0, 1, 1, 0])
    probability = np.asarray([0.9, 0.2, 0.5, 0.5, 0.8, 0.1])
    thresholds, metrics = choose_direction_thresholds(labels, knn, agent, probability)
    pred, route = route_with_direction_thresholds(
        knn,
        agent,
        probability,
        thresholds["knn0_agent1"],
        thresholds["knn1_agent0"],
    )
    assert metrics["router"]["accuracy"] >= metrics["always_knn"]["accuracy"]
    assert np.array_equal(pred, labels)
    assert route.sum() == 4


def test_v31_safe_direction_policy_requires_precision_lower_bound() -> None:
    target = np.asarray([1] * 30 + [0] * 30)
    eligible = np.ones(len(target), dtype=bool)
    informative = np.asarray([0.9] * 30 + [0.1] * 30)
    selected = choose_safe_direction_policy(
        target,
        informative,
        eligible,
        family="logistic_regression",
        profile="output_query_knn",
    )
    assert selected["risk_gate_passed"] is True
    assert selected["net_rescues"] > 0
    assert selected["precision_wilson_lower"] > 0.5

    uninformative = np.full(len(target), 0.5)
    rejected = choose_safe_direction_policy(
        target,
        uninformative,
        eligible,
        family="logistic_regression",
        profile="output_query_knn",
    )
    assert rejected["risk_gate_passed"] is False
    assert rejected["threshold"] == 1.01
    assert wilson_lower_bound(30, 60) < 0.5


def test_v31_fold_calibrated_ensemble_scores_external_rows() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(100, 3))
    target = (matrix[:, 0] + 0.2 * matrix[:, 1] > 0).astype(int)
    folds = np.arange(100) % 5
    eligible = np.ones(100, dtype=bool)
    ensemble = fit_direction_calibrated_ensemble(
        "logistic_regression",
        matrix,
        target,
        folds,
        eligible,
    )
    probability = ensemble.predict_probability(matrix[:10])
    assert len(ensemble.components) == 5
    assert probability.shape == (10,)
    assert np.all((probability > 0) & (probability < 1))


def test_v31_learning_curve_subsampling_is_deterministic_and_stratified() -> None:
    folds = np.repeat(np.arange(5), 20)
    target = np.tile(np.repeat([0, 1], 10), 5)
    eligible = np.ones(100, dtype=bool)
    first = stratified_subsample_mask(
        eligible, target, folds, fraction=0.25, seed=11
    )
    second = stratified_subsample_mask(
        eligible, target, folds, fraction=0.25, seed=11
    )
    assert np.array_equal(first, second)
    assert int(first.sum()) == 20
    for fold in range(5):
        for label in (0, 1):
            assert int((first & (folds == fold) & (target == label)).sum()) == 2
    assert np.array_equal(
        stratified_subsample_mask(eligible, target, folds, fraction=1.0, seed=1),
        eligible,
    )


def test_v31_learning_curve_uses_direction_only_or_comparator() -> None:
    knn = np.asarray([0, 0, 1, 1, 0, 1])
    agent = np.asarray([0, 1, 0, 1, 1, 0])
    assert np.array_equal(
        direction_only_prediction(knn, agent),
        np.asarray([0, 1, 1, 1, 1, 1]),
    )


def test_v31_diagnostics_share_canonical_row_arrays_and_directions() -> None:
    rows = [
        {
            "fold": 0,
            "Y": 1,
            "knn_prediction": 0,
            "agent_prediction": 1,
            "paired_outcome": "agent_only_correct",
        },
        {
            "fold": 1,
            "Y": 0,
            "knn_prediction": 1,
            "agent_prediction": 0,
            "paired_outcome": "agent_only_correct",
        },
        {
            "fold": 2,
            "Y": 1,
            "knn_prediction": 1,
            "agent_prediction": 1,
            "paired_outcome": "both_correct",
        },
    ]
    arrays = PostSelectorArrays.from_rows(rows)
    masks = arrays.direction_masks()
    assert arrays.folds.tolist() == [0, 1, 2]
    assert arrays.agent_win_target.tolist() == [1, 1, 0]
    assert masks["knn0_agent1"].tolist() == [True, False, False]
    assert masks["knn1_agent0"].tolist() == [False, True, False]


def test_v31_learning_curve_summary_and_stop_gate() -> None:
    jobs = []
    for budget, deltas in ((100, (0.01, 0.02)), (200, (0.02, 0.03))):
        for seed, delta in enumerate(deltas):
            jobs.append(
                {
                    "budget": budget,
                    "seed": seed,
                    "mean_realized_outer_train_disagreements": budget * 0.8,
                    "incremental_vs_direction_only": {
                        "observed_accuracy_delta": delta,
                        "observed_macro_f1_delta": delta / 2,
                        "accuracy_delta_ci95": [-0.01, 0.04],
                        "macro_f1_delta_ci95": [-0.02, 0.03],
                    },
                }
            )
    summary = summarize_learning_curve_jobs(jobs)
    assert [row["budget"] for row in summary] == [100, 200]
    assert summary[0]["accuracy_positive_seed_fraction"] == 1.0
    gate = evaluate_learning_curve_continuation_gate(
        summary, full_disagreements=200
    )
    assert gate["passed"] is False
    assert gate["decision"] == "stop_router_main_method"


def test_v31_transfer_weights_tasks_equally() -> None:
    task = np.asarray(["bbb"] * 8 + ["oral"] * 2, dtype=object)
    eligible = np.ones(10, dtype=bool)
    weights = task_balanced_weights(task, eligible)
    assert weights[task == "bbb"].sum() == pytest.approx(
        weights[task == "oral"].sum()
    )
    assert weights[eligible].mean() == pytest.approx(1.0)


def test_v31_transfer_excludes_cross_task_identity_and_group_overlap() -> None:
    data = TransferData(
        task_names=("bbb", "oral"),
        rows_by_task={"bbb": [], "oral": []},
        global_task=np.asarray(["bbb", "bbb", "oral", "oral", "oral"], dtype=object),
        global_fold=np.asarray([0, 1, 0, 1, 2]),
        global_target=np.asarray([0, 1, 0, 1, 0]),
        global_knn=np.zeros(5, dtype=int),
        global_agent=np.ones(5, dtype=int),
        global_identity=np.asarray(["heldout", "safe-bbb", "heldout", "safe-oral", "other"], dtype=object),
        global_group=np.asarray(["g-held", "g-safe", "g-other", "g-held", "g-safe-oral"], dtype=object),
        global_matrix=np.zeros((5, 2)),
        task_slices={"bbb": slice(0, 2), "oral": slice(2, 5)},
    )
    direction = np.ones(5, dtype=bool)
    target_allowed = np.asarray([False, True, False, False, False])
    target_excluded = np.asarray([True, False, False, False, False])
    selected = cross_task_training_mask(
        data,
        target_task="bbb",
        direction_mask=direction,
        target_allowed=target_allowed,
        target_excluded=target_excluded,
    )
    assert selected.tolist() == [False, True, False, False, True]


def test_promotion_gate_falls_back_when_interval_is_not_positive() -> None:
    labels = np.asarray([0, 1] * 20)
    knn = labels.copy()
    candidate = labels.copy()
    result = evaluate_promotion_gate(
        labels, knn, candidate, seed=7, repetitions=100
    )
    assert result["passed"] is False
    assert "macro_f1_delta_lower_ci_not_positive" in result["failure_reasons"]


def test_agent_progress_requires_all_stage_gates(tmp_path: Path) -> None:
    spec = TASK_SPECS["bbb_martins"]
    fold_root = tmp_path / spec.task / "fold_00"
    write_jsonl_atomic(
        fold_root / "agent_input" / "valid.jsonl",
        [{"drug": "CC", "Y": 0, "oof_train_index": 0, "oof_fold": 0}],
    )
    for condition, groups in (
        (f"{spec.task}__none", []),
        (spec.direct_condition, ["Direct.bbb"]),
    ):
        run_dir = (
            fold_root
            / "agent"
            / "runs_identity_blind_parent_disjoint"
            / spec.task
            / condition
            / "runs"
            / f"{condition}_idx00000"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"expected_group_ids": groups}), encoding="utf-8"
        )
        (run_dir / "single_molecule_reasoning_output.json").write_text(
            json.dumps({"status": "ok"}), encoding="utf-8"
        )
        (run_dir / "final_reasoning_output.json").write_text(
            json.dumps({"status": "ok"}), encoding="utf-8"
        )
        (run_dir / "group_reasoning_outputs.jsonl").write_text(
            "".join(json.dumps({"group_id": group, "status": "ok"}) + "\n" for group in groups),
            encoding="utf-8",
        )
    progress = summarize_agent_progress(
        [spec], output_root=tmp_path, folds={spec.task: [0]}
    )
    assert progress["paired_complete"] == 1
    direct_group = next(
        fold_root.glob(
            "agent/runs_identity_blind_parent_disjoint/bbb_martins/"
            "bbb_martins__starling_direct/runs/*/group_reasoning_outputs.jsonl"
        )
    )
    direct_group.write_text(
        json.dumps({"group_id": "Direct.bbb", "status": "error"}) + "\n",
        encoding="utf-8",
    )
    progress = summarize_agent_progress(
        [spec], output_root=tmp_path, folds={spec.task: [0]}
    )
    assert progress["paired_complete"] == 0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
