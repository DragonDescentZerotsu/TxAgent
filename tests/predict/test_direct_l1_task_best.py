from __future__ import annotations

import hashlib
import json
from pathlib import Path

from predict.retrieval.assay_reranking.cache_matched import TASKS, load_cache_policy
from predict.retrieval.assay_reranking.runtime import cache_profile_root, model_profile
from predict.retrieval.assay_reranking.score_tdc_ranked_retrieval import (
    DirectL1PromptRenderer,
    GOLD_PROFILE,
    TDC_PROFILE,
    _profile,
    _rows,
)


FIXTURE = Path(__file__).with_name("fixtures") / "direct_l1_task_best_prompts.v1.json"


def test_released_direct_l1_prompts_render_exactly() -> None:
    for case in json.loads(FIXTURE.read_text()):
        train = {row["benchmark_row_id"]: row for row in _rows(
            case["task"], case["release"], "train"
        )[1]}
        valid = {row["benchmark_row_id"]: row for row in _rows(
            case["task"], case["release"], "valid"
        )[1]}
        prompt = DirectL1PromptRenderer(case["task"], case["release"]).render(
            train[case["known_id"]], valid[case["query_id"]]
        )
        assert hashlib.sha256(prompt.encode()).hexdigest() == case["prompt_sha256"]


def test_task_best_models_and_profiles_are_explicitly_active() -> None:
    tasks = (
        "ames", "dili", "carcinogens",
        "bbb_martins", "bioavailability_ma", "skin_reaction",
    )
    for task in tasks:
        profile = model_profile(task, "direct_task_best")
        assert profile["revision"] and len(profile["revision"]) == 40
        assert profile["prompt_profile"] in {
            "gold_v1_context.v10.3",
            "tdc_binary_same_different_parent_smiles.v1",
        }
    for profile in (GOLD_PROFILE, TDC_PROFILE):
        assert "data/caches/assay_reranking/active" in str(cache_profile_root(profile))


def test_tdc_safety_direct_models_reuse_frozen_l1_universe() -> None:
    for task in ("ames", "dili", "carcinogens"):
        model = model_profile(task, "direct_tdc_best")
        assert model["revision"] and len(model["revision"]) == 40
        assert model["prompt_profile"] == "tdc_binary_same_different_parent_smiles.v1"
        assert _profile(task, "tdc-v1").startswith(f"flat_v5/tdc_v1/{task}/l1/")
        assert DirectL1PromptRenderer(task, "tdc-v1").render(
            {"drug": "CCO", "Y": 1}, {"drug": "CCN"}
        )


def test_task_best_bundles_route_l1_to_published_profiles() -> None:
    bundle_root = Path("predict/retrieval/assay_reranking")
    cases = (
        (
            "ranked_level_retrieval_gold_v1_all_tasks_v1.yaml",
            ("ames", "dili", "carcinogens"),
            GOLD_PROFILE,
        ),
        (
            "ranked_level_retrieval_tdc_v1_task_best_l1_v1.yaml",
            ("bbb_martins", "bioavailability_ma", "skin_reaction"),
            TDC_PROFILE,
        ),
    )
    for bundle_name, tasks, profile in cases:
        for task in tasks:
            for subset in ("valid", "test"):
                policy = load_cache_policy(
                    bundle_root / bundle_name,
                    task,
                    subset,
                    "assay-transfer",
                    TASKS[task][1],
                )
                assert Path(policy["cache_indexes"]["L1"]).parent.parent.name == profile
