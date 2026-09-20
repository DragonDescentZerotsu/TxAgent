from __future__ import annotations

import json
from pathlib import Path

import pytest

from predict.harnesses.progressive import runner
from predict.harnesses.progressive.prompt import build_level_messages
from predict.retrieval.assay_reranking.runtime import (
    DATA_ACTIVE_CACHE_ROOT,
    cache_profile_root,
)
from predict.utils.json import read_jsonl, sha256_file


@pytest.mark.parametrize("task,subset,queries", [
    ("bbb_martins", "valid", 197),
    ("bbb_martins", "test", 530),
    ("bioavailability_ma", "valid", 64),
    ("bioavailability_ma", "test", 128),
])
def test_mixed_l1_cache_is_complete_and_disjoint(
    task: str, subset: str, queries: int
) -> None:
    root = cache_profile_root("tdc_mixed_l1_v1") / task
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["subsets"][subset]
    rows = read_jsonl(root / entry["cache"])

    assert cache_profile_root("tdc_mixed_l1_v1").parent == DATA_ACTIVE_CACHE_ROOT
    assert manifest["schema_version"] == "tdc_mixed_l1_morgan.v1"
    assert manifest["status"] == "complete"
    assert len(rows) == entry["queries"] == queries
    assert sha256_file(root / entry["cache"]) == entry["cache_sha256"]
    assert all(len(row["cards"]) == 10 for row in rows)
    assert all(
        card["molecule_identity_key"] != row["query_molecule_identity_key"]
        and (
            not row["query_scaffold"]
            or card["bemis_murcko_scaffold"] != row["query_scaffold"]
        )
        for row in rows
        for card in row["cards"]
    )
    assert {card["source_kind"] for row in rows for card in row["cards"]} == {
        "gold_v1",
        "tdc_v1",
    }


def test_tdc_harness_is_opt_in_l1_only_and_renders_source_labels() -> None:
    args = runner.parse_args([
        "--harness-version", "tdc-mixed-progressive-v1",
        "--tasks", "bbb_martins",
        "--prepare-only",
    ])
    assert args.gold_label_version == "current"
    assert args.prompt_version == "tdc_mixed_progressive_v1"
    assert args.reranking == args.l1_ranking == "morgan"
    assert args.l1_source == "tdc_mixed_train"
    assert args.max_level == 1
    assert args.query_prior == "none"
    assert args.skip_tool_prefetch is True

    records = read_jsonl(runner.PROGRESSIVE_TASKS["bbb_martins"].input_jsonl)
    candidates, audit = runner._tdc_mixed_l1_candidates(
        task="bbb_martins",
        records=records,
        cache_root=Path(args.tdc_mixed_l1_cache),
        subset="valid",
        prompt_version=args.prompt_version,
    )
    query_id = str(records[0]["benchmark_row_id"])
    active, _ = runner.select_initial_evidence(candidates[query_id], level=1)
    prepared = {
        "query_smiles": records[0]["drug"],
        "condition_sentence": "",
        "query_prior": {},
        "query_tool_summary": {},
        "l1_ranking": "morgan",
    }
    messages, references = build_level_messages(
        contract=runner._task_contract("bbb_martins", args.prompt_version),
        levels=runner._run_levels(args, "bbb_martins"),
        current_level=1,
        prepared=prepared,
        active=active,
        prior_state=None,
        profile="standard",
        prompt_version=args.prompt_version,
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=50,
        molecule_limit=10,
        include_indirect=False,
    )
    assert audit["selection_policy"] == "tdc_mixed_l1_morgan.v1"
    assert "TDC external dataset" in messages[1]["content"]
    assert "Gold-v1 conditioned benchmark" in messages[1]["content"]
    assert references and {row["unit_kind"] for row in references} == {
        "molecule",
        "record",
    }


def test_default_progressive_contract_remains_gold_current() -> None:
    args = runner.parse_args([])
    assert args.gold_label_version == "current"
    assert args.harness_version == "reranked-progressive-v2"
    assert args.l1_source == "gold_train"


def test_tdc_harness_rejects_gold_release_selector() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--harness-version", "tdc-mixed-progressive-v1",
            "--gold-label-version", "v1",
            "--prepare-only",
        ])


@pytest.mark.parametrize("extra", [
    ["--tasks", "skin_reaction"],
    ["--reranking", "assay-transfer"],
    ["--max-level", "2"],
])
def test_tdc_harness_rejects_contract_expansion(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--harness-version", "tdc-mixed-progressive-v1",
            "--prepare-only",
            *extra,
        ])
