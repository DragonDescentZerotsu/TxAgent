"""Regression against authored full-record placement judgments, not rule replay."""

import json
from pathlib import Path

import pytest

from tools.chembl_tool.tasks.ames.reviewed_source import (
    apply_placement,
    load_placements,
)
from tools.chembl_tool.tasks.ames.source_contract import (
    Decision,
    DIRECT,
    classify,
    indirect_family,
)

ROOT = Path("data/starling_data/ames")


def judgments():
    origins = {"source_review_v5", "source_review_v6"} | {
        row["review_origin"] for row in load_placements().values()
    }
    for origin in sorted(origins):
        root = ROOT / origin
        samples = {
            r["source_record_id"]: r
            for r in map(json.loads, (root / "sample.jsonl").read_text().splitlines())
        }
        for row in map(
            json.loads, (root / "audit_annotations.jsonl").read_text().splitlines()
        ):
            sample = samples[row["source_record_id"]]
            if sample["level"] == 1:
                continue  # L1 is checked against actual post-collapse voters.
            expected = (
                (
                    row["proposed_groups"][0]
                    if row["disposition"] == "move"
                    else row["group_id"]
                )
                if origin == "source_review_v5"
                else row["group_id"]
            )
            yield sample, expected


@pytest.mark.parametrize(
    "sample,expected",
    list(judgments()),
    ids=lambda x: x.get("source_record_id") if isinstance(x, dict) else x,
)
def test_authored_placement(sample, expected):
    decision = classify(sample["source_id"], sample["raw"])
    review = load_placements().get(sample["source_record_id"])
    if review:
        decision = apply_placement(review, sample["raw"], decision)
    assert decision.group == expected


def test_placement_cannot_hide_new_text_or_change_gold():
    sample, _ = next(
        (s, e) for s, e in judgments() if s["source_record_id"] in load_placements()
    )
    review = load_placements()[sample["source_record_id"]]
    raw = sample["raw"]
    changed = {
        **raw,
        "support_text": raw["support_text"]
        + " An Ames-positive result was also reported.",
    }
    with pytest.raises(ValueError, match="payload changed"):
        apply_placement(review, changed, classify(sample["source_id"], changed))
    with pytest.raises(ValueError, match="change gold"):
        apply_placement(review, raw, Decision(DIRECT, "gold", 1))
    with pytest.raises(ValueError, match="grant eligibility"):
        apply_placement(review, raw, Decision("", "excluded"))


@pytest.mark.parametrize(
    "sample,expected",
    [
        (s, e)
        for s, e in judgments()
        if s["level"] > 2 and s["source_record_id"] not in load_placements()
    ],
)
def test_same_indirect_content_has_same_family_in_any_run(sample, expected):
    assert indirect_family(sample["raw"]).group == expected
    assert {
        classify(src, sample["raw"]).group
        for src in ("ames_base", "ames_v1", "ames_v2", "ames_v3")
    } == {expected}
