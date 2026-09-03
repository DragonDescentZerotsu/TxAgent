import json

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.build_endpoint_unit_profile import (
    SUPPORTED_TASKS,
)
from tools.chembl_tool.tasks.clintox.build_stage2_review_samples import (
    build_endpoint_review_cards,
    score_gold_replay,
)


def test_endpoint_review_cards_hide_route_and_separate_diagnostic_context(tmp_path):
    cleaned = tmp_path / "records.parquet"
    candidates = tmp_path / "candidates.jsonl"
    profile = tmp_path / "profile.json"
    output = tmp_path / "review.jsonl"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cleaned_record_id": "row-1",
                    "source_id": "organ_specific_toxicity",
                    "canonical_endpoint_name": "liver injury",
                    "endpoint_name": "liver injury",
                    "measurement_text": None,
                    "support_text": "Liver injury was observed.",
                    "effect_status": "injury_observed",
                    "molecule_name": "example",
                }
            ]
        ),
        cleaned,
    )
    candidates.write_text(
        json.dumps({"task_id": "clintox", "cases": 1})
        + "\n"
        + json.dumps(
            {
                "audit_case_id": "clintox:row-1",
                "source_id": "organ_specific_toxicity",
                "route_bucket": "categorical",
                "input": {
                    "endpoint_name": "liver injury",
                    "measurement_text": None,
                    "support_text": "Liver injury was observed.",
                },
            }
        )
        + "\n"
    )
    profile.write_text(json.dumps({"endpoints": {}}) + "\n")

    manifest = build_endpoint_review_cards(cleaned, candidates, profile, output)
    card = json.loads(output.read_text().splitlines()[1])

    assert "clintox" in SUPPORTED_TASKS
    assert manifest["observed_route_hidden"] is True
    assert "route_bucket" not in card
    assert card["controlled_inputs"] == {"effect_status": "injury_observed"}
    assert card["diagnostic_context"]["molecule_name"] == "example"
    assert card["endpoint_profile_block"] == (
        "[endpoint: liver injury - no reliable deterministic summary]"
    )


def test_gold_replay_scoring_checks_raw_and_canonical_pairs(tmp_path):
    gold = tmp_path / "gold.jsonl"
    mapping = tmp_path / "mapping.parquet"
    metrics = tmp_path / "metrics.json"
    gold.write_text(
        json.dumps({"corpus_version": "test", "cases": 1})
        + "\n"
        + json.dumps(
            {
                "audit_case_id": "clintox:row-1",
                "source_id": "general_cytotoxicity",
                "route_bucket": "extract",
                "input": {"measurement_text": "5"},
                "expected": {
                    "status": "ok",
                    "measurements": [
                        {
                            "measurement": "5",
                            "unit": "µM",
                            "expected_scalar": 5.0,
                            "expected_canonical_unit": "µM",
                        }
                    ],
                },
            }
        )
        + "\n"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cleaned_record_id": "row-1",
                    "status": "ok",
                    "measurements_json": json.dumps(
                        [{"measurement": "5", "unit": "µM"}]
                    ),
                }
            ]
        ),
        mapping,
    )

    result = score_gold_replay(gold, mapping, metrics)

    assert result["overall"]["canonical_accuracy"] == 1.0
    assert result["mismatches"] == []
    assert metrics.is_file()
