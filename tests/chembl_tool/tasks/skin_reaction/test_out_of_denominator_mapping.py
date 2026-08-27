import hashlib
import json
from decimal import Decimal
from pathlib import Path

from tools.chembl_tool.common.starling.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)


DATA_DIR = (
    Path(__file__).parents[4]
    / "tools/chembl_tool/tasks/skin_reaction/data_processing/canonicalization_v7"
)


def test_reviewed_out_of_denominator_keys_match_exact_unit_map() -> None:
    map_path = DATA_DIR / "exact_measurement_unit_map.v2.json"
    review = json.loads(
        (DATA_DIR / "out_of_denominator_review.v1.json").read_text(encoding="utf-8")
    )
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    entries_sha = hashlib.sha256(
        json.dumps(
            payload["entries"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert entries_sha == review["reviewed_mapping_entries_sha256"]

    mapping = load_exact_unit_mapping(map_path)
    counts = {"responder_fraction": 0, "bounded_scale": 0, "mixed_or_ambiguous": 0}
    affected_rows = 0
    for decision in review["keys"]:
        key = (
            "skin_reaction",
            decision["canonical_endpoint_name"],
            decision["input_unit"],
        )
        rule = mapping[key]
        assert rule["action"] == "map"
        assert rule["canonical_unit"] == decision["canonical_unit"]
        assert rule["scale"] == decision["scale"]
        counts[decision["classification"]] += 1
        affected_rows += decision["affected_rows"]

        denominator = Decimal(decision["denominator"])
        values = [Decimal(row["input_measurement"]) for row in decision["records"]]
        assert len(values) == decision["affected_rows"]
        assert all(Decimal(0) <= value <= denominator for value in values)
        if decision["classification"] == "responder_fraction":
            assert rule["domain"] == "nonnegative"
            assert all(value == value.to_integral_value() for value in values)

    summary = review["summary"]
    assert len(review["keys"]) == summary["exact_keys"]
    assert counts["responder_fraction"] == summary["responder_fraction_keys"]
    assert counts["bounded_scale"] == summary["bounded_scale_keys"]
    assert counts["mixed_or_ambiguous"] == summary["mixed_or_ambiguous_keys"]
    assert affected_rows == summary["affected_rows"]
