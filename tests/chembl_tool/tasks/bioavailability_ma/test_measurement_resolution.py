import json

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.build_endpoint_unit_profile import build_profile
from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
    TaskConfig,
    candidate_rows,
)
from tools.chembl_tool.common.starling.measurement_routing import SourceRoutingRules


def test_bioavailability_config_and_gold_contract() -> None:
    config = TaskConfig("bioavailability_ma")
    assert config.BATCH_SIZE == 10
    assert set(config.SOURCE_IDS) == {
        "oral_exposure", "fa", "fg", "fh", "hf_bioavailability"
    }
    prompt = config.render_prompt("oral_exposure")
    assert "Solve concisely." in prompt
    assert "fed/fasted, formulation, treatment" in prompt

    lines = [
        json.loads(line)
        for line in config.DEFAULT_BASE_MAPPING_PATH.parent.parent.joinpath(
            "measurement_resolution_v1",
            "measurement_resolution_gold.jsonl"
        ).read_text().splitlines()
    ]
    manifest, cases = lines[0], lines[1:]
    assert manifest["corpus_version"] == "bioavailability_measurement_resolution_gold.v4"
    assert manifest["cases"] == len(cases) == 300
    assert manifest["source_counts"] == {
        "fa": 75, "fg": 75, "fh": 75, "oral_exposure": 75
    }
    assert len({case["audit_case_id"] for case in cases}) == 300
    assert all(case["route_bucket"] == "extract" for case in cases)


def test_candidate_rows_prefer_persisted_stage01_route_and_endpoint(tmp_path) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cleaned_record_id": "categorical",
                    "source_id": "fa",
                    "endpoint_name": "solubility",
                    "measurement_text": "5 ± 1",
                    "unit_text": "mg/mL",
                    "support_text": "categorical row",
                    "measurement_resolution_route": "categorical",
                    "canonical_endpoint_name": "persisted_categorical",
                },
                {
                    "cleaned_record_id": "forced_extract",
                    "source_id": "fa",
                    "endpoint_name": "solubility",
                    "measurement_text": "5",
                    "unit_text": "mg/mL",
                    "support_text": "persisted extraction row",
                    "measurement_resolution_route": "extract",
                    "canonical_endpoint_name": "persisted_endpoint",
                },
                {
                    "cleaned_record_id": "legacy_extract",
                    "source_id": "fa",
                    "endpoint_name": "solubility",
                    "measurement_text": "5 ± 1",
                    "unit_text": "mg/mL",
                    "support_text": "legacy row",
                    "measurement_resolution_route": None,
                    "canonical_endpoint_name": None,
                },
            ]
        ),
        path,
    )
    rows = candidate_rows(path, TaskConfig("bioavailability_ma"))
    assert [row["id"] for row in rows] == ["forced_extract", "legacy_extract"]
    assert rows[0]["canonical_endpoint_name"] == "persisted_endpoint"
    assert rows[1]["canonical_endpoint_name"] == "solubility"


def test_endpoint_profile_prefers_record_level_endpoint_hook(tmp_path) -> None:
    path = tmp_path / "canonical.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "s",
                    "cleaned_record_id": str(index),
                    "endpoint_name": "raw",
                    "measurement_text": str(index + 1),
                    "unit_text": "mg/mL",
                    "support_text": "",
                }
                for index in range(5)
            ]
        ),
        path,
    )
    profile = build_profile(
        path,
        {"s": SourceRoutingRules(source_id="s")},
        endpoint_resolver=lambda _source, _endpoint: "legacy_endpoint",
        endpoint_record_resolver=lambda _row: "record_endpoint",
    )
    assert "s|record_endpoint" in profile["endpoints"]
    assert "s|legacy_endpoint" not in profile["endpoints"]


def test_endpoint_profile_honors_persisted_stage01_routes(tmp_path) -> None:
    path = tmp_path / "stage01.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "s",
                    "cleaned_record_id": str(index),
                    "endpoint_name": "raw",
                    "canonical_endpoint_name": "endpoint",
                    "measurement_text": str(index + 1),
                    "unit_text": "mg/mL",
                    "support_text": "",
                    "measurement_resolution_route": (
                        "accept" if index < 5 else "extract"
                    ),
                }
                for index in range(10)
            ]
        ),
        path,
    )
    profile = build_profile(
        path,
        {"s": SourceRoutingRules(source_id="s")},
        endpoint_record_resolver=lambda row: row["canonical_endpoint_name"],
    )
    endpoint = profile["endpoints"]["s|endpoint"]
    assert endpoint["resolved_rows"] == 5
    assert endpoint["extraction_candidates"] == 5
    assert endpoint["contribution_kinds"] == {
        "routed_accept": 5,
        "resolved_without_notation": 0,
    }
