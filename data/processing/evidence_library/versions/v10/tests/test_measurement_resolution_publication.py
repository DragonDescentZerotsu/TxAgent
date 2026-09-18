from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
    materialize,
)


def test_materialization_refuses_prompt_or_guard_drift(tmp_path: Path) -> None:
    expected = {"prompt_version": "test.v1", "module_sha256": "before"}
    changed = {"prompt_version": "test.v1", "module_sha256": "after"}
    calls = 0

    def prompt_manifest(**_: object) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return expected if calls == 1 else changed

    config = SimpleNamespace(
        task_id="test",
        MAPPING_VERSION="test.v1",
        BATCH_SIZE=1,
        module=SimpleNamespace(),
        prompt_manifest=prompt_manifest,
    )
    assignment = {
        "cleaned_record_id": "row-1",
        "source_id": "source",
        "status": "unsure",
        "measurements_json": "[]",
        "quantity_count": 0,
        "assignment_method": "model_single_pass",
        "rejected_response_json": None,
        "raw_response_json": "{}",
    }
    cache = SimpleNamespace(
        assignments={"row-1": assignment},
        attempted={"row-1"},
        provenance={},
        events=[],
    )
    records_path = tmp_path / "records"
    profile_path = tmp_path / "profile"
    records_path.write_text("records", encoding="utf-8")
    profile_path.write_text("profile", encoding="utf-8")
    mapping_path = tmp_path / "measurement_resolution.parquet"

    with pytest.raises(ValueError, match="changed while assignments"):
        materialize(
            [{"id": "row-1", "source_id": "source"}],
            cache,
            config,
            mapping_path=mapping_path,
            records_path=records_path,
            profile_path=profile_path,
            model="model",
            api_base_url="https://example.test/v1",
            max_completion_tokens=10,
            reasoning_mode="none",
            expected_prompt_manifest=expected,
        )

    assert not mapping_path.exists()
    assert calls == 2
