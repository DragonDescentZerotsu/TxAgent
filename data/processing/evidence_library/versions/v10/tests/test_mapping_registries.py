from __future__ import annotations

import pytest

from data.processing.evidence_library.versions.v10.task_registry import (
    import_task_module,
)


@pytest.mark.parametrize(
    "task",
    [
        "bbb_martins",
        "bioavailability_ma",
        "skin_reaction",
        "ames",
        "dili",
        "carcinogens",
    ],
)
def test_task_mapping_registry_is_hash_pinned(task: str) -> None:
    module = import_task_module(task, "mapping_registry")
    module.validate_mapping_hashes()
    registry = module.mapping_registry()
    assert {"measurement_resolution", "exact_measurement_units"} <= set(
        registry["mappings"]
    )
    if task in {"ames", "dili", "carcinogens"}:
        assert registry["not_applicable_mappings"] == ["auxiliary_context"]
