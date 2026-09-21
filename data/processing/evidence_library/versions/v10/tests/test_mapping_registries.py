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
        auxiliary = registry["mappings"]["auxiliary_context"]
        assert auxiliary["acceptance_exception"]
        reference = registry["mappings"]["reference_semantics"]
        assert reference["outputs"] == [
            "canonical_reference_scope",
            "canonical_reference_basis",
        ]


@pytest.mark.parametrize("task", ["ames", "dili", "carcinogens"])
def test_imported_stage1_uses_frozen_measurement_assets(task: str) -> None:
    policy = import_task_module(task, "starling_policy").POLICY
    registry = import_task_module(task, "mapping_registry").mapping_registry()

    assert policy.stage1_measurement_routing_enabled is True
    assert policy.measurement_resolution_enabled is True
    assert str(policy.exact_unit_mapping_path).endswith(
        registry["mappings"]["exact_measurement_units"]["path"]
    )
