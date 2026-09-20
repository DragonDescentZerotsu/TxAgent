"""Fail-closed imports for task modules owned by this construction release."""

from importlib import import_module
from types import ModuleType


_FULL_TASK_MODULES = frozenset(
    {
        "build_normalized_starling_evidence_library",
        "build_starling_downstream_artifacts",
        "mapping_registry",
        "starling_categorical_response",
        "starling_measurement_resolution",
        "starling_policy",
        "starling_reference_semantics",
    }
)
TASK_MODULES = {
    task_id: _FULL_TASK_MODULES
    for task_id in ("bbb_martins", "bioavailability_ma", "skin_reaction")
}
TASK_MODULES["ames"] = frozenset(
    {
        "build_normalized_starling_evidence_library",
        "mapping_registry",
        "starling_categorical_response",
        "starling_measurement_resolution",
        "starling_policy",
    }
)
TASK_MODULES["dili"] = frozenset(
    {
        "build_normalized_starling_evidence_library",
        "mapping_registry",
        "starling_measurement_resolution",
        "starling_policy",
    }
)
TASK_MODULES["carcinogens"] = frozenset(
    {
        "build_normalized_starling_evidence_library",
        "mapping_registry",
        "starling_measurement_resolution",
        "starling_policy",
    }
)
TASK_IDS = frozenset(TASK_MODULES)


def task_module_name(task_id: str, module_name: str) -> str:
    if task_id not in TASK_IDS:
        raise ValueError(f"unsupported evidence-library task: {task_id!r}")
    if module_name not in TASK_MODULES[task_id]:
        raise ValueError(f"unsupported task module: {module_name!r}")
    return f"{__package__}.tasks.{task_id}.{module_name}"


def import_task_module(task_id: str, module_name: str) -> ModuleType:
    return import_module(task_module_name(task_id, module_name))
