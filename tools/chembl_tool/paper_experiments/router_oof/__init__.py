"""Train-only OOF infrastructure for task-local KNN-agent routers."""

from .contract import FEATURE_SCHEMA_VERSION, FOLD_SCHEMA_VERSION, TASK_SPECS, TaskSpec

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "FOLD_SCHEMA_VERSION",
    "TASK_SPECS",
    "TaskSpec",
]
