"""Compatibility import for predict.harnesses.branches.tasks.bbb_martins.config."""

from predict.harnesses.branches.tasks.bbb_martins.config import *  # noqa: F401,F403
from predict.harnesses.branches.tasks.bbb_martins.config import __dict__ as _implementation
from predict.harnesses.progressive.tasks.bbb_martins import (  # noqa: F401
    PROGRESSIVE_ASSAY_ENDPOINT_DESCRIPTIONS,
    PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS,
    get_progressive_task_contract,
)


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
