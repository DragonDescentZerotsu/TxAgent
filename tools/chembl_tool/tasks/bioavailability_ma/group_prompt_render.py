"""Compatibility import for predict.harnesses.branches.tasks.bioavailability_ma.group_prompt."""

from predict.harnesses.branches.tasks.bioavailability_ma.group_prompt import *  # noqa: F401,F403
from predict.harnesses.branches.tasks.bioavailability_ma.group_prompt import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
