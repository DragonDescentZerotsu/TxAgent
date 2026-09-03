"""Compatibility import for predict.harnesses.branches.tasks.bioavailability_ma.context."""

from predict.harnesses.branches.tasks.bioavailability_ma.context import *  # noqa: F401,F403
from predict.harnesses.branches.tasks.bioavailability_ma.context import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
