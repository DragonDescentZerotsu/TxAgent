"""Compatibility import for predict.harnesses.branches.tasks.bbb_martins.context."""

from predict.harnesses.branches.tasks.bbb_martins.context import *  # noqa: F401,F403
from predict.harnesses.branches.tasks.bbb_martins.context import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
