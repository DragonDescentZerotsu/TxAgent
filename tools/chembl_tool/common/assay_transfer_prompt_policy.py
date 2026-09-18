"""Compatibility import for predict.harnesses.branches.prompt."""

from predict.harnesses.branches.prompt import *  # noqa: F401,F403
from predict.harnesses.branches.prompt import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
