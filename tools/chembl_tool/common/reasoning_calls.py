"""Compatibility import for predict.harnesses.branches.inference."""

from predict.harnesses.branches.inference import *  # noqa: F401,F403
from predict.harnesses.branches.inference import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
