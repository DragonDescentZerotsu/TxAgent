"""Compatibility import for canonical branch artifact replay."""

from predict.harnesses.branches.artifacts import *  # noqa: F401,F403
from predict.harnesses.branches.artifacts import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
