"""Compatibility import for the canonical branch runner CLI arguments."""

from predict.harnesses.branches.runner import *  # noqa: F401,F403
from predict.harnesses.branches.runner import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
