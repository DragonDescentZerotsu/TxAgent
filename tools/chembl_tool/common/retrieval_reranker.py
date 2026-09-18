"""Compatibility import for the canonical branch retrieval contract."""

from predict.harnesses.branches.retrieval import *  # noqa: F401,F403
from predict.harnesses.branches.retrieval import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
