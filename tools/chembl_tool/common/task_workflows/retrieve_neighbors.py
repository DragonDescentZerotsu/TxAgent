"""Compatibility import for predict.retrieval.retrieve."""

from predict.retrieval.retrieve import *  # noqa: F401,F403
from predict.retrieval.retrieve import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
