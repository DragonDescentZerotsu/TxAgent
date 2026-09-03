"""Compatibility import for predict.retrieval.policies."""

from predict.retrieval.policies import *  # noqa: F401,F403
from predict.retrieval.policies import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
