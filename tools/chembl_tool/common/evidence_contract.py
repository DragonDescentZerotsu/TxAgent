"""Compatibility import for predict.llm_io.evidence."""

from predict.llm_io.evidence import *  # noqa: F401,F403
from predict.llm_io.evidence import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
