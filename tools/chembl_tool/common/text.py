"""Compatibility import for predict.utils.text."""

from predict.utils.text import *  # noqa: F401,F403
from predict.utils.text import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
