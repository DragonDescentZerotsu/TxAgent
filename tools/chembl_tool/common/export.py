"""Compatibility import for predict.utils.files."""

from predict.utils.files import *  # noqa: F401,F403
from predict.utils.files import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
