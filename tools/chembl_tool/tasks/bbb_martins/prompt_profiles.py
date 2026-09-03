"""Compatibility import for predict.tasks.bbb_martins.prompts."""

from predict.tasks.bbb_martins.prompts import *  # noqa: F401,F403
from predict.tasks.bbb_martins.prompts import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
