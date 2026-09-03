"""Compatibility import for predict.tasks.prompt_profiles."""

from predict.tasks.prompt_profiles import *  # noqa: F401,F403
from predict.tasks.prompt_profiles import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
