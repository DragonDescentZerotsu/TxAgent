"""Compatibility import for predict.tasks.bioavailability_ma.prompts."""

from predict.tasks.bioavailability_ma.prompts import *  # noqa: F401,F403
from predict.tasks.bioavailability_ma.prompts import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
