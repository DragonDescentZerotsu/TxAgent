"""Compatibility import for predict.tasks.bioavailability_ma.constants."""

from predict.tasks.bioavailability_ma.constants import *  # noqa: F401,F403
from predict.tasks.bioavailability_ma.constants import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
