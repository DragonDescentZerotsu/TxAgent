"""Compatibility import for predict.harnesses.branches.reasoning.final_decision."""

from predict.harnesses.branches.reasoning.final_decision import *  # noqa: F401,F403
from predict.harnesses.branches.reasoning.final_decision import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
