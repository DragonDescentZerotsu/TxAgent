"""Compatibility import for predict.harnesses.branches.tasks.bbb_martins.contract."""

from predict.harnesses.branches.tasks.bbb_martins.contract import *  # noqa: F401,F403
from predict.harnesses.branches.tasks.bbb_martins.contract import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
