"""Compatibility import for predict.harnesses.branches.tasks.skin_reaction.pipeline."""

from predict.harnesses.branches.tasks.skin_reaction.pipeline import *  # noqa: F401,F403
from predict.harnesses.branches.tasks.skin_reaction.pipeline import __dict__ as _implementation


def __getattr__(name: str):
    try:
        return _implementation[name]
    except KeyError as exc:
        raise AttributeError(name) from exc


if __name__ == "__main__":
    raise SystemExit(_implementation["main"]())
