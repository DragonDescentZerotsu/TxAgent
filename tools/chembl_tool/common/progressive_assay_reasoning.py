"""Compatibility imports for the canonical progressive state contract."""

from predict.harnesses.progressive import state as _impl


def __getattr__(name: str):
    return getattr(_impl, name)


__all__ = [name for name in vars(_impl) if not name.startswith("_")]
globals().update({name: getattr(_impl, name) for name in __all__})
