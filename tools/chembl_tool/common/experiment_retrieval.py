"""Compatibility import path for the prediction retrieval implementation.

The canonical code lives in ``predict.harnesses.branches.retrieval``. Historical
task, analysis, and test imports continue to resolve through this module.
"""

from predict.harnesses.branches import retrieval as _impl

# Preserve historical imports while the prediction package uses explicit
# branch vocabulary.
EvidenceGroupSpec = _impl.BranchDefinition
SourceExperimentConfig = _impl.BranchRetrievalConfig


def retrieve_group_specs_view(*args, **kwargs):
    """Translate the historical ``specs`` keyword to the branch API."""
    if "specs" in kwargs:
        kwargs["branch_definitions"] = kwargs.pop("specs")
    return _impl.retrieve_branches(*args, **kwargs)


def __getattr__(name: str):
    return getattr(_impl, name)


__all__ = [name for name in vars(_impl) if not name.startswith("_")]
__all__ += ["EvidenceGroupSpec", "SourceExperimentConfig", "retrieve_group_specs_view"]
globals().update({name: getattr(_impl, name) for name in __all__ if hasattr(_impl, name)})
