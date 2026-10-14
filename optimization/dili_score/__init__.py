"""Score one DILI evidence record without models or external services."""

from .scoring import DILI_SIMPLE_VERSION, score_dili_record

__version__ = "0.1.0"
__all__ = ["DILI_SIMPLE_VERSION", "score_dili_record"]
