"""DILI execution overlay for the reviewed 512-request local pool."""

from typing import Any

from data.processing.evidence_library.versions.v10.tasks.dili.starling_measurement_resolution_three_endpoint_256 import *  # noqa: F403
from data.processing.evidence_library.versions.v10.tasks.dili.starling_measurement_resolution_three_endpoint_256 import validate_generation_args as _validate_generation_args
from data.processing.paths import REPO_ROOT


PROVIDER_POOL_CONFIG = (
    REPO_ROOT
    / "predict/api_client/providers/measurement_resolution_v10_three_endpoint_512.json"
)
PROVIDER_POOL_PARALLELISM = 1536


def validate_generation_args(args: Any) -> None:
    _validate_generation_args(args, PROVIDER_POOL_CONFIG, PROVIDER_POOL_PARALLELISM)
