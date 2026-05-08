from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 8765
    enable_molgpka: bool = True
    prewarm_molgpka: bool = True
    logd_ph: float = 7.4


def get_settings() -> ServiceSettings:
    return ServiceSettings(
        host=os.getenv("TXAGENT_TOOL_SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("TXAGENT_TOOL_SERVICE_PORT", "8765")),
        enable_molgpka=_env_bool("TXAGENT_ENABLE_MOLGPKA", True),
        prewarm_molgpka=_env_bool("TXAGENT_PREWARM_MOLGPKA", True),
        logd_ph=float(os.getenv("TXAGENT_LOGD_PH", "7.4")),
    )
