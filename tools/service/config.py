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
    native_threads: int = 1
    batch_workers: int = min(128, max(1, (os.cpu_count() or 8) // 2))
    cache_path: Path | None = None
    cache_memory_entries: int = 10_000


def get_settings() -> ServiceSettings:
    cache_value = os.getenv(
        "TXAGENT_TOOL_CACHE_PATH",
        f"/local/tmp/txagent-tool-cache-{os.getuid()}.sqlite3",
    ).strip()
    return ServiceSettings(
        host=os.getenv("TXAGENT_TOOL_SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("TXAGENT_TOOL_SERVICE_PORT", "8765")),
        enable_molgpka=_env_bool("TXAGENT_ENABLE_MOLGPKA", True),
        prewarm_molgpka=_env_bool("TXAGENT_PREWARM_MOLGPKA", True),
        logd_ph=float(os.getenv("TXAGENT_LOGD_PH", "7.4")),
        native_threads=max(1, int(os.getenv("TXAGENT_TOOL_NATIVE_THREADS", "1"))),
        batch_workers=max(
            1,
            int(
                os.getenv(
                    "TXAGENT_TOOL_BATCH_WORKERS",
                    str(min(128, max(1, (os.cpu_count() or 8) // 2))),
                )
            ),
        ),
        cache_path=Path(cache_value).expanduser() if cache_value else None,
        cache_memory_entries=max(0, int(os.getenv("TXAGENT_TOOL_CACHE_MEMORY_ENTRIES", "10000"))),
    )
