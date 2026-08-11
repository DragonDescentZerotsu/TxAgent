#!/usr/bin/env python3
"""Import GPT-OSS checkpoints with the non-TE output layer fusion disabled.

Megatron Bridge currently enables gradient accumulation fusion whenever
Transformer Engine imports successfully.  GPT-OSS still builds its output
projection with Megatron's native ``ColumnParallelLinear``, which requires the
optional Apex fused-weight-gradient extension when that flag is enabled.  The
extension is irrelevant during checkpoint import, so this thin wrapper disables
the flag after applying the upstream conversion topology and otherwise delegates
to the supported Megatron Bridge conversion worker unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _install_bridge_paths() -> None:
    nemo_rl_root = Path(os.environ.get("NEMO_RL_ROOT", "/local/tianang/nemo-rl"))
    bridge_root = (
        nemo_rl_root / "3rdparty" / "Megatron-Bridge-workspace" / "Megatron-Bridge"
    )
    for path in (
        bridge_root / "scripts" / "conversion",
        bridge_root / "src",
        bridge_root / "3rdparty" / "Megatron-LM",
    ):
        sys.path.insert(0, str(path))


def main() -> None:
    _install_bridge_paths()

    import gpu_backend  # noqa: PLC0415
    import run_conversion  # noqa: PLC0415
    from megatron.bridge import AutoBridge  # noqa: PLC0415
    from megatron.bridge.utils import fusions  # noqa: PLC0415

    configure_model_provider = gpu_backend._configure_model_provider
    to_megatron_provider = AutoBridge.to_megatron_provider
    fusions.can_enable_gradient_accumulation_fusion = lambda: False

    def provide_without_apex_fusion(*args: object, **kwargs: object):
        model_provider = to_megatron_provider(*args, **kwargs)
        model_provider.gradient_accumulation_fusion = False
        return model_provider

    def configure_without_apex_fusion(*args: object, **kwargs: object) -> None:
        configure_model_provider(*args, **kwargs)
        model_provider = args[0]
        model_provider.gradient_accumulation_fusion = False

    gpu_backend._configure_model_provider = configure_without_apex_fusion
    AutoBridge.to_megatron_provider = provide_without_apex_fusion
    run_conversion._configure_logging()
    run_conversion.main()


if __name__ == "__main__":
    main()
