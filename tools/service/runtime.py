from __future__ import annotations

import os

from tools.service.config import ServiceSettings


def configure_compute_runtime(settings: ServiceSettings) -> None:
    """Bound nested native parallelism before molecular model inference."""
    threads = str(settings.native_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads

    import torch

    torch.set_num_threads(settings.native_threads)
    try:
        torch.set_num_interop_threads(settings.native_threads)
    except RuntimeError:
        # PyTorch only permits changing inter-op threads before parallel work starts.
        pass
