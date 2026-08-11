"""Register TxAgent adapters, then delegate training to the pinned NeMo RL runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_lora.nemo_adapter import register_starling_components
from rl_lora.one_pass_runtime import load_fresh_one_pass_audit
from rl_lora.validate_backend_contract import validate_nemo_config


def _config_path(argv: list[str]) -> Path | None:
    for index, value in enumerate(argv):
        if value == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if value.startswith("--config="):
            return Path(value.split("=", 1)[1])
    return None


def _validate_current_one_pass_config() -> None:
    config_path = _config_path(sys.argv[1:])
    if config_path is None or not config_path.name.startswith(
        "grpo_gpt_oss_20b_one_pass_bio"
    ):
        return
    result = validate_nemo_config(config_path)
    if result["status"] != "pass":
        raise ValueError(
            "NeMo config differs from the shared one-pass training contract: "
            f"{result['deviations'] or result['backend_invariant_failures']}"
        )
    data_audit = load_fresh_one_pass_audit(
        Path(result["data_path"]),
        minimum_contract="one_pass_data_audit.v2",
    )
    result["data_audit"] = data_audit["audit_path"]
    result["data_audit_sha256"] = data_audit["audit_sha256"]
    result["data_sha256"] = data_audit["data_sha256"]
    result["data_n_rows"] = data_audit["n_rows"]
    log_dir = Path(result["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "backend_contract.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    nemo_root = Path(os.environ.get("NEMO_RL_ROOT", "/local/tianang/nemo-rl"))
    examples = nemo_root / "examples"
    if not (examples / "run_grpo.py").is_file():
        raise FileNotFoundError(f"NeMo RL runner not found under {examples}")
    _validate_current_one_pass_config()
    sys.path.insert(0, str(examples))
    register_starling_components()
    from run_grpo import main as nemo_main

    nemo_main()


if __name__ == "__main__":
    main()
