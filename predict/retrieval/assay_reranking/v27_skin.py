"""Frozen Skin V27 prompt and model provenance for L2/L3 reranking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import runtime


TASK_ID = "skin_reaction"
LEVELS = ("L2", "L3")
ASSET_ROOT = Path(__file__).with_name("prompts") / "v27_skin"
PROJECTION_PATH = ASSET_ROOT / "prompt_projection.json"
PROMPT_SHA256 = "b76517548cc5f432dacd2b15e46aef4035ccbe49a65961842b78c322ad45d21d"
TRAINING_PROMPT_SHA256 = "435d70f963d9b55ac9b13da6f8c32af5414ee402d2af3af08fc5be01c4c7203d"
PROJECTION_SHA256 = "404d75c26b9ebd228d6459f6c57621701645eb5aed89b6d94dd03f261c40eceb"
MODEL = {
    "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-skin-reaction-combined-best",
    "revision": "398a41ccb38cadd120ee26b989bdbce5cab6c021",
    "checkpoint_step": 380,
    "dataset": "jiosephlee/assay-transfer-record-level-v27-skin-reaction-combined-intern",
    "dataset_revision": "1fa33cf3a7df7b6bd07a3b959f37502f951fc8ed",
    "prompt_profile": "v27_skin",
    "prompt_sha256": PROMPT_SHA256,
    "training_prompt_sha256": TRAINING_PROMPT_SHA256,
    "projection_sha256": PROJECTION_SHA256,
    "measurement_display": "canonical_first_atomic_source_fallback.v27_skin",
    "source_snapshot": {
        "records": "819cce097a2459f8509e136f70f600b5ef50ad8cc48e14ebf283d44ce636484a",
        "level_mapping": "5414abdbc607b9a3de25158f5aca075cf47b4e4d9d1c1987692df5c177598a65",
        "source_contract": "b283445be0b1ef5012fa07e135ee65cdce9597c02749eea5a66c12a9f34b3ff9",
    },
}
EMPTY_TEXT = {"", "unknown", "__unknown__", "none", "not_applicable"}


def _clean(value: Any) -> str | None:
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = str(value).strip()
    return None if text.lower() in EMPTY_TEXT else text


class SkinV27PromptRenderer:
    task_id = TASK_ID

    def __init__(self) -> None:
        self.template_hash = runtime.file_sha256(ASSET_ROOT / "prompt.jinja")
        self.projection_hash = runtime.file_sha256(PROJECTION_PATH)
        if self.template_hash != PROMPT_SHA256 or self.projection_hash != PROJECTION_SHA256:
            raise ValueError("Vendored Skin V27 prompt assets changed")
        self.projection = json.loads(PROJECTION_PATH.read_text(encoding="utf-8"))
        if self.projection.get("schema_version") != "assay_transfer_prompt_projection.v19":
            raise ValueError("Unexpected Skin V27 prompt-projection schema")
        self.environment = Environment(
            loader=FileSystemLoader(str(ASSET_ROOT)), undefined=StrictUndefined,
            autoescape=False, keep_trailing_newline=False, trim_blocks=True,
            lstrip_blocks=True, auto_reload=False,
        )

    def _fields(self, record: Mapping[str, Any], names: list[str]) -> list[tuple[str, str]]:
        labels = self.projection["labels"]
        return [
            (str(labels[name]), value)
            for name in names
            if (value := _clean(record.get(name))) is not None
        ]

    def render(self, record: Mapping[str, Any], query_smiles: str) -> str:
        if record.get("task_id") != TASK_ID or record.get("progressive_level") not in LEVELS:
            raise ValueError("Skin V27 prompt record has an incompatible task or level")
        source_id = str(record.get("source_id") or "")
        try:
            binding = self.projection["tasks"][TASK_ID][source_id]
        except KeyError as exc:
            raise ValueError(f"Unknown Skin V27 source: {source_id}") from exc
        payload = dict(record.get("source_fields") or {})
        payload["measurement_text"] = record.get("display_measurement_text")
        payload["unit_text"] = record.get("display_unit_text")
        query_names = list(binding["query"])
        known_only = list(binding["known_only"])
        measurement = ["measurement_text"] if "measurement_text" in known_only else []
        known_names = query_names[:1] + measurement + query_names[1:]
        known_names += [name for name in known_only if name != "measurement_text"]
        return self.environment.get_template("prompt.jinja").render(
            known_smiles=str(record.get("canonical_smiles") or ""),
            query_smiles=query_smiles,
            known_fields=self._fields(payload, known_names),
            query_fields=self._fields(payload, query_names),
        ).strip()

    def prompt_task(self, record: Mapping[str, Any], query_smiles: str) -> runtime.PromptTask:
        return runtime.build_prompt_task(
            self, record, query_smiles=query_smiles,
            group_id=str(record["progressive_level"]),
            molecule_id=str(record.get("parent_id") or ""),
            model=MODEL["model"], model_revision=MODEL["revision"],
        )
