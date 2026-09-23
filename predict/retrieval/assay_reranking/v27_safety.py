"""Pinned V27-general safety record prompts for offline assay reranking."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library.versions.v10.tasks.ames.semantic_display import (
    semantic_prompt_payload,
)

from . import runtime


TASKS = ("ames", "dili", "carcinogens")
MODELS = {
    "ames": {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-ames-general-best",
        "revision": "e731fd48ebebe95aa621b3500e59f28d4c27b036",
        "dataset": "jiosephlee/assay-transfer-record-level-v27-ames-general-intern",
        "dataset_manifest_sha256": "6f2d2a3d80869e4c604d193568e3b9373a75200e9e5a9f9736f0f593e3be2c75",
    },
    "dili": {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-dili-general-best",
        "revision": "e34d922504cc67290ce5289607f54e5d74f6868e",
        "dataset": "jiosephlee/assay-transfer-record-level-v27-dili-general-intern",
        "dataset_manifest_sha256": "8bd8cba9c059a441138a093d02d9d94d5298b77203c70fce0cfd204a35e01e1f",
    },
    "carcinogens": {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-carcinogens-general-best",
        "revision": "13a182fbc9f8b2a49c5286514dd5b6110393544b",
        "dataset": "jiosephlee/assay-transfer-record-level-v27-carcinogens-general-intern",
        "dataset_manifest_sha256": "408548b285f32523ac190f9d0340c39049a8e8d7af83d895b59775f80478476c",
    },
}
PROMPT_ROOT = Path(__file__).with_name("prompts") / "v27_skin"
TEMPLATE = "prompt_training.jinja"
TEMPLATE_SHA256 = "435d70f963d9b55ac9b13da6f8c32af5414ee402d2af3af08fc5be01c4c7203d"
HIDDEN_SOURCE_FIELDS = frozenset({
    "paragraph_idx", "support_text", "extra_details", "confidence", "needs_more_context",
    "pmid", "extraction_id", "measurement_text", "canonical_measurement_text", "smiles",
})
RESULT_FIELD_TOKENS = (
    "result", "outcome", "conclusion", "classification_label", "effect_direction",
    "response_value", "quantitative_readout_value", "carcinogenicity_conclusion",
)
def _clean(value: Any) -> str | None:
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip() or None


def _fields(payload: Mapping[str, Any], *, known: bool) -> list[tuple[str, str]]:
    hidden = HIDDEN_SOURCE_FIELDS - {"measurement_text"} if known else HIDDEN_SOURCE_FIELDS
    output = []
    for name, raw in payload.items():
        if name in hidden or (not known and any(token in name.lower() for token in RESULT_FIELD_TOKENS)):
            continue
        value = _clean(raw)
        if value is not None:
            output.append((name.replace("_", " ").capitalize(), value))
    return output


class SafetyV27PromptRenderer:
    def __init__(self, task_id: str):
        if task_id not in TASKS:
            raise ValueError(f"Unsupported V27 safety task: {task_id}")
        self.task_id = task_id
        self.template_hash = runtime.file_sha256(PROMPT_ROOT / TEMPLATE)
        if self.template_hash != TEMPLATE_SHA256:
            raise ValueError("V27 safety prompt template changed")
        self.projection_hash = hashlib.sha256(json.dumps({
            "contract": "v27_general_safety_candidate_context_copy.v1",
            "hidden": sorted(HIDDEN_SOURCE_FIELDS),
            "result_tokens": RESULT_FIELD_TOKENS,
            "ames_semantic_adapter": runtime.file_sha256(Path(semantic_prompt_payload.__code__.co_filename)),
        }, sort_keys=True).encode()).hexdigest()
        self.environment = Environment(
            loader=FileSystemLoader(str(PROMPT_ROOT)), undefined=StrictUndefined,
            autoescape=False, auto_reload=False,
        )

    def _payload(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if record.get("task_id") != self.task_id:
            raise ValueError("V27 safety prompt record has an incompatible task")
        endpoint = record.get("canonical_endpoint_name")
        if not endpoint:
            raise ValueError("V27 safety requires the pinned canonical endpoint projection")
        pair_fields = json.loads(str(record["canonical_pair_fields_json"]))
        payload = dict(record.get("source_fields") or {})
        payload.update(pair_fields)
        payload.update(
            endpoint_name=str(endpoint),
            measurement_text=record.get("canonical_measurement_text") or record.get("display_measurement_text"),
            unit_text=record.get("canonical_unit_text") or record.get("display_unit_text"),
        )
        payload = json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        if self.task_id == "ames":
            payload = semantic_prompt_payload({
                **record, **pair_fields,
                "canonical_category_id": record.get("canonical_category_id"),
            }, payload)
        return payload

    def render_pair(self, known: Mapping[str, Any], query: Mapping[str, Any]) -> str:
        return self.environment.get_template(TEMPLATE).render(
            known_smiles=str(known["canonical_smiles"]),
            query_smiles=str(query["canonical_smiles"]),
            known_fields=_fields(self._payload(known), known=True),
            query_fields=_fields(self._payload(query), known=False),
        ).strip()

    def render(self, record: Mapping[str, Any], query_smiles: str) -> str:
        payload = self._payload(record)
        return self.environment.get_template(TEMPLATE).render(
            known_smiles=str(record["canonical_smiles"]), query_smiles=query_smiles,
            known_fields=_fields(payload, known=True),
            query_fields=_fields(payload, known=False),
        ).strip()

    def prompt_task(self, record: Mapping[str, Any], query_smiles: str) -> runtime.PromptTask:
        model = MODELS[self.task_id]
        return runtime.build_prompt_task(
            self, record, query_smiles=query_smiles,
            group_id=str(record["progressive_level"]),
            molecule_id=str(record["parent_id"]),
            model=model["model"], model_revision=model["revision"],
        )
