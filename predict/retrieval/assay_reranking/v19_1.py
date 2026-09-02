"""V19.1 BBB numeric-indirect prompt and cache provenance."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from predict.retrieval.assay_reranking.runtime import (
    CACHE_ROOT,
    CachedAssayReranker,
    file_sha256,
    model_profile as load_model_profile,
)


ASSET_ROOT = Path(__file__).with_name("prompts") / "v19_1"
PROJECTION_PATH = ASSET_ROOT / "prompt_projection.json"
PROMPT_SHA256 = "e6a712824bf53e2e149138cd57511554bcee5b996f74215adc71a439e046f8b9"
PROJECTION_SHA256 = "a81e30e10a3fcef8d216867bb873a16bfaf69386e3f08f21071188c313ea0166"
PROFILE_NAME = "v19_1_numeric_all_indirect_top75"
TEMPLATE_PROFILE = "v19_1_retrieval_context_copy"
QUERY_CONTEXT_POLICY = "copy_retrieval_assay_context_value_hidden.v19_1"
GROUP_IDS = (
    "Mechanism.tier_2",
    "Mechanism.tier_3",
    "Mechanism.tier_4",
)
SOURCE_GROUP_IDS = (
    "Mechanism.passive_permeability",
    "Mechanism.efflux_transport",
    "Mechanism.influx_transport",
)
SOURCE_STAGE03_MANIFEST_SHA256 = (
    "c7b219c7b13074b1d7e916ae427122a71f284541e8adc09ec1493faaa8da0c26"
)
VALID_QUERY_SHA256 = (
    "acef8e37518581e166b16454681607ab50d9b45e63b4494f4a803902dacae187"
)
VALID_QUERY_ROWS = 397
EMPTY_TEXT = {"", "unknown", "__unknown__", "none", "not_applicable"}
_ENDPOINT_SEPARATORS = re.compile(r"[\s_\-\u2010-\u2015\u2212]+")


def model_profile(task_id: str) -> dict[str, Any]:
    return load_model_profile(task_id, "indirect")


def default_cache_paths(
    task_id: str, *, split: str = "scaffold", subset: str = "valid"
) -> dict[str, str]:
    model_profile(task_id)
    root = CACHE_ROOT / PROFILE_NAME / task_id / split / subset
    return {
        "catalog": "",
        "candidate_manifest": "",
        "cache": str(root / "scores.sqlite3"),
        "version": str(root / "VERSION.json"),
    }


def load_projection() -> dict[str, Any]:
    projection = json.loads(PROJECTION_PATH.read_text(encoding="utf-8"))
    if projection.get("schema_version") != "assay_transfer_prompt_projection.v19.1":
        raise ValueError("Unexpected V19.1 prompt-projection schema")
    return projection


def verify_vendored_assets() -> dict[str, str]:
    observed = file_sha256(ASSET_ROOT / "prompt.jinja")
    if observed != PROMPT_SHA256:
        raise ValueError(
            "Vendored V19.1 prompt hash mismatch: "
            + json.dumps(
                {"expected": PROMPT_SHA256, "observed": observed}, sort_keys=True
            )
        )
    projection_hash = file_sha256(PROJECTION_PATH)
    if projection_hash != PROJECTION_SHA256:
        raise ValueError("Vendored V19.1 prompt projection hash mismatch")
    return {"prompt.jinja": observed, "prompt_projection.json": projection_hash}


def canonicalize_endpoint(value: Any) -> str:
    """Normalize endpoint spelling without applying semantic aliases."""
    text = str(value).strip() if value is not None else ""
    text = unicodedata.normalize("NFKC", text or "missing_endpoint").casefold()
    return _ENDPOINT_SEPARATORS.sub("_", text).strip("_")


def _clean(value: Any) -> str | None:
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = str(value).strip()
    return None if text.lower() in EMPTY_TEXT else text


class V191PromptRenderer:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.projection = load_projection()
        if task_id not in self.projection["tasks"]:
            raise ValueError(f"V19.1 projection has no task {task_id}")
        self.template_hash = file_sha256(ASSET_ROOT / "prompt.jinja")
        self.projection_hash = file_sha256(PROJECTION_PATH)
        self.environment = Environment(
            loader=FileSystemLoader(str(ASSET_ROOT)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=False,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )

    def _binding(self, source_id: str) -> dict[str, list[str]]:
        try:
            return self.projection["tasks"][self.task_id][source_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown V19.1 source {self.task_id}/{source_id}"
            ) from exc

    def _fields(
        self, record: Mapping[str, Any], names: list[str]
    ) -> list[tuple[str, str]]:
        labels = self.projection["labels"]
        return [
            (str(labels[name]), value)
            for name in names
            if (value := _clean(record.get(name))) is not None
        ]

    def render(self, record: Mapping[str, Any], query_smiles: str) -> str:
        if str(record.get("task_id") or "") != self.task_id:
            raise ValueError("V19.1 prompt record belongs to another task")
        binding = self._binding(str(record.get("source_id") or ""))
        query_names = list(binding["query"])
        known_only = list(binding["known_only"])
        measurement = ["measurement_text"] if "measurement_text" in known_only else []
        known_names = query_names[:1] + measurement + query_names[1:]
        known_names += [name for name in known_only if name != "measurement_text"]
        known_smiles = str(record.get("smiles") or record.get("canonical_smiles") or "")
        return self.environment.get_template("prompt.jinja").render(
            known_smiles=known_smiles,
            query_smiles=query_smiles,
            known_fields=self._fields(record, known_names),
            query_fields=self._fields(record, query_names),
        ).strip()

    def prompt_fields(self, source_id: str) -> list[str]:
        binding = self._binding(source_id)
        return [*binding["query"], *binding["known_only"]]

    def canonical_endpoint_key(self, record: Mapping[str, Any]) -> str:
        endpoint = record.get("endpoint_name")
        if endpoint in (None, ""):
            return f"unspecified_endpoint:{record.get('source_id') or ''}"
        return canonicalize_endpoint(endpoint)


class V191CachedAssayReranker(CachedAssayReranker):
    def __init__(
        self,
        *,
        task_id: str,
        cache_path: str | Path,
        model: str | None = None,
        model_revision: str | None = None,
        allow_missing: bool = False,
        cache_mode: str = "read_only",
        **kwargs: Any,
    ):
        if cache_mode != "read_only":
            raise ValueError("V19.1 inference only accepts a finalized read-only cache")
        profile = model_profile(task_id)
        super().__init__(
            task_id=task_id,
            cache_path=cache_path,
            model=model or str(profile["model"]),
            model_revision=model_revision or str(profile["revision"]),
            renderer=V191PromptRenderer(task_id),
            profile_name=PROFILE_NAME,
            template_profile=TEMPLATE_PROFILE,
            query_context_policy=QUERY_CONTEXT_POLICY,
            allow_missing=allow_missing,
            **kwargs,
        )
