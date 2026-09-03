"""Build globally reconciled BBB assay-context and species mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.chembl_tool.common.llm_client import load_distillation_secret
from data.processing.evidence_library.shared.v1.clustered_auxiliary_mapping import (
    AuxiliaryExtractionSpec,
    DEFAULT_EMBEDDING_MODEL,
    build_clustered_auxiliary_mapping,
    reconciliation_plan,
)
from data.processing.evidence_library.versions.v7.prompts import PROMPT_ROOT
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_auxiliary_metadata import (
    DEFAULT_MAPPING_PATH,
    MAPPING_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[8]
DATA_ROOT = REPO_ROOT / "data/raw/starling/bbb_martins"
PROMPT_REGISTRY_PATH = PROMPT_ROOT / "auxiliary_canonicalization/bbb.json"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
PROMPT_VERSION = "bbb_martins_embedding_bucket_mapping.v1"


def _load_prompt_registry() -> dict:
    payload = json.loads(PROMPT_REGISTRY_PATH.read_text(encoding="utf-8"))
    if payload.get("registry_version") != "bbb_martins_auxiliary_prompts.v1":
        raise ValueError("BBB auxiliary prompt registry version mismatch")
    expected = {
        (source, output)
        for source in ("direct_bbb", "passive_permeability", "efflux_transport")
        for output in ("global_context", "global_species_context")
    }
    actual = {
        (source, output)
        for source, outputs in payload.get("prompts", {}).items()
        for output in outputs
    }
    if actual != expected:
        raise ValueError(f"BBB auxiliary prompt inventory mismatch: {actual}")
    return payload


def extraction_specs() -> tuple[AuxiliaryExtractionSpec, ...]:
    prompts = _load_prompt_registry()["prompts"]
    source_paths = {
        "direct_bbb": DATA_ROOT / "Direct_BBB/records.parquet",
        "passive_permeability": DATA_ROOT / "passive_permeability/extractions.parquet",
        "efflux_transport": DATA_ROOT / "efflux_transport/extractions.parquet",
    }
    output: list[AuxiliaryExtractionSpec] = []
    for source_id, sections in prompts.items():
        for output_field, section in sections.items():
            output.append(
                AuxiliaryExtractionSpec(
                    source_id=source_id,
                    input_path=source_paths[source_id],
                    input_column=section["input_column"],
                    output_field=output_field,
                    prompt=section["prompt"],
                    null_sentinel=(
                        "no species" if output_field == "global_species_context" else None
                    ),
                )
            )
    return tuple(output)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cluster-target-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.embedding_batch_size,
        args.cluster_target_size,
        args.workers,
        args.max_retries,
    ) < 1:
        parser.error("batch size, cluster size, workers, and retries must be positive")
    if args.retry_delay < 0:
        parser.error("retry delay cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = extraction_specs()
    if args.dry_run:
        print(
            json.dumps(
                reconciliation_plan(specs, cluster_target_size=args.cluster_target_size),
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    audit = build_clustered_auxiliary_mapping(
        specs=specs,
        output_path=args.output,
        mapping_version=MAPPING_VERSION,
        prompt_version=PROMPT_VERSION,
        api_key_loader=lambda: load_distillation_secret("OPENAI_API_KEY"),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        device=args.device,
        cluster_target_size=args.cluster_target_size,
        workers=args.workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
