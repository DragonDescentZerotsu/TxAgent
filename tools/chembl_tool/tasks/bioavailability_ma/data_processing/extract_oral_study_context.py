#!/usr/bin/env python3
"""Extract canonical species and biological matrix from oral study context."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from tools.chembl_tool.common.llm_client import openai_client


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data/starling_data/bioavailability_ma/Oral_AUC-Cmax_Exposure/extractions.parquet"
)
DEFAULT_OUTPUT = Path(__file__).with_name("oral_study_context_extractions.json")
DEFAULT_MODEL = "gpt-5.4"
PROMPT_VERSION = "oral_study_context_extraction.v1"
NULL_LIKE = {
    "",
    "-",
    "n/a",
    "na",
    "nan",
    "none",
    "not specified",
    "not stated",
    "null",
    "unknown",
    "unspecified",
}
SPECIES_LABELS = {
    "human",
    "rat",
    "mouse",
    "dog",
    "monkey",
    "rabbit",
    "pig",
    "horse",
    "other_animal",
    "multiple_species",
}
MATRIX_LABELS = {
    "plasma",
    "serum",
    "whole_blood",
    "urine",
    "tissue_or_other",
    "multiple_matrices",
}
OUTPUT_FIELDS = ("canonical_species_context", "canonical_biological_matrix")
SYSTEM_PROMPT = """Extract two fields from every supplied oral-exposure study_context.

Use only information explicitly stated in study_context. Do not infer species or matrix
from the drug, disease, route, study design, or outside knowledge.

canonical_species_context must be one of:
- human, rat, mouse, dog, monkey, rabbit, pig, horse
- other_animal: an explicit animal not covered above
- multiple_species: more than one explicit study species
- null: no study species is explicit

Normalize common aliases, including volunteers/subjects/patients -> human,
murine -> mouse, canine -> dog, porcine -> pig, and nonhuman primate or macaque
-> monkey. Ignore sex, strain, age, health status, and population qualifiers.

canonical_biological_matrix must be one of:
- plasma, serum, whole_blood, urine
- tissue_or_other: an explicit biological sampling matrix outside the four above
- multiple_matrices: more than one explicit sampling matrix
- null: no biological sampling matrix is explicit

Treat arterial/venous plasma and blood plasma as plasma. Do not map an unqualified
mention of blood sampling to whole_blood unless whole blood is explicit. A formulation,
food matrix, or assay medium is not a biological sampling matrix.

Return exactly one object for every supplied ID and no additional IDs. Use JSON null,
not a string such as "unknown" or "none".

Output format:
{"mapping":{"<input_id>":{"canonical_species_context":"human","canonical_biological_matrix":"plasma"}}}
"""


def _distinct_contexts(series: pd.Series) -> list[str]:
    values = {
        str(value).strip()
        for value in series.dropna()
        if str(value).strip().casefold() not in NULL_LIKE
    }
    return sorted(values, key=lambda value: (value.casefold(), value))


def _validate_response(
    content: str | None, item_ids: set[str]
) -> dict[str, dict[str, str | None]]:
    try:
        payload = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"mapping"}:
        raise ValueError("response must contain only the mapping object")
    mapping = payload["mapping"]
    if not isinstance(mapping, dict) or set(mapping) != item_ids:
        raise ValueError("response IDs do not exactly match the requested IDs")

    validated: dict[str, dict[str, str | None]] = {}
    for item_id, result in mapping.items():
        if not isinstance(result, dict) or set(result) != set(OUTPUT_FIELDS):
            raise ValueError(f"{item_id} must contain exactly {OUTPUT_FIELDS}")
        species = result["canonical_species_context"]
        matrix = result["canonical_biological_matrix"]
        if isinstance(species, str) and species.strip().casefold() == "null":
            species = None
        if isinstance(matrix, str) and matrix.strip().casefold() == "null":
            matrix = None
        if species is not None and species not in SPECIES_LABELS:
            raise ValueError(f"invalid species label for {item_id}: {species!r}")
        if matrix is not None and matrix not in MATRIX_LABELS:
            raise ValueError(f"invalid matrix label for {item_id}: {matrix!r}")
        validated[item_id] = {
            "canonical_species_context": species,
            "canonical_biological_matrix": matrix,
        }
    return validated


def _batch_identity(
    contexts: list[str], *, model: str, reasoning_effort: str
) -> str:
    payload = {
        "model": model,
        "prompt": SYSTEM_PROMPT,
        "prompt_version": PROMPT_VERSION,
        "reasoning_effort": reasoning_effort,
        "values": contexts,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _query_batch(
    client: OpenAI,
    contexts: list[str],
    *,
    model: str,
    reasoning_effort: str,
    max_retries: int,
    retry_delay: float,
) -> dict[str, dict[str, str | None]]:
    items = {f"v{index:04d}": value for index, value in enumerate(contexts)}
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            prompt = SYSTEM_PROMPT
            if attempt > 1:
                prompt += "\nYour previous response was invalid. Follow the schema exactly."
            request: dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "items": [
                                    {"id": item_id, "study_context": context}
                                    for item_id, context in items.items()
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            }
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**request)
            return _validate_response(
                response.choices[0].message.content, set(items)
            )
        except Exception as exc:
            last_error = exc
            if getattr(exc, "status_code", None) in {400, 401, 403, 404}:
                break
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    raise RuntimeError(
        f"batch failed after {attempts} attempts: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                cached[record["identity"]] = record
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid checkpoint line {line_number}: {exc}"
                ) from exc
    return cached


def _extract(
    contexts: list[str],
    *,
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    batch_size: int,
    workers: int,
    max_retries: int,
    retry_delay: float,
    checkpoint_path: Path,
) -> dict[str, dict[str, str | None]]:
    batches = [
        contexts[start : start + batch_size]
        for start in range(0, len(contexts), batch_size)
    ]
    cached = _load_checkpoint(checkpoint_path)
    results: dict[str, dict[str, dict[str, str | None]]] = {}
    pending: list[tuple[str, list[str]]] = []
    for batch in batches:
        identity = _batch_identity(
            batch, model=model, reasoning_effort=reasoning_effort
        )
        record = cached.get(identity)
        if record is None:
            pending.append((identity, batch))
            continue
        results[identity] = _validate_response(
            json.dumps({"mapping": record["mapping"]}),
            {f"v{index:04d}" for index in range(len(batch))},
        )

    print(
        f"contexts={len(contexts):,} batches={len(batches):,} "
        f"cached={len(batches) - len(pending):,} pending={len(pending):,}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _query_batch,
                client,
                batch,
                model=model,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_delay=retry_delay,
            ): (identity, batch)
            for identity, batch in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            identity, _ = futures[future]
            mapping = future.result()
            results[identity] = mapping
            line = json.dumps(
                {"identity": identity, "mapping": mapping},
                ensure_ascii=False,
                sort_keys=True,
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"completed={completed:,}/{len(pending):,}", flush=True)

    output: dict[str, dict[str, str | None]] = {}
    for batch in batches:
        identity = _batch_identity(
            batch, model=model, reasoning_effort=reasoning_effort
        )
        mapping = results[identity]
        for index, context in enumerate(batch):
            output[context] = mapping[f"v{index:04d}"]
    if set(output) != set(contexts):
        raise ValueError("final study_context coverage mismatch")
    return output


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_path}; pass --overwrite")
    frame = pd.read_parquet(input_path, columns=["study_context"])
    contexts = _distinct_contexts(frame["study_context"])
    client = openai_client(
        api_key_env=args.api_key_env,
        base_url=args.base_url or None,
    )
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")
    mapping = _extract(
        contexts,
        client=client,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        batch_size=args.batch_size,
        workers=args.workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        checkpoint_path=checkpoint_path,
    )
    counts = {
        field: {
            str(label) if label is not None else "null": sum(
                value[field] == label for value in mapping.values()
            )
            for label in sorted(
                {value[field] for value in mapping.values()},
                key=lambda value: "" if value is None else value,
            )
        }
        for field in OUTPUT_FIELDS
    }
    payload = {
        "artifact_version": PROMPT_VERSION,
        "input": str(input_path),
        "input_rows": len(frame),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "source_column": "study_context",
        "distinct_contexts": len(contexts),
        "allowed_values": {
            "canonical_species_context": sorted(SPECIES_LABELS),
            "canonical_biological_matrix": sorted(MATRIX_LABELS),
        },
        "counts": counts,
        "mapping": mapping,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(counts, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.workers, args.max_retries) < 1:
        parser.error("batch size, workers, and retries must be positive")
    if args.retry_delay < 0:
        parser.error("retry delay cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
