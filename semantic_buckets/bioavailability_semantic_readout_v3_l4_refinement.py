"""Re-cluster only Oral L4 pair-bucket atoms from the immutable V3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
from typing import Any

import httpx
import pandas as pd

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import bioavailability_semantic_readout_v3 as v3
from semantic_buckets.artifacts import generation_root
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "bioavailability_semantic_readout.v3.l4_refinement.v1"
GROUP_KEY = "L4|fa"
LEVEL = "L4"
SOURCE = "fa"
MERGE_ROUNDS = 2
BASE = generation_root(
    "bioavailability_ma", "bioavailability_semantic_readout_v10_v3"
)
DEFAULT_OUTPUT = generation_root(
    "bioavailability_ma",
    "bioavailability_semantic_readout_v10_v3_l4_refinement_v1",
)
DEFAULT_REVIEW = Path(
    "/tmp/bioavailability_semantic_readout_v3_l4_refinement_v1_prompt_review"
)
PROMPT_ROOT = (
    Path(__file__).with_name("prompts")
    / "bioavailability_semantic_readout_v3_l4_refinement_v1"
)
EXPECTED_BASE_MAP_SHA256 = (
    "e811fbcf3b3b770711906cc10c2a291f4ec4dc119000ed58ed57282205b45b0d"
)
EXPECTED_ATOMS = 102_316
EXPECTED_L4_ATOMS = 5_056
EXPECTED_L4_TERMINALS = 240
EXPECTED_L4_BUCKETS = 13


def configure_core() -> None:
    v3.configure_core()
    core.PROMPT_ROOT = PROMPT_ROOT
    core.BATCH_SEED_VERSION = VERSION


def load_base() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    dict[str, Any],
]:
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != v3.VERSION:
        raise ValueError("base semantic artifact is not Oral V3")
    if manifest.get("status") != "awaiting_cross_source_mapping":
        raise ValueError("base Oral V3 artifact is not source-local complete")
    base_map_path = BASE / "source_semantic_bucket_map.parquet"
    if core._file_sha256(base_map_path) != EXPECTED_BASE_MAP_SHA256:
        raise ValueError("base Oral V3 source map hash changed")

    atoms = pd.read_parquet(BASE / "input_atoms.parquet")
    base_map = pd.read_parquet(base_map_path)
    sample_cards = core._read_gzip_json(BASE / "sample_cards.json.gz")
    state = json.loads((BASE / "semantic_state.json").read_text(encoding="utf-8"))
    group = state["groups"][GROUP_KEY]
    terminals = {
        str(bucket): [str(atom) for atom in members]
        for bucket, members in group["terminal"].items()
    }

    if len(atoms) != EXPECTED_ATOMS or atoms["atom_id"].duplicated().any():
        raise ValueError("base Oral V3 atom coverage changed")
    if len(base_map) != EXPECTED_ATOMS or base_map["atom_id"].duplicated().any():
        raise ValueError("base Oral V3 source map coverage changed")
    if set(base_map["atom_id"]) != set(atoms["atom_id"]):
        raise ValueError("base Oral V3 source map does not match its atoms")
    if group.get("active") or not group.get("finished"):
        raise ValueError("base Oral L4 group is not terminal")
    if set(group["consumed_columns"]) != set(core.REFINEMENT_COLUMNS[SOURCE]):
        raise ValueError("base Oral L4 did not consume every allowed column")
    terminal_atoms = [atom for members in terminals.values() for atom in members]
    expected_l4 = set(
        atoms.loc[
            atoms["level"].eq(LEVEL) & atoms["source_id"].eq(SOURCE), "atom_id"
        ]
    )
    if (
        len(terminals) != EXPECTED_L4_TERMINALS
        or len(terminal_atoms) != len(set(terminal_atoms))
        or set(terminal_atoms) != expected_l4
        or len(expected_l4) != EXPECTED_L4_ATOMS
        or len(group["final_buckets"]) != EXPECTED_L4_BUCKETS
    ):
        raise ValueError("base Oral L4 terminal lineage changed")
    atomic_buckets = {
        core.bucket_id([str(atom)]): [str(atom)] for atom in sorted(expected_l4)
    }
    return atoms, base_map, sample_cards, atomic_buckets, manifest


def replace_l4_mapping(
    base_map: pd.DataFrame, buckets: dict[str, list[str]]
) -> pd.DataFrame:
    target = base_map["level"].eq(LEVEL) & base_map["source_id"].eq(SOURCE)
    expected = set(base_map.loc[target, "atom_id"])
    atom_to_bucket = {
        atom: bucket for bucket, members in buckets.items() for atom in members
    }
    if len(atom_to_bucket) != sum(map(len, buckets.values())):
        raise ValueError("refined Oral L4 buckets overlap")
    if set(atom_to_bucket) != expected:
        raise ValueError("refined Oral L4 coverage changed")
    result = base_map.copy()
    result.loc[target, "source_semantic_bucket_id"] = result.loc[
        target, "atom_id"
    ].map(atom_to_bucket)
    if len(result) != len(base_map) or result["atom_id"].duplicated().any():
        raise ValueError("derived source map changed atom coverage")
    if not result.loc[~target].equals(base_map.loc[~target]):
        raise ValueError("derived source map changed a non-L4 assignment")
    return result


def endpoint_receipts() -> tuple[int, list[dict[str, Any]]]:
    receipts = []
    maximum_context = None
    for spec in core.ENDPOINTS:
        base_url = str(spec["base_url"])
        health = httpx.get(base_url.rsplit("/v1", 1)[0] + "/health", timeout=10)
        health.raise_for_status()
        models_response = httpx.get(f"{base_url}/models", timeout=10)
        models_response.raise_for_status()
        models = models_response.json().get("data", [])
        matching = [row for row in models if row.get("id") == core.MODEL]
        if len(matching) != 1:
            raise ValueError(
                f"endpoint {spec['name']} does not uniquely serve {core.MODEL}"
            )
        endpoint_context = int(matching[0].get("max_model_len") or 0)
        if endpoint_context <= 0:
            raise ValueError(f"endpoint {spec['name']} omitted max_model_len")
        maximum_context = (
            endpoint_context
            if maximum_context is None
            else min(maximum_context, endpoint_context)
        )
        receipts.append(
            {
                "name": spec["name"],
                "base_url": base_url,
                "health_status": health.status_code,
                "model": core.MODEL,
                "max_model_len": endpoint_context,
            }
        )
    if maximum_context is None:
        raise ValueError("no DeepSeek endpoints are configured")
    return maximum_context, receipts


def first_round_prompts(
    buckets: dict[str, list[str]],
    lookup: dict[str, dict[str, Any]],
    sample_cards: dict[str, dict[str, Any]],
) -> list[tuple[list[str], str]]:
    phase = "l4_refinement|round1"
    batches = core.deterministic_batches(
        list(buckets), seed=f"{VERSION}|{phase}"
    )
    return [
        (
            batch,
            core._render(
                "merge_buckets",
                core._merge_bucket_payload(
                    LEVEL,
                    SOURCE,
                    {bucket: buckets[bucket] for bucket in batch},
                    lookup,
                    sample_cards,
                ),
            ),
        )
        for batch in batches
    ]


def prepare_prompt_review(output: Path = DEFAULT_REVIEW) -> dict[str, Any]:
    configure_core()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prompt review directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atoms, _base_map, sample_cards, atomic_buckets, base_manifest = load_base()
    maximum_context, endpoints = endpoint_receipts()
    prompts = first_round_prompts(
        atomic_buckets, core._atom_lookup(atoms), sample_cards
    )
    prompt_rows = []
    for index, (batch, prompt) in enumerate(prompts):
        path = output / f"merge_l4_batch_{index:04d}.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        input_tokens = core._token_count(prompt)
        prompt_rows.append(
            {
                "batch": index,
                "path": path.name,
                "sha256": core._file_sha256(path),
                "bucket_count": len(batch),
                "input_tokens": input_tokens,
                "fits_model_context": (
                    input_tokens + core.HIGH_MAX_TOKENS <= maximum_context
                ),
            }
        )
    if not all(row["fits_model_context"] for row in prompt_rows):
        raise ValueError("an Oral L4 merge prompt exceeds the model context")

    template = PROMPT_ROOT / "merge_buckets.jinja"
    manifest = {
        "version": f"{VERSION}.prompt_review.v1",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "model": core.MODEL,
        "endpoint_receipts": endpoints,
        "max_model_len": maximum_context,
        "base_v3": {
            "path": str(BASE),
            "manifest_sha256": core._file_sha256(BASE / "manifest.json"),
            "source_map_sha256": EXPECTED_BASE_MAP_SHA256,
            "request_summary": base_manifest["request_summary"],
        },
        "scope": {
            "group": GROUP_KEY,
            "input_pair_bucket_atoms": len(atomic_buckets),
            "base_v3_terminal_buckets": EXPECTED_L4_TERMINALS,
            "atoms": EXPECTED_L4_ATOMS,
            "existing_final_buckets": EXPECTED_L4_BUCKETS,
            "merge_rounds": MERGE_ROUNDS,
            "merge_batch_size": core.MERGE_BATCH_SIZE,
            "sample_limit": core.SAMPLE_LIMIT,
            "reasoning_effort": "high",
        },
        "prompt_template": {
            "path": str(template),
            "sha256": core._file_sha256(template),
        },
        "first_round_prompts": prompt_rows,
    }
    write_json_atomic(output / "manifest.json", manifest)
    lines = [
        "# Oral L4 V3 refinement prompt review",
        "",
        "No completion requests were made.",
        "",
        "| Batch | Buckets | Input tokens | Sendable |",
        "|---:|---:|---:|:---:|",
    ]
    lines.extend(
        f"| {row['batch']} | {row['bucket_count']} | {row['input_tokens']:,} | "
        f"{'yes' if row['fits_model_context'] else 'no'} |"
        for row in prompt_rows
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_dominance_report(
    output: Path, derived: pd.DataFrame, atoms: pd.DataFrame
) -> dict[str, Any]:
    joined = derived.merge(
        atoms[["atom_id", "record_count", "values_json"]],
        on="atom_id",
        how="left",
        validate="one_to_one",
    )
    l4 = joined[joined["level"].eq(LEVEL) & joined["source_id"].eq(SOURCE)]
    rows = []
    for bucket, members in l4.groupby("source_semantic_bucket_id", sort=True):
        endpoints = sorted(
            {
                str(json.loads(value)["canonical_endpoint_concept"])
                for value in members["values_json"]
            }
        )
        rows.append(
            {
                "source_semantic_bucket_id": str(bucket),
                "pair_bucket_count": len(members),
                "record_count": int(members["record_count"].sum()),
                "endpoint_concepts_json": core._canonical_json(endpoints),
            }
        )
    report = pd.DataFrame(rows)
    report["level_pair_bucket_fraction"] = (
        report["pair_bucket_count"] / report["pair_bucket_count"].sum()
    )
    report["level_record_fraction"] = (
        report["record_count"] / report["record_count"].sum()
    )
    report["review_required"] = (
        (report["level_pair_bucket_fraction"] > 0.10)
        | (report["level_record_fraction"] > 0.10)
    )
    report = report.sort_values(
        ["pair_bucket_count", "record_count"], ascending=False
    )
    table = output / "l4_dominance_audit.parquet"
    report.to_parquet(table, index=False)
    lines = [
        "# Oral L4 V3 refinement dominance audit",
        "",
        "| Bucket | Pair buckets | Pair-bucket share | Records | Record share | Review |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in report.itertuples(index=False):
        lines.append(
            f"| `{row.source_semantic_bucket_id}` | {row.pair_bucket_count:,} | "
            f"{row.level_pair_bucket_fraction:.1%} | {row.record_count:,} | "
            f"{row.level_record_fraction:.1%} | "
            f"{'yes' if row.review_required else 'no'} |"
        )
    markdown = output / "l4_dominance_audit.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "bucket_count": len(report),
        "review_required_count": int(report["review_required"].sum()),
        "table": {"path": table.name, "sha256": core._file_sha256(table)},
        "report": {
            "path": markdown.name,
            "sha256": core._file_sha256(markdown),
        },
    }


def run(
    output: Path,
    *,
    review_manifest_path: Path,
    approved_review_sha256: str,
    parallelism: int,
) -> dict[str, Any]:
    configure_core()
    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    if core._file_sha256(review_manifest_path) != approved_review_sha256:
        raise ValueError("approved prompt review hash mismatch")
    review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review.get("status") != "awaiting_user_prompt_approval":
        raise ValueError("prompt review is not awaiting approval")
    if review.get("completion_requests_made") != 0:
        raise ValueError("prompt review unexpectedly made completion requests")
    if review["prompt_template"]["sha256"] != core._file_sha256(
        PROMPT_ROOT / "merge_buckets.jinja"
    ):
        raise ValueError("reviewed Oral L4 prompt changed")

    atoms, base_map, sample_cards, atomic_buckets, _base_manifest = load_base()
    expected_prompts = first_round_prompts(
        atomic_buckets, core._atom_lookup(atoms), sample_cards
    )
    if len(expected_prompts) != len(review["first_round_prompts"]):
        raise ValueError("reviewed Oral L4 batch schedule changed")
    for (_batch, prompt), row in zip(
        expected_prompts, review["first_round_prompts"], strict=True
    ):
        path = review_manifest_path.parent / row["path"]
        if (
            core._file_sha256(path) != row["sha256"]
            or path.read_text(encoding="utf-8") != prompt + "\n"
        ):
            raise ValueError("reviewed Oral L4 rendered prompt changed")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("approved_prompt_review_sha256") != approved_review_sha256:
            raise ValueError("existing output uses a different prompt review")
        if manifest.get("status") == "complete":
            return manifest
    elif any(output.iterdir()):
        raise FileExistsError(f"new Oral L4 output directory is not empty: {output}")
    else:
        manifest = {
            "version": VERSION,
            "status": "running",
            "approved_prompt_review": str(review_manifest_path),
            "approved_prompt_review_sha256": approved_review_sha256,
            "base_v3_source_map_sha256": EXPECTED_BASE_MAP_SHA256,
            "requests": "requests.sqlite3",
        }
        write_json_atomic(manifest_path, manifest)

    maximum_context, _ = core._endpoint_contract()
    client, endpoint_receipts, total_parallelism = core._build_completion_pool(
        parallelism
    )
    slots = threading.BoundedSemaphore(total_parallelism)
    connection = core._request_database(output / "requests.sqlite3")
    try:
        lookup = core._atom_lookup(atoms)
        buckets = atomic_buckets
        for round_number in range(1, MERGE_ROUNDS + 1):
            buckets = core._merge_bucket_round(
                connection,
                level=LEVEL,
                source=SOURCE,
                buckets=buckets,
                lookup=lookup,
                sample_cards=sample_cards,
                phase=f"l4_refinement|round{round_number}",
                maximum_context=maximum_context,
                parallelism=total_parallelism,
                client=client,
                request_slots=slots,
            )
        derived = replace_l4_mapping(base_map, buckets)
        map_path = output / "source_semantic_bucket_map.parquet"
        derived.to_parquet(map_path, index=False)
        dominance = write_dominance_report(output, derived, atoms)
        request_row = connection.execute(
            """
            SELECT count(*) AS requests,
                   sum(status='complete') AS complete,
                   sum(status='failed') AS failed,
                   sum(input_tokens) AS input_tokens,
                   sum(output_tokens) AS output_tokens
            FROM requests
            """
        ).fetchone()
        request_summary = {key: int(request_row[key] or 0) for key in request_row.keys()}
        if request_summary["complete"] != request_summary["requests"]:
            raise RuntimeError("Oral L4 refinement request ledger is incomplete")
        manifest.update(
            {
                "status": "complete",
                "model": core.MODEL,
                "reasoning_effort": "high",
                "per_endpoint_parallelism": parallelism,
                "total_parallelism": total_parallelism,
                "endpoint_pool_at_start": endpoint_receipts,
                "endpoint_pool_final_snapshot": client.snapshot(),
                "request_summary": request_summary,
                "input": {
                    "pair_bucket_atoms": len(atomic_buckets),
                    "base_v3_terminal_buckets": EXPECTED_L4_TERMINALS,
                    "l4_atoms": EXPECTED_L4_ATOMS,
                    "existing_final_buckets": EXPECTED_L4_BUCKETS,
                },
                "output": {
                    "l4_bucket_count": len(buckets),
                    "source_map": {
                        "path": map_path.name,
                        "rows": len(derived),
                        "sha256": core._file_sha256(map_path),
                    },
                    "dominance_audit": dominance,
                },
            }
        )
        write_json_atomic(manifest_path, manifest)
        return manifest
    except Exception:
        manifest["status"] = "incomplete"
        write_json_atomic(manifest_path, manifest)
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-prompts", help="render Oral L4 prompts without completions"
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_REVIEW)
    execute = subparsers.add_parser(
        "run", help="run the approved V3-derived Oral L4 re-merge"
    )
    execute.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    execute.add_argument("--review-manifest", type=Path, required=True)
    execute.add_argument("--approved-review-sha256", required=True)
    execute.add_argument("--parallelism", type=int, default=64)
    args = parser.parse_args()
    if args.command == "prepare-prompts":
        result = prepare_prompt_review(args.output)
    else:
        result = run(
            args.output,
            review_manifest_path=args.review_manifest,
            approved_review_sha256=args.approved_review_sha256,
            parallelism=args.parallelism,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
