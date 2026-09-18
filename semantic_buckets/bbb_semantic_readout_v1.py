"""Run the reviewed semantic-bucket workflow on the BBB V10 library."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined
import pandas as pd
import pyarrow.parquet as pq

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets.artifacts import generation_root
from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.tasks.bbb_martins import source_family_purity as family_policy
from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
    BBBSourceFamilyClassifier,
    CLASSIFIER_COLUMNS,
    DIRECT_GROUP,
    EFFLUX_GROUP,
    INFLUX_GROUP,
    NEAR_DIRECT_GROUP,
    PASSIVE_GROUP,
    PURITY_VERSION,
    load_near_direct_reviews,
)


VERSION = "bbb_semantic_readout.v1"
BATCH_SEED_VERSION = VERSION
BASE_URL = core.BASE_URL
ROOT = Path(__file__).resolve().parents[1]
V10_RECORDS = ROOT / "data/evidence_libraries/bbb_martins/v10/03_pair_buckets/records.parquet"
INPUT_ROOT = generation_root("bbb_martins", "bbb_semantic_level_map_v10_v1")
RECORD_MAP = INPUT_ROOT / "record_relevance_map.parquet"
DEFAULT_OUTPUT = generation_root("bbb_martins", "bbb_semantic_readout_v10_v1")
PROMPT_ROOT = Path(__file__).with_name("prompts") / "bbb_semantic_readout_v1"
CROSS_SOURCE_PROMPT_ROOT = (
    Path(__file__).with_name("prompts") / "bbb_semantic_cross_source_v1"
)
RANKING_TEMPLATE = Path(__file__).with_name("prompts") / "bbb_relevance_bucket_level_v3.jinja"
REQUEST_TIMEOUT_S = 3_600
MAX_ATTEMPTS = 12
ENDPOINTS = (
    {
        "name": "dgx024",
        "base_url": "http://10.218.137.44:50001/v1",
        "provider": "local",
        "credential_env": "",
    },
    {
        "name": "dgx027",
        "base_url": "http://dgx027:50001/v1",
        "provider": "local",
        "credential_env": "",
    },
)
LEVELS = ("L2", "L3", "L4", "L5")
EXPECTED_LEVEL_COUNTS = {
    "L1": 8_527,
    "L2": 267_544,
    "L3": 14_693,
    "L4": 164_050,
    "L5": 43_693,
}
PAIR_COLUMNS = {
    "direct_bbb": (
        "canonical_endpoint_name",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_assay_context",
        "canonical_species_context",
        "condition_group",
    ),
    "efflux_transport": (
        "canonical_endpoint_name",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_transporter_identifier",
        "canonical_evidence_type",
        "canonical_assay_context",
        "canonical_species_context",
    ),
    "influx_transport": (
        "canonical_endpoint_name",
        "canonical_unit_text",
        "canonical_transport_mechanism",
        "canonical_kinetic_symbol",
    ),
    "passive_permeability": (
        "canonical_endpoint_name",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_assay_type",
        "canonical_assay_context",
        "canonical_species_context",
    ),
}
INITIAL_COLUMNS = {
    "direct_bbb": (
        "canonical_endpoint_name",
        "canonical_measurement_scale_id",
        "canonical_assay_context",
    ),
    "efflux_transport": (
        "canonical_endpoint_name",
        "canonical_measurement_scale_id",
        "canonical_transporter_identifier",
        "canonical_evidence_type",
    ),
    "influx_transport": (
        "canonical_endpoint_name",
        "canonical_transport_mechanism",
        "canonical_kinetic_symbol",
    ),
    "passive_permeability": (
        "canonical_endpoint_name",
        "canonical_measurement_scale_id",
        "canonical_assay_type",
    ),
}
SAMPLE_CARD_COLUMNS = {
    "direct_bbb": (*PAIR_COLUMNS["direct_bbb"], "assay_model"),
    "efflux_transport": (*PAIR_COLUMNS["efflux_transport"], "assay_system", "perturbation"),
    "influx_transport": (
        *PAIR_COLUMNS["influx_transport"],
        "mediator_name",
        "mediator_identifier",
        "evidence_basis",
        "assay_model",
    ),
    "passive_permeability": (
        *PAIR_COLUMNS["passive_permeability"],
        "assay_type",
        "biological_system",
    ),
}
REFINEMENT_COLUMNS = PAIR_COLUMNS
PROMPT_DIMENSION_COLUMNS = PAIR_COLUMNS
DEFER_TECHNICAL_BRANCHES = False
SEMANTIC_SIZE_REVIEW_REQUIRED = False
INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS = True
INCLUDE_DOWNSTREAM_PROMPT_REVIEW = True
COLUMN_DESCRIPTIONS = {
    "canonical_endpoint_name": "normalized BBB, brain-exposure, permeability, or transport endpoint",
    "canonical_unit_text": "normalized measurement unit",
    "canonical_measurement_scale_id": "measurement scale or response representation",
    "canonical_assay_context": "assay system or experimental context",
    "canonical_species_context": "species or population in which the readout was measured",
    "condition_group": "normalized condition identity attached to a direct-source record",
    "canonical_transporter_identifier": "normalized efflux transporter identity",
    "canonical_evidence_type": "transport evidence design such as inhibition, genetics, or exposure change",
    "canonical_transport_mechanism": "normalized influx or uptake mechanism",
    "canonical_kinetic_symbol": "kinetic quantity such as Km, Vmax, or uptake rate",
    "canonical_assay_type": "passive-permeability assay class",
}
LEVEL_BY_GROUP = {
    DIRECT_GROUP: "L1",
    NEAR_DIRECT_GROUP: "L2",
    PASSIVE_GROUP: "L3",
    EFFLUX_GROUP: "L4",
    INFLUX_GROUP: "L5",
}
FAMILY_BY_GROUP = {
    DIRECT_GROUP: "direct_bbb_evidence",
    NEAR_DIRECT_GROUP: "central_functional_access_proxy",
    PASSIVE_GROUP: "passive_permeability",
    EFFLUX_GROUP: "efflux_transport",
    INFLUX_GROUP: "influx_transport",
}


def configure_core() -> None:
    """Apply BBB configuration to the shared, tested workflow engine."""

    core.VERSION = VERSION
    core.BATCH_SEED_VERSION = BATCH_SEED_VERSION
    core.BASE_URL = BASE_URL
    core.TASK_NAME = "BBB_Martins"
    core.RECORD_MAP = RECORD_MAP
    core.RECORDS = V10_RECORDS
    core.PROMPT_ROOT = PROMPT_ROOT
    core.CROSS_SOURCE_PROMPT_ROOT = CROSS_SOURCE_PROMPT_ROOT
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.ENDPOINTS = ENDPOINTS
    core.LEVELS = LEVELS
    core.EXPECTED_RECORD_COUNT = sum(EXPECTED_LEVEL_COUNTS[level] for level in LEVELS)
    manifest_path = INPUT_ROOT / "manifest.json"
    core.EXPECTED_ATOM_COUNT = (
        json.loads(manifest_path.read_text())["pair_bucket_atoms"]
        if manifest_path.is_file()
        else None
    )
    core.CROSS_SOURCE_LEVELS = LEVELS
    core.PAIR_KEY_OPTIONAL_TRAILING_COLUMNS = {}
    core.SELECTOR_PROFILE_LIMIT = 24
    core.REQUEST_TIMEOUT_S = REQUEST_TIMEOUT_S
    # Preserve four failed PARCC gateway receipts while allowing up to eight
    # direct-DGX attempts without rewriting attempt provenance.
    core.MAX_ATTEMPTS = MAX_ATTEMPTS
    core.PAIR_COLUMNS = PAIR_COLUMNS
    core.REFINEMENT_COLUMNS = REFINEMENT_COLUMNS
    core.PROMPT_DIMENSION_COLUMNS = PROMPT_DIMENSION_COLUMNS
    core.INITIAL_COLUMNS = INITIAL_COLUMNS
    core.SAMPLE_CARD_COLUMNS = SAMPLE_CARD_COLUMNS
    core.COLUMN_DESCRIPTIONS = COLUMN_DESCRIPTIONS
    core.DEFER_TECHNICAL_BRANCHES = DEFER_TECHNICAL_BRANCHES
    core.SEMANTIC_SIZE_REVIEW_REQUIRED = SEMANTIC_SIZE_REVIEW_REQUIRED
    core.INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS = (
        INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS
    )
    core.INCLUDE_DOWNSTREAM_PROMPT_REVIEW = INCLUDE_DOWNSTREAM_PROMPT_REVIEW


def build_input(output: Path = INPUT_ROOT) -> dict[str, Any]:
    """Replay the frozen family classifier and publish a V10 level ledger."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"BBB semantic input directory is not empty: {output}")
    available = set(pq.read_schema(V10_RECORDS).names)
    columns = list(dict.fromkeys(
        [column for column in CLASSIFIER_COLUMNS if column in available]
        + ["source_row_uid", "retrieval_source_id", "pair_bucket_key"]
    ))
    records = pq.read_table(V10_RECORDS, columns=columns).to_pylist()
    uids = [str(row["source_row_uid"]) for row in records]
    record_ids = [str(row["canonical_record_id"]) for row in records]
    if len(uids) != len(set(uids)) or len(record_ids) != len(set(record_ids)):
        raise ValueError("BBB V10 record and source-row identities must be unique")
    vote_indices = {
        int(row["source_index"])
        for row in records
        if row.get("retrieval_source_id") == "direct_vote"
        and row.get("source_index") is not None
    }
    classifier = BBBSourceFamilyClassifier(load_near_direct_reviews(), vote_indices)
    counts: Counter[str] = Counter()
    mapped = []
    for index, row in enumerate(records, start=1):
        move = classifier.classify_target(row)
        level = LEVEL_BY_GROUP[move.new_group]
        counts[level] += 1
        if level in LEVELS:
            source = str(row["source_id"])
            pair_values = dict(zip(PAIR_COLUMNS[source], json.loads(row["pair_bucket_key"])[1:]))
            identity = {"source_id": source}
            identity.update({column: str(pair_values[column]) for column in INITIAL_COLUMNS[source]})
            relevance_bucket = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
            mapped.append(
                {
                    "canonical_record_id": str(row["canonical_record_id"]),
                    "source_row_uid": str(row["source_row_uid"]),
                    "source_id": source,
                    "level": level,
                    "source_group_id": move.new_group,
                    "family_key": FAMILY_BY_GROUP[move.new_group],
                    "pair_bucket_key": str(row["pair_bucket_key"]),
                    "relevance_bucket": relevance_bucket,
                    "node_key": core._canonical_json(
                        {"level": level, "relevance_bucket": relevance_bucket}
                    ),
                }
            )
        if index % 100_000 == 0:
            print(f"classified {index}/{len(records)} BBB V10 records", flush=True)
    if dict(sorted(counts.items())) != EXPECTED_LEVEL_COUNTS:
        raise ValueError(f"BBB V10 level counts drifted: {dict(sorted(counts.items()))}")
    frame = pd.DataFrame(mapped)
    atom_columns = ["level", "source_id", "pair_bucket_key", "node_key"]
    if frame.groupby(["level", "source_id", "pair_bucket_key"])["node_key"].nunique().gt(1).any():
        raise ValueError("one BBB V10 pair bucket maps to multiple initial semantic buckets")
    atom_count = len(frame.drop_duplicates(atom_columns))
    output.mkdir(parents=True, exist_ok=False)
    record_map = output / "record_relevance_map.parquet"
    frame.to_parquet(record_map, index=False)
    manifest = {
        "version": "bbb_semantic_level_map_v10.v1",
        "status": "complete",
        "task_id": "bbb_martins",
        "family_policy_version": PURITY_VERSION,
        "family_policy_code": {
            "path": str(Path(family_policy.__file__).resolve()),
            "sha256": core._file_sha256(Path(family_policy.__file__).resolve()),
        },
        "near_direct_review": {
            "path": str(family_policy.DEFAULT_REVIEW_LEDGER),
            "sha256": core._file_sha256(family_policy.DEFAULT_REVIEW_LEDGER),
        },
        "input": str(V10_RECORDS),
        "input_sha256": core._file_sha256(V10_RECORDS),
        "record_count": len(records),
        "level_counts": dict(sorted(counts.items())),
        "rankable_record_count": len(frame),
        "pair_bucket_atoms": atom_count,
        "record_map": record_map.name,
        "record_map_sha256": core._file_sha256(record_map),
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _ranking_bucket_payload(
    bucket: str,
    members: list[str],
    lookup: dict[str, dict[str, Any]],
    sample_cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = str(lookup[members[0]]["source_id"])
    dimensions = {
        column: sorted({str(lookup[atom]["values"][column]) for atom in members})
        for column in core.PROMPT_DIMENSION_COLUMNS[source]
    }
    return {
        "source": source,
        "identity": {"source_components": [{"source_id": source, "canonical_dimensions": dimensions}]},
        "sample_records": [sample_cards[atom] for atom in sorted(members)[: core.SAMPLE_LIMIT]],
    }


def prepare_review(output: Path) -> dict[str, Any]:
    """Render representative semantic and ranking prompts without completions."""

    configure_core()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prompt review directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atoms, sample_cards = core._load_atoms()
    lookup = core._atom_lookup(atoms)
    grouped = core.initial_buckets(atoms)
    rendered: dict[str, str] = {}
    examples: dict[str, tuple[str, dict[str, list[str]]]] = {}
    for source in sorted(PAIR_COLUMNS):
        level, buckets = max(
            ((level, buckets) for (level, found), buckets in grouped.items() if found == source),
            key=lambda item: len(item[1]),
        )
        examples[source] = (level, buckets)
        rendered[f"select_column_{source}"] = core._render(
            "select_column",
            core._column_selection_payload(level, source, buckets, lookup, INITIAL_COLUMNS[source]),
        )
        candidate = next(
            (
                (bucket, column)
                for bucket, members in sorted(buckets.items())
                for column in core.REFINEMENT_COLUMNS[source]
                if column not in INITIAL_COLUMNS[source]
                and len({lookup[atom]["values"][column] for atom in members}) > 1
            ),
            None,
        )
        if candidate is not None:
            bucket, column = candidate
            payload, _ = core._value_payload(
                bucket,
                buckets[bucket],
                column,
                lookup,
                context_columns=INITIAL_COLUMNS[source],
            )
            payload.update({"task": "BBB_Martins", "level": level, "source_id": source})
            rendered[f"split_bucket_{source}"] = core._render("split_bucket", payload)
            rendered[f"merge_values_{source}"] = core._render("merge_values", payload)
    merge_source = max(examples, key=lambda source: len(examples[source][1]))
    merge_level, merge_buckets = examples[merge_source]
    selected = dict(list(sorted(merge_buckets.items()))[: core.MERGE_BATCH_SIZE])
    rendered["merge_buckets_40"] = core._render(
        "merge_buckets",
        core._merge_bucket_payload(merge_level, merge_source, selected, lookup, sample_cards),
    )
    ranking_level = max(grouped, key=lambda key: len(grouped[key]))[0]
    ranking_candidates = [
        (bucket, members)
        for (level, _source), buckets in sorted(grouped.items())
        if level == ranking_level
        for bucket, members in sorted(buckets.items())
    ][:2]
    ranking_payload = {
        "groups": {
            bucket: _ranking_bucket_payload(bucket, members, lookup, sample_cards)
            for bucket, members in ranking_candidates
        }
    }
    rendered["semantic_ranking"] = Environment(
        undefined=StrictUndefined, autoescape=False
    ).from_string(RANKING_TEMPLATE.read_text()).render(
        compact_payload_json=core._canonical_json(ranking_payload)
    ).strip()
    if not core.INCLUDE_DOWNSTREAM_PROMPT_REVIEW:
        rendered.pop("semantic_ranking", None)
    maximum_context, models = core._endpoint_contract()
    prompt_rows = []
    for name, prompt in sorted(rendered.items()):
        path = output / f"{name}.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        reserve = core.HIGH_MAX_TOKENS if name.startswith("merge_") else core.LOW_MAX_TOKENS
        input_tokens = core._token_count(prompt)
        prompt_rows.append(
            {
                "name": name,
                "path": path.name,
                "sha256": core._file_sha256(path),
                "input_tokens": input_tokens,
                "completion_token_reserve": reserve,
                "fits_model_context": input_tokens + reserve <= maximum_context,
            }
        )
    manifest = {
        "version": f"{VERSION}.prompt_review.v1",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "model": core.MODEL,
        "base_url": core.BASE_URL,
        "endpoint_models": models,
        "max_model_len": maximum_context,
        "source_record_map": {"path": str(RECORD_MAP), "sha256": core._file_sha256(RECORD_MAP)},
        "v10_records": {"path": str(V10_RECORDS), "sha256": core._file_sha256(V10_RECORDS)},
        "counts": {
            "records_level_scoped": core.EXPECTED_RECORD_COUNT,
            "level_source_pair_bucket_atoms": len(atoms),
            "initial_level_scoped_buckets": sum(len(buckets) for buckets in grouped.values()),
        },
        "settings": {
            "max_source_rounds": core.MAX_SOURCE_ROUNDS,
            "merge_batch_size": core.MERGE_BATCH_SIZE,
            "merge_rounds_per_operation": 2,
            "sample_limit": core.SAMPLE_LIMIT,
            "parallelism": 128,
            "selector_profile_limit": core.SELECTOR_PROFILE_LIMIT,
            "keep_is_terminal": True,
            "technical_noop_policy": (
                "defer" if core.DEFER_TECHNICAL_BRANCHES else "keep"
            ),
            "batch_seed_version": core.BATCH_SEED_VERSION,
            "refinement_columns": {
                source: list(columns)
                for source, columns in sorted(core.REFINEMENT_COLUMNS.items())
            },
            "ignored_pair_bucket_columns": {
                source: sorted(set(PAIR_COLUMNS[source]) - set(columns))
                for source, columns in sorted(core.REFINEMENT_COLUMNS.items())
            },
            "same_parent_recombination_allowed": True,
        },
        "prompts": prompt_rows,
        "non_sendable_prompts": [row["name"] for row in prompt_rows if not row["fits_model_context"]],
        "oversized_prompt_policy": "mark_bucket_column_ineligible_without_a_completion_request",
        "prompt_template_hashes": {
            path.name: core._file_sha256(path) for path in sorted(PROMPT_ROOT.glob("*.jinja"))
        },
        **(
            {"ranking_prompt_sha256": core._file_sha256(RANKING_TEMPLATE)}
            if core.INCLUDE_DOWNSTREAM_PROMPT_REVIEW
            else {}
        ),
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-input")
    build.add_argument("--output", type=Path, default=INPUT_ROOT)
    review = subparsers.add_parser("prepare-review")
    review.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run-semantic")
    run.add_argument("--review-manifest", type=Path, required=True)
    run.add_argument("--approved-review-sha256", required=True)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--parallelism", type=int, default=128)
    run.add_argument("--seed-request-cache", type=Path)
    publish = subparsers.add_parser("publish-final-map")
    publish.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    publish.add_argument("--review", type=Path, required=True)
    cross_review = subparsers.add_parser("prepare-cross-source-prompts")
    cross_review.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cross_review.add_argument("--review-output", type=Path, required=True)
    cross_run = subparsers.add_parser("run-cross-source")
    cross_run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cross_run.add_argument("--review-manifest", type=Path, required=True)
    cross_run.add_argument("--approved-review-sha256", required=True)
    cross_run.add_argument("--parallelism", type=int, default=64)
    audit = subparsers.add_parser("audit-semantic-size")
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    approve = subparsers.add_parser("approve-semantic-size")
    approve.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    approve.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    configure_core()
    if args.command == "build-input":
        result = build_input(args.output)
    elif args.command == "prepare-review":
        result = prepare_review(args.output)
    elif args.command == "run-semantic":
        result = core.run_semantic(
            args.output,
            review_manifest_path=args.review_manifest,
            approved_review_sha256=args.approved_review_sha256,
            parallelism=args.parallelism,
            seed_request_cache=args.seed_request_cache,
        )
    elif args.command == "audit-semantic-size":
        result = core.audit_semantic_bucket_sizes(args.output)
    elif args.command == "prepare-cross-source-prompts":
        result = core.prepare_cross_source_prompt_review(
            args.output, review_output=args.review_output
        )
    elif args.command == "run-cross-source":
        result = core.run_cross_source_merges(
            args.output,
            review_manifest_path=args.review_manifest,
            approved_review_sha256=args.approved_review_sha256,
            parallelism=args.parallelism,
        )
    elif args.command == "approve-semantic-size":
        result = core.approve_semantic_bucket_sizes(
            args.output, review_path=args.review
        )
    else:
        result = core.publish_final_map(args.output, review_path=args.review)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
