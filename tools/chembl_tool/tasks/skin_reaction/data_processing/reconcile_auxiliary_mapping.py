"""Build the human-gated Skin_Reaction auxiliary reconciliation proposal.

This entry point intentionally has no publish command.  It freezes the local
GPT mappings, prepares complete review packets, assembles independently
reviewed decisions, and verifies an unpublished runtime-compatible proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .auxiliary_reconciliation import write_provisional_artifacts
from .auxiliary_reconciliation_review import (
    audit_proposal,
    build_label_catalog,
    build_review_packets,
    write_proposal,
)


DATA_PROCESSING_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = DATA_PROCESSING_DIR / "auxiliary_reconciliation_v2"
DEFAULT_PROVENANCE_DIR = DEFAULT_WORK_ROOT / "provenance"
DEFAULT_CACHE_DIR = (
    DATA_PROCESSING_DIR / "globally_reconciled_auxiliary_value_mapping.json.cache"
)
DEFAULT_RUNTIME_MAPPING = (
    DATA_PROCESSING_DIR / "globally_reconciled_auxiliary_value_mapping.json"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                records.append(record)
    return records


def _review_assignments(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "review_eligible" not in frame:
        raise ValueError("assignment artifact lacks review_eligible")
    return frame.loc[frame["review_eligible"]].copy()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser(
        "snapshot", help="freeze exact local outputs and the aggregated baseline"
    )
    snapshot.add_argument("--output-dir", default=str(DEFAULT_PROVENANCE_DIR))
    snapshot.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    snapshot.add_argument("--overwrite", action="store_true")

    prepare = commands.add_parser(
        "prepare-review", help="write the label catalog and cluster-atomic packets"
    )
    prepare.add_argument(
        "--assignments",
        default=str(DEFAULT_PROVENANCE_DIR / "cluster_assignments.parquet"),
    )
    prepare.add_argument("--output-dir", default=str(DEFAULT_WORK_ROOT))
    prepare.add_argument("--max-packet-bytes", type=int, default=750_000)

    propose = commands.add_parser(
        "propose", help="validate independent reviews and write an unpublished mapping"
    )
    propose.add_argument(
        "--provisional-mapping",
        default=str(DEFAULT_PROVENANCE_DIR / "pre_reconciliation_mapping.json"),
    )
    propose.add_argument(
        "--assignments",
        default=str(DEFAULT_PROVENANCE_DIR / "cluster_assignments.parquet"),
    )
    propose.add_argument("--primary", nargs="+", required=True)
    propose.add_argument("--checks", nargs="+", required=True)
    propose.add_argument("--adjudications", nargs="+", required=True)
    propose.add_argument("--review-summaries", nargs="+", required=True)
    propose.add_argument(
        "--packet-manifest",
        default=str(DEFAULT_WORK_ROOT / "review_packets/manifest.json"),
    )
    propose.add_argument("--output-dir", default=str(DEFAULT_WORK_ROOT / "proposal"))

    audit = commands.add_parser(
        "audit", help="replay and hash-check the complete unpublished proposal"
    )
    audit.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    audit.add_argument("--runtime-mapping", default=str(DEFAULT_RUNTIME_MAPPING))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "snapshot":
        result = write_provisional_artifacts(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
        )
    elif args.command == "prepare-review":
        assignments = _review_assignments(args.assignments)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog = build_label_catalog(assignments)
        catalog_path = output_dir / "label_catalog.parquet"
        catalog.to_parquet(catalog_path, index=False, compression="zstd")
        packet_manifest = build_review_packets(
            assignments,
            output_dir=output_dir / "review_packets",
            max_packet_bytes=args.max_packet_bytes,
        )
        result = {
            "label_catalog": str(catalog_path),
            "labels": len(catalog),
            "review_packets": packet_manifest,
            "publication_status": "not_applicable",
        }
    elif args.command == "propose":
        result = write_proposal(
            output_dir=args.output_dir,
            provisional_mapping=_read_json(args.provisional_mapping),
            assignments=_review_assignments(args.assignments),
            primary=_read_jsonl(args.primary),
            checks=_read_jsonl(args.checks),
            adjudications=_read_jsonl(args.adjudications),
            review_summaries=[_read_json(path) for path in args.review_summaries],
            packet_manifest=_read_json(args.packet_manifest),
        )
    else:
        result = audit_proposal(
            work_root=args.work_root,
            runtime_mapping_path=args.runtime_mapping,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
