"""Globally reconcile Direct endpoint local proposals without another GPT pass.

The local ``gpt-5.4-mini`` results are grouped by provisional endpoint label and
packaged for disjoint subagent review.  Primary, checker, and adjudicator JSONL
records are then validated and replayed into one unpublished runtime-shaped
proposal.  This module deliberately has no publication command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import clean_text
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.data_processing.build_direct_endpoint_mapping import (
    DEFAULT_ROOT,
    MAPPING_VERSION as LOCAL_MAPPING_VERSION,
    SOURCE_PATH,
)
from data.processing.evidence_library.shared.v2.clustered_auxiliary_mapping import distinct_values
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    endpoint_inventory_hash,
    normalize_endpoint_name,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_endpoint_normalization import (
    DIRECT_ENDPOINT_MAPPING_VERSION,
    EXPECTED_DIRECT_INVENTORY_COUNT,
    EXPECTED_DIRECT_INVENTORY_SHA256,
    EXPECTED_DIRECT_SOURCE_SHA256,
)


RECONCILIATION_VERSION = "bbb_martins_direct_endpoint_subagent_reconciliation.v1"
ALLOWED_ACTIONS = frozenset({"retain", "rename", "split"})
CATALOG_ACTIONS = frozenset({"retain", "merge"})
CHECK_VERDICTS = frozenset({"agree", "disagree"})
ADJUDICATION_VERDICTS = frozenset({"accept", "reject"})
_CANONICAL_LABEL = r"[a-z0-9]+(?:_[a-z0-9]+)*"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            output.append(row)
    return output


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"review row lacks {field}")
    return value.strip()


def _canonical(value: Any) -> str:
    text = _required_string({"value": value}, "value")
    if re.fullmatch(_CANONICAL_LABEL, text) is None:
        raise ValueError(f"non-canonical endpoint label: {text!r}")
    return text


def _validate_assignment_source(
    assignments: pd.DataFrame, *, enforce_source_pins: bool
) -> None:
    required = {
        "source_id",
        "input_column",
        "output_field",
        "cluster_id",
        "raw_value",
        "provisional_endpoint",
    }
    if missing := required - set(assignments.columns):
        raise ValueError(f"cluster assignments lack {sorted(missing)}")
    expected_constants = {
        "source_id": {"direct_bbb"},
        "input_column": {"quant_metric"},
        "output_field": {"canonical_endpoint"},
    }
    for column, expected in expected_constants.items():
        actual = {str(value) for value in assignments[column].dropna().unique()}
        if actual != expected:
            raise ValueError(f"unexpected assignment {column}: {sorted(actual)}")
    if assignments["raw_value"].isna().any() or assignments["raw_value"].duplicated().any():
        raise ValueError("Direct raw endpoint assignments must be non-null and unique")
    for value in assignments["provisional_endpoint"]:
        _canonical(value)
    if not enforce_source_pins:
        return
    if _sha256(SOURCE_PATH) != EXPECTED_DIRECT_SOURCE_SHA256:
        raise ValueError("Direct source SHA-256 differs from the frozen source")
    source = pd.read_parquet(SOURCE_PATH, columns=["quant_metric"])
    expected_raw = set(distinct_values(source["quant_metric"].tolist()))
    if set(assignments["raw_value"].astype(str)) != expected_raw:
        raise ValueError("cluster assignments do not exactly cover the frozen Direct inventory")
    normalized = sorted(
        {normalize_endpoint_name(value) or "" for value in source["quant_metric"]}
    )
    if (
        len(normalized) != EXPECTED_DIRECT_INVENTORY_COUNT
        or endpoint_inventory_hash(normalized) != EXPECTED_DIRECT_INVENTORY_SHA256
    ):
        raise ValueError("Direct normalized inventory differs from its frozen pin")


def build_review_packets(
    *,
    assignments_path: str | Path,
    output_dir: str | Path,
    max_packet_bytes: int = 750_000,
    enforce_source_pins: bool = True,
) -> dict[str, Any]:
    if max_packet_bytes < 10_000:
        raise ValueError("max_packet_bytes must be at least 10,000")
    assignments_path = Path(assignments_path)
    assignments = pd.read_parquet(assignments_path)
    _validate_assignment_source(assignments, enforce_source_pins=enforce_source_pins)
    labels: list[dict[str, Any]] = []
    for provisional, group in assignments.groupby(
        "provisional_endpoint", sort=True, dropna=False
    ):
        if not isinstance(provisional, str) or not provisional:
            raise ValueError("local endpoint proposal contains a null label")
        labels.append(
            {
                "provisional_endpoint": provisional,
                "assignment_count": int(len(group)),
                "cluster_ids": sorted({str(value) for value in group["cluster_id"]}),
                "raw_values": sorted({str(value) for value in group["raw_value"]}),
            }
        )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    packets: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for label in labels:
        candidate = current + [label]
        payload = {
            "packet_version": RECONCILIATION_VERSION,
            "labels": candidate,
        }
        # Reserve room for packet_id and the review instructions added below.
        if current and len(_json_bytes(payload)) + 4_096 > max_packet_bytes:
            packets.append(current)
            current = []
        current.append(label)
    if current:
        packets.append(current)
    packet_rows: list[dict[str, Any]] = []
    owned_labels: set[str] = set()
    owned_assignments = 0
    for index, packet_labels in enumerate(packets, start=1):
        packet_id = f"direct_endpoint_global_packet_{index:04d}"
        path = target / f"{packet_id}.json"
        payload = {
            "packet_version": RECONCILIATION_VERSION,
            "packet_id": packet_id,
            "instructions": {
                "goal": "Map lexical or symbolic aliases to one scientifically comparable endpoint label.",
                "preserve": [
                    "matrix",
                    "total_vs_unbound",
                    "concentration_vs_ratio",
                    "kp_vs_kp_uu",
                    "auc_vs_point_measurement",
                    "transport_direction",
                ],
                "actions": ["retain", "rename", "split"],
                "global_gpt_pass": False,
            },
            "labels": packet_labels,
        }
        _write_json(path, payload)
        size = path.stat().st_size
        if size > max_packet_bytes and len(packet_labels) != 1:
            raise ValueError(f"review packet exceeds byte bound: {path}")
        names = {str(row["provisional_endpoint"]) for row in packet_labels}
        if owned_labels & names:
            raise AssertionError("provisional label appears in multiple review packets")
        owned_labels.update(names)
        assignments_count = sum(int(row["assignment_count"]) for row in packet_labels)
        owned_assignments += assignments_count
        packet_rows.append(
            {
                "packet_id": packet_id,
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": size,
                "labels": len(packet_labels),
                "assignments": assignments_count,
            }
        )
    if owned_labels != {str(row["provisional_endpoint"]) for row in labels}:
        raise AssertionError("review packet label coverage mismatch")
    if owned_assignments != len(assignments):
        raise AssertionError("review packet assignment coverage mismatch")
    manifest = {
        "manifest_version": RECONCILIATION_VERSION,
        "input": {"path": str(assignments_path), "sha256": _sha256(assignments_path)},
        "max_packet_bytes": max_packet_bytes,
        "packets": packet_rows,
        "coverage": {
            "local_clusters": int(assignments["cluster_id"].nunique()),
            "provisional_labels": len(labels),
            "raw_assignments": len(assignments),
            "packets": len(packet_rows),
        },
        "validations": {
            "every_local_cluster_represented": True,
            "every_provisional_label_owned_once": True,
            "every_raw_assignment_represented": True,
            "no_second_gpt_pass": True,
            "source_and_inventory_pins_verified": enforce_source_pins,
        },
    }
    _write_json(target / "manifest.json", manifest)
    return manifest


def _validate_primary(
    primary: Sequence[Mapping[str, Any]], assignments: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    raw_by_label = {
        str(label): {str(value) for value in group["raw_value"]}
        for label, group in assignments.groupby("provisional_endpoint", sort=True)
    }
    decisions: dict[str, dict[str, Any]] = {}
    labels_seen: set[str] = set()
    for row in primary:
        decision_id = _required_string(row, "decision_id")
        if decision_id in decisions:
            raise ValueError(f"duplicate primary decision: {decision_id}")
        reviewer_id = _required_string(row, "reviewer_id")
        provisional = _required_string(row, "provisional_endpoint")
        if provisional not in raw_by_label or provisional in labels_seen:
            raise ValueError(f"unknown or repeated provisional endpoint: {provisional}")
        action = _required_string(row, "action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"invalid primary action: {action}")
        rationale = _required_string(row, "rationale")
        decision = {
            "decision_id": decision_id,
            "reviewer_id": reviewer_id,
            "provisional_endpoint": provisional,
            "action": action,
            "rationale": rationale,
        }
        if action == "split":
            mappings = row.get("raw_value_assignments")
            if not isinstance(mappings, list) or not mappings:
                raise ValueError(f"split {decision_id} lacks raw assignments")
            resolved: dict[str, str] = {}
            for mapping in mappings:
                if not isinstance(mapping, Mapping):
                    raise ValueError(f"split {decision_id} has a non-object assignment")
                raw = _required_string(mapping, "raw_value")
                if raw in resolved:
                    raise ValueError(f"split {decision_id} repeats raw value {raw!r}")
                resolved[raw] = _canonical(mapping.get("final_endpoint"))
            if set(resolved) != raw_by_label[provisional]:
                raise ValueError(f"split {decision_id} raw coverage mismatch")
            decision["raw_value_assignments"] = resolved
        else:
            final = _canonical(row.get("final_endpoint"))
            if action == "retain" and final != provisional:
                raise ValueError(f"retain {decision_id} changes its label")
            if action == "rename" and final == provisional:
                raise ValueError(f"rename {decision_id} is an identity")
            decision["final_endpoint"] = final
        decisions[decision_id] = decision
        labels_seen.add(provisional)
    if labels_seen != set(raw_by_label):
        raise ValueError(
            "primary provisional-label coverage mismatch: "
            f"missing={len(set(raw_by_label) - labels_seen)}"
        )
    return decisions


def _validate_change_reviews(
    *,
    primary: Mapping[str, Mapping[str, Any]],
    checker_path: str | Path,
    adjudication_path: str | Path,
    change_actions: set[str],
    primary_identity_field: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    change_ids = {
        decision_id
        for decision_id, decision in primary.items()
        if str(decision["action"]) in change_actions
    }
    checks: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(Path(checker_path)):
        decision_id = _required_string(row, "decision_id")
        if decision_id not in change_ids or decision_id in checks:
            raise ValueError(f"checker references unknown/repeated change: {decision_id}")
        verdict = _required_string(row, "verdict")
        if verdict not in CHECK_VERDICTS:
            raise ValueError(f"invalid checker verdict: {verdict}")
        checks[decision_id] = {
            "checker_id": _required_string(row, "checker_id"),
            "verdict": verdict,
            "rationale": _required_string(row, "rationale"),
        }
    adjudications: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(Path(adjudication_path)):
        decision_id = _required_string(row, "decision_id")
        if decision_id not in change_ids or decision_id in adjudications:
            raise ValueError(f"adjudicator references unknown/repeated change: {decision_id}")
        verdict = _required_string(row, "verdict")
        if verdict not in ADJUDICATION_VERDICTS:
            raise ValueError(f"invalid adjudication verdict: {verdict}")
        adjudications[decision_id] = {
            "adjudicator_id": _required_string(row, "adjudicator_id"),
            "verdict": verdict,
            "rationale": _required_string(row, "rationale"),
        }
    if set(checks) != change_ids or set(adjudications) != change_ids:
        raise ValueError("every changed decision requires checker and adjudicator rows")
    accepted: dict[str, dict[str, Any]] = {}
    for decision_id in sorted(change_ids):
        decision = primary[decision_id]
        identities = {
            str(decision[primary_identity_field]),
            checks[decision_id]["checker_id"],
            adjudications[decision_id]["adjudicator_id"],
        }
        if len(identities) != 3:
            raise ValueError(f"review roles are not independent for {decision_id}")
        if adjudications[decision_id]["verdict"] == "accept":
            accepted[decision_id] = dict(decision)
    return checks, adjudications, accepted


def _candidate_mapping(
    assignments: pd.DataFrame, accepted: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    decision_by_label = {
        str(decision["provisional_endpoint"]): decision
        for decision in accepted.values()
    }
    output: dict[str, str] = {}
    for row in assignments.itertuples(index=False):
        raw = str(row.raw_value)
        provisional = str(row.provisional_endpoint)
        decision = decision_by_label.get(provisional)
        if decision is None:
            final = provisional
        elif decision["action"] == "split":
            final = str(decision["raw_value_assignments"][raw])
        else:
            final = str(decision["final_endpoint"])
        if raw in output and output[raw] != final:
            raise ValueError(f"raw endpoint receives conflicting decisions: {raw!r}")
        output[raw] = final
    return output


def _validated_local_mapping(
    local_mapping_path: Path, assignments: pd.DataFrame
) -> dict[str, Any]:
    payload = json.loads(local_mapping_path.read_text(encoding="utf-8"))
    if payload.get("mapping_version") != LOCAL_MAPPING_VERSION:
        raise ValueError("unexpected local Direct mapping version")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"direct_bbb"}:
        raise ValueError("local mapping must contain only direct_bbb")
    section = sources["direct_bbb"].get("canonical_endpoint")
    if not isinstance(section, Mapping) or section.get("source_columns") != ["quant_metric"]:
        raise ValueError("local mapping has the wrong Direct source-column contract")
    serialized = section.get("mapping")
    if not isinstance(serialized, Mapping):
        raise ValueError("local mapping lacks its serialized endpoint mapping")
    decoded: dict[str, str] = {}
    saw_null = False
    for key, label in serialized.items():
        raw_key = json.loads(str(key))
        if not isinstance(raw_key, list) or len(raw_key) != 1:
            raise ValueError("invalid serialized local endpoint key")
        if raw_key[0] is None:
            saw_null = label is None
            continue
        decoded[str(raw_key[0])] = _canonical(label)
    expected = {
        str(row.raw_value): str(row.provisional_endpoint)
        for row in assignments.itertuples(index=False)
    }
    if not saw_null or decoded != expected:
        raise ValueError("local mapping does not exactly match cluster assignments")
    return payload


def _validate_generation_provenance(
    *,
    local_mapping_path: Path,
    assignments_path: Path,
    generation_audit_path: Path,
    provisional_manifest_path: Path,
    review_packet_manifest_path: Path,
    catalog_manifest_path: Path,
    primary_path: Path,
    checker_path: Path,
    adjudication_path: Path,
) -> dict[str, str]:
    generation = json.loads(generation_audit_path.read_text(encoding="utf-8"))
    if (
        generation.get("mapping_version") != LOCAL_MAPPING_VERSION
        or generation.get("mapping_sha256") != _sha256(local_mapping_path)
        or generation.get("model") != "gpt-5.4-mini"
        or generation.get("reasoning_effort") != "medium"
        or generation.get("max_cluster_size") != 500
        or generation.get("global_model_reconciliation_enabled") is not False
    ):
        raise ValueError("local GPT generation audit does not match the frozen contract")
    provisional = json.loads(provisional_manifest_path.read_text(encoding="utf-8"))
    provisional_generation = provisional.get("generation") or {}
    provisional_assignments = (provisional.get("artifacts") or {}).get("cluster_assignments") or {}
    source = provisional.get("source") or {}
    if (
        provisional.get("human_approved") is not False
        or provisional_generation.get("mapping_sha256") != _sha256(local_mapping_path)
        or provisional_assignments.get("sha256") != _sha256(assignments_path)
        or source.get("sha256") != EXPECTED_DIRECT_SOURCE_SHA256
        or source.get("normalized_inventory_count_including_missing") != EXPECTED_DIRECT_INVENTORY_COUNT
        or source.get("normalized_inventory_sha256") != EXPECTED_DIRECT_INVENTORY_SHA256
    ):
        raise ValueError("provisional manifest is not bound to the frozen local generation")
    packets = json.loads(review_packet_manifest_path.read_text(encoding="utf-8"))
    if (packets.get("input") or {}).get("sha256") != _sha256(assignments_path):
        raise ValueError("raw-level review packet manifest has the wrong assignment input")
    catalog = json.loads(catalog_manifest_path.read_text(encoding="utf-8"))
    inputs = catalog.get("inputs") or {}
    if (
        inputs.get("assignments_sha256") != _sha256(assignments_path)
        or inputs.get("primary_review_sha256") != _sha256(primary_path)
        or inputs.get("checker_review_sha256") != _sha256(checker_path)
        or inputs.get("adjudication_review_sha256") != _sha256(adjudication_path)
    ):
        raise ValueError("global catalog manifest is not bound to the accepted local review")
    catalog_artifacts = catalog.get("artifacts") or {}
    global_catalog = catalog_artifacts.get("global_catalog") or {}
    global_catalog_path = catalog_manifest_path.parent / str(global_catalog.get("path") or "")
    if global_catalog.get("sha256") != _sha256(global_catalog_path):
        raise ValueError("global catalog artifact hash mismatch")
    for packet in catalog_artifacts.get("ownership_packets") or []:
        packet_path = catalog_manifest_path.parent / str(packet.get("path") or "")
        if packet.get("sha256") != _sha256(packet_path):
            raise ValueError(f"global catalog ownership packet hash mismatch: {packet_path}")
    output = {
        "generation_audit_sha256": _sha256(generation_audit_path),
        "provisional_manifest_sha256": _sha256(provisional_manifest_path),
        "review_packet_manifest_sha256": _sha256(review_packet_manifest_path),
        "catalog_manifest_sha256": _sha256(catalog_manifest_path),
    }
    targeted_manifest_path = catalog_manifest_path.parent / "targeted_manifest.json"
    if targeted_manifest_path.exists():
        targeted = json.loads(targeted_manifest_path.read_text(encoding="utf-8"))
        if targeted.get("global_catalog_sha256") != _sha256(global_catalog_path):
            raise ValueError("targeted catalog manifest has the wrong global catalog")
        for packet in targeted.get("packets") or []:
            packet_path = targeted_manifest_path.parent / str(packet.get("path") or "")
            if packet.get("sha256") != _sha256(packet_path):
                raise ValueError(f"targeted catalog packet hash mismatch: {packet_path}")
        output["targeted_catalog_manifest_sha256"] = _sha256(targeted_manifest_path)
    return output


def _review_history_provenance(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    manifests = []
    review_history = root / "reviews/history"
    artifact_history = root / "history"
    if review_history.exists():
        manifests.extend(review_history.glob("**/manifest.json"))
    if artifact_history.exists():
        manifests.extend(artifact_history.glob("*_manifest.json"))
    for manifest_path in sorted(set(manifests)):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"review history manifest lacks artifacts: {manifest_path}")
        for name, metadata in artifacts.items():
            artifact_path = manifest_path.parent / str(name)
            if not isinstance(metadata, Mapping) or metadata.get("sha256") != _sha256(artifact_path):
                raise ValueError(f"review history artifact hash mismatch: {artifact_path}")
        output.append(
            {
                "path": str(manifest_path.relative_to(root)),
                "sha256": _sha256(manifest_path),
                "round": payload.get("round"),
                "status": payload.get("status"),
            }
        )
    return output


def _accepted_local_reviews(
    *,
    assignments: pd.DataFrame,
    primary_path: str | Path,
    checker_path: str | Path,
    adjudication_path: str | Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    primary = _validate_primary(_read_jsonl(Path(primary_path)), assignments)
    checks, adjudications, accepted = _validate_change_reviews(
        primary=primary,
        checker_path=checker_path,
        adjudication_path=adjudication_path,
        change_actions={"rename", "split"},
        primary_identity_field="reviewer_id",
    )
    return primary, checks, adjudications, accepted


def build_global_catalog(
    *,
    assignments_path: str | Path,
    primary_path: str | Path,
    checker_path: str | Path,
    adjudication_path: str | Path,
    output_dir: str | Path,
    owned_labels_per_packet: int = 500,
    enforce_source_pins: bool = True,
) -> dict[str, Any]:
    """Build a global candidate namespace after accepted raw-level reviews.

    Ownership packets are disjoint, but every packet points reviewers to the
    same full catalog.  A subsequent catalog review can therefore merge
    synonyms created in different local GPT clusters or primary packets.
    """
    if owned_labels_per_packet < 1 or owned_labels_per_packet > 500:
        raise ValueError("owned_labels_per_packet must be between 1 and 500")
    assignments_path = Path(assignments_path)
    assignments = pd.read_parquet(assignments_path)
    _validate_assignment_source(assignments, enforce_source_pins=enforce_source_pins)
    primary, checks, adjudications, accepted = _accepted_local_reviews(
        assignments=assignments,
        primary_path=primary_path,
        checker_path=checker_path,
        adjudication_path=adjudication_path,
    )
    candidate_by_raw = _candidate_mapping(assignments, accepted)
    provisional_by_raw = {
        str(row.raw_value): str(row.provisional_endpoint)
        for row in assignments.itertuples(index=False)
    }
    grouped: dict[str, list[str]] = {}
    for raw, candidate in candidate_by_raw.items():
        grouped.setdefault(candidate, []).append(raw)
    labels = []
    for candidate, raws in sorted(grouped.items()):
        labels.append(
            {
                "candidate_endpoint": candidate,
                "assignment_count": len(raws),
                "provisional_origins": sorted({provisional_by_raw[raw] for raw in raws}),
                "raw_examples": sorted(raws)[:25],
            }
        )
    cleaned_groups: dict[str, list[str]] = {}
    for raw in candidate_by_raw:
        cleaned = clean_text(raw)
        if cleaned:
            cleaned_groups.setdefault(cleaned, []).append(raw)
    cleaned_alias_candidates = [
        {
            "cleaned_endpoint": cleaned,
            "raw_aliases": sorted(raws),
            "candidate_endpoints": sorted({candidate_by_raw[raw] for raw in raws}),
            "requires_catalog_convergence": len({candidate_by_raw[raw] for raw in raws}) > 1,
        }
        for cleaned, raws in sorted(cleaned_groups.items())
        if len(raws) > 1
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    catalog_path = output / "global_candidate_catalog.json"
    _write_json(
        catalog_path,
        {
            "catalog_version": RECONCILIATION_VERSION,
            "instructions": {
                "goal": "Merge scientifically synonymous candidate endpoint labels into one shared BBB endpoint namespace.",
                "preserve": [
                    "matrix",
                    "total_vs_unbound",
                    "concentration_vs_ratio",
                    "kp_vs_kp_uu",
                    "auc_vs_point_measurement",
                    "transport_direction",
                ],
                "actions": ["retain", "merge"],
                "global_gpt_pass": False,
            },
            "labels": labels,
            "cleaned_raw_alias_groups": cleaned_alias_candidates,
        },
    )
    packet_rows = []
    for offset in range(0, len(labels), owned_labels_per_packet):
        index = offset // owned_labels_per_packet + 1
        packet_id = f"direct_endpoint_catalog_packet_{index:04d}"
        owned = [row["candidate_endpoint"] for row in labels[offset : offset + owned_labels_per_packet]]
        path = output / f"{packet_id}.json"
        _write_json(
            path,
            {
                "packet_version": RECONCILIATION_VERSION,
                "packet_id": packet_id,
                "global_catalog_path": catalog_path.name,
                "global_catalog_sha256": _sha256(catalog_path),
                "owned_candidate_endpoints": owned,
                "require_full_catalog_comparison": True,
            },
        )
        packet_rows.append(
            {
                "packet_id": packet_id,
                "path": path.name,
                "sha256": _sha256(path),
                "owned_labels": len(owned),
            }
        )
    manifest = {
        "manifest_version": RECONCILIATION_VERSION,
        "inputs": {
            "assignments_sha256": _sha256(assignments_path),
            "primary_review_sha256": _sha256(Path(primary_path)),
            "checker_review_sha256": _sha256(Path(checker_path)),
            "adjudication_review_sha256": _sha256(Path(adjudication_path)),
        },
        "counts": {
            "raw_assignments": len(candidate_by_raw),
            "provisional_labels": int(assignments["provisional_endpoint"].nunique()),
            "candidate_labels": len(labels),
            "accepted_local_changes": len(accepted),
            "catalog_packets": len(packet_rows),
            "cleaned_raw_alias_groups": len(cleaned_alias_candidates),
            "cleaned_alias_groups_requiring_convergence": sum(
                row["requires_catalog_convergence"] for row in cleaned_alias_candidates
            ),
            "extra_raw_aliases": sum(
                len(row["raw_aliases"]) - 1 for row in cleaned_alias_candidates
            ),
        },
        "artifacts": {
            "global_catalog": {"path": catalog_path.name, "sha256": _sha256(catalog_path)},
            "ownership_packets": packet_rows,
        },
        "validations": {
            "every_candidate_label_owned_once": sum(row["owned_labels"] for row in packet_rows) == len(labels),
            "every_reviewer_can_access_full_catalog": True,
            "cross_local_cluster_namespace_review_enabled": True,
            "cleaned_alias_convergence_exposed_to_reviewers": True,
            "second_gpt_pass": False,
            "source_and_inventory_pins_verified": enforce_source_pins,
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _validate_catalog_primary(
    rows: Sequence[Mapping[str, Any]], candidate_labels: set[str]
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    labels_seen: set[str] = set()
    for row in rows:
        decision_id = _required_string(row, "decision_id")
        if decision_id in decisions:
            raise ValueError(f"duplicate catalog decision: {decision_id}")
        candidate = _required_string(row, "candidate_endpoint")
        if candidate not in candidate_labels or candidate in labels_seen:
            raise ValueError(f"unknown or repeated catalog endpoint: {candidate}")
        action = _required_string(row, "action")
        if action not in CATALOG_ACTIONS:
            raise ValueError(f"invalid catalog action: {action}")
        final = _canonical(row.get("final_endpoint"))
        if action == "retain" and final != candidate:
            raise ValueError(f"catalog retain changes {candidate}")
        if action == "merge" and final == candidate:
            raise ValueError(f"catalog merge is an identity for {candidate}")
        decisions[decision_id] = {
            "decision_id": decision_id,
            "reviewer_id": _required_string(row, "reviewer_id"),
            "candidate_endpoint": candidate,
            "action": action,
            "final_endpoint": final,
            "rationale": _required_string(row, "rationale"),
        }
        labels_seen.add(candidate)
    if labels_seen != candidate_labels:
        raise ValueError(
            "catalog label coverage mismatch: "
            f"missing={len(candidate_labels - labels_seen)}"
        )
    by_candidate = {
        str(decision["candidate_endpoint"]): decision for decision in decisions.values()
    }
    for decision in decisions.values():
        if decision["action"] != "merge":
            continue
        target = str(decision["final_endpoint"])
        if target not in candidate_labels:
            raise ValueError(f"catalog merge target is not an existing candidate: {target}")
        if by_candidate[target]["action"] != "retain":
            raise ValueError(
                f"catalog merge target must be a retained representative: {target}"
            )
    return decisions


def _near_duplicate_pairs(labels: Sequence[str]) -> list[dict[str, Any]]:
    """Find cheap token-order variants without erasing directional semantics."""
    output: list[dict[str, Any]] = []
    groups: dict[tuple[str, ...], list[str]] = {}
    relation_entities = {
        "a",
        "b",
        "ab",
        "ba",
        "abluminal",
        "apical",
        "basal",
        "basolateral",
        "blood",
        "brain",
        "csf",
        "ecf",
        "isf",
        "luminal",
        "plasma",
        "serum",
        "tissue",
        "unbound",
    }
    for label in sorted(set(labels)):
        tokens = label.split("_")
        base = tuple(
            sorted(
                token.removesuffix("s")
                for token in tokens
                if token not in {"to", "of", "the"}
            )
        )
        relation_sequence = tuple(
            token.removesuffix("s")
            for token in tokens
            if token in relation_entities or token in {"to", "over"}
        )
        has_explicit_relation = any(token in {"to", "over"} for token in tokens)
        matrix_sequence = tuple(
            token.removesuffix("s") for token in tokens if token in relation_entities
        )
        relation_guard = (
            relation_sequence
            if has_explicit_relation
            else matrix_sequence
            if len(matrix_sequence) > 1 and "and" not in tokens
            else ()
        )
        signature = (*base, "__relation__", *relation_guard)
        groups.setdefault(signature, []).append(label)
    for signature, variants in sorted(groups.items()):
        if len(variants) < 2:
            continue
        for index, left in enumerate(variants):
            for right in variants[index + 1 :]:
                output.append(
                    {
                        "left": left,
                        "right": right,
                        "token_signature": list(signature),
                    }
                )
    return output


def _cleaned_final_mapping(
    final_by_raw: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    cleaned_mapping: dict[str, str] = {}
    cleaned_aliases: dict[str, list[str]] = {}
    for raw, final in sorted(final_by_raw.items()):
        cleaned = clean_text(raw)
        if not cleaned:
            raise ValueError("non-null local assignment cleaned to null")
        if cleaned in cleaned_mapping and cleaned_mapping[cleaned] != final:
            raise ValueError(
                f"raw aliases collapsing to {cleaned!r} retain conflicting global labels"
            )
        cleaned_mapping[cleaned] = final
        cleaned_aliases.setdefault(cleaned, []).append(raw)
    return cleaned_mapping, cleaned_aliases


def consolidate(
    *,
    assignments_path: str | Path,
    local_mapping_path: str | Path,
    primary_path: str | Path,
    checker_path: str | Path,
    adjudication_path: str | Path,
    catalog_primary_path: str | Path,
    catalog_checker_path: str | Path,
    catalog_adjudication_path: str | Path,
    output_dir: str | Path,
    enforce_source_pins: bool = True,
    generation_audit_path: str | Path | None = None,
    provisional_manifest_path: str | Path | None = None,
    review_packet_manifest_path: str | Path | None = None,
    catalog_manifest_path: str | Path | None = None,
    verify_generation_provenance: bool = True,
) -> dict[str, Any]:
    assignments_path = Path(assignments_path)
    local_mapping_path = Path(local_mapping_path)
    assignments = pd.read_parquet(assignments_path)
    _validate_assignment_source(assignments, enforce_source_pins=enforce_source_pins)
    local_mapping = _validated_local_mapping(local_mapping_path, assignments)
    root = local_mapping_path.parent.parent
    generation_provenance: dict[str, str] = {}
    if verify_generation_provenance:
        generation_provenance = _validate_generation_provenance(
            local_mapping_path=local_mapping_path,
            assignments_path=assignments_path,
            generation_audit_path=Path(generation_audit_path or local_mapping_path.with_suffix(local_mapping_path.suffix + ".generation.json")),
            provisional_manifest_path=Path(provisional_manifest_path or root / "provenance/provisional_manifest.json"),
            review_packet_manifest_path=Path(review_packet_manifest_path or root / "global_review_packets/manifest.json"),
            catalog_manifest_path=Path(catalog_manifest_path or root / "global_catalog_review_packets/manifest.json"),
            primary_path=Path(primary_path),
            checker_path=Path(checker_path),
            adjudication_path=Path(adjudication_path),
        )
    review_history = _review_history_provenance(root)
    primary, checks, adjudications, accepted = _accepted_local_reviews(
        assignments=assignments,
        primary_path=primary_path,
        checker_path=checker_path,
        adjudication_path=adjudication_path,
    )
    candidate_by_raw = _candidate_mapping(assignments, accepted)
    catalog_primary = _validate_catalog_primary(
        _read_jsonl(Path(catalog_primary_path)), set(candidate_by_raw.values())
    )
    catalog_checks, catalog_adjudications, accepted_catalog = _validate_change_reviews(
        primary=catalog_primary,
        checker_path=catalog_checker_path,
        adjudication_path=catalog_adjudication_path,
        change_actions={"merge"},
        primary_identity_field="reviewer_id",
    )
    catalog_by_candidate = {
        str(decision["candidate_endpoint"]): decision
        for decision in accepted_catalog.values()
    }
    final_by_raw = {
        raw: str(catalog_by_candidate[candidate]["final_endpoint"])
        if candidate in catalog_by_candidate
        else candidate
        for raw, candidate in candidate_by_raw.items()
    }
    cleaned_mapping, cleaned_aliases = _cleaned_final_mapping(final_by_raw)
    if len(final_by_raw) != len(assignments):
        raise ValueError("final raw endpoint coverage mismatch")
    if enforce_source_pins:
        source = pd.read_parquet(SOURCE_PATH, columns=["quant_metric"])
        expected_cleaned = {
            clean_text(value)
            for value in source["quant_metric"]
            if clean_text(value)
        }
        if set(cleaned_mapping) != expected_cleaned:
            raise ValueError("proposed mapping lacks exact cleaned Direct inventory coverage")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    proposal_path = output / "proposed_endpoint_mapping.json"
    proposal = {
        "mapping_version": DIRECT_ENDPOINT_MAPPING_VERSION,
        "source": {
            "sha256": EXPECTED_DIRECT_SOURCE_SHA256,
            "inventory_count_including_missing": EXPECTED_DIRECT_INVENTORY_COUNT,
            "inventory_sha256": EXPECTED_DIRECT_INVENTORY_SHA256,
        },
        "approval": {
            "human_approved": False,
            "approved_by": None,
            "approved_at": None,
        },
        "mapping": dict(sorted(cleaned_mapping.items())),
        "provenance": {
            "reconciliation_version": RECONCILIATION_VERSION,
            "local_mapping_version": local_mapping.get("mapping_version"),
            "local_mapping_sha256": _sha256(local_mapping_path),
            "cluster_assignments_sha256": _sha256(assignments_path),
            "primary_review_sha256": _sha256(Path(primary_path)),
            "checker_review_sha256": _sha256(Path(checker_path)),
            "adjudication_review_sha256": _sha256(Path(adjudication_path)),
            "catalog_primary_review_sha256": _sha256(Path(catalog_primary_path)),
            "catalog_checker_review_sha256": _sha256(Path(catalog_checker_path)),
            "catalog_adjudication_review_sha256": _sha256(Path(catalog_adjudication_path)),
            "global_gpt_pass": False,
            "review_history_manifests": review_history,
            **generation_provenance,
        },
    }
    _write_json(proposal_path, proposal)
    alias_path = output / "cleaned_raw_aliases.parquet"
    pd.DataFrame(
        [
            {"cleaned_endpoint": cleaned, "raw_aliases_json": json.dumps(raws, ensure_ascii=False)}
            for cleaned, raws in sorted(cleaned_aliases.items())
        ]
    ).to_parquet(alias_path, index=False)
    provisional_by_raw = {
        str(row.raw_value): str(row.provisional_endpoint)
        for row in assignments.itertuples(index=False)
    }
    origins_by_final: dict[str, set[str]] = {}
    for raw, final in final_by_raw.items():
        origins_by_final.setdefault(final, set()).add(provisional_by_raw[raw])
    merged_origins = [
        {"final_endpoint": final, "provisional_origins": sorted(origins)}
        for final, origins in sorted(origins_by_final.items())
        if len(origins) > 1
    ]
    merged_origins_path = output / "cross_cluster_label_merges.json"
    _write_json(merged_origins_path, merged_origins)
    near_duplicates = _near_duplicate_pairs(sorted(set(final_by_raw.values())))
    near_duplicates_path = output / "unresolved_near_duplicate_labels.json"
    _write_json(near_duplicates_path, near_duplicates)
    local_change_ids = {key for key, row in primary.items() if row["action"] != "retain"}
    catalog_change_ids = {key for key, row in catalog_primary.items() if row["action"] == "merge"}
    alias_collision_groups = sum(len(raws) > 1 for raws in cleaned_aliases.values())
    extra_raw_aliases = sum(max(0, len(raws) - 1) for raws in cleaned_aliases.values())
    manifest = {
        "manifest_version": RECONCILIATION_VERSION,
        "publication_status": "awaiting_human_approval",
        "publication_blockers": ["human_approval_not_recorded"],
        "human_approved": False,
        "counts": {
            "raw_assignments": len(final_by_raw),
            "cleaned_endpoint_keys": len(cleaned_mapping),
            "cleaned_alias_collision_groups": alias_collision_groups,
            "extra_raw_aliases": extra_raw_aliases,
            "primary_decisions": len(primary),
            "change_decisions": len(local_change_ids),
            "accepted_changes": len(accepted),
            "checker_disagreements": sum(row["verdict"] == "disagree" for row in checks.values()),
            "catalog_decisions": len(catalog_primary),
            "catalog_merge_decisions": len(catalog_change_ids),
            "accepted_catalog_merges": len(accepted_catalog),
            "catalog_checker_disagreements": sum(row["verdict"] == "disagree" for row in catalog_checks.values()),
            "final_endpoint_labels": len(set(final_by_raw.values())),
            "final_labels_spanning_provisional_labels": len(merged_origins),
            "unresolved_near_duplicate_candidates": len(near_duplicates),
        },
        "artifacts": {
            "proposal": {"path": str(proposal_path), "sha256": _sha256(proposal_path)},
            "cleaned_raw_aliases": {"path": str(alias_path), "sha256": _sha256(alias_path)},
            "cross_cluster_label_merges": {"path": str(merged_origins_path), "sha256": _sha256(merged_origins_path)},
            "unresolved_near_duplicate_labels": {"path": str(near_duplicates_path), "sha256": _sha256(near_duplicates_path)},
        },
        "validations": {
            "all_provisional_labels_reviewed": True,
            "all_changes_independently_checked": True,
            "all_changes_independently_adjudicated": True,
            "global_catalog_review_complete": True,
            "all_catalog_merges_independently_checked": True,
            "all_catalog_merges_independently_adjudicated": True,
            "all_raw_values_mapped_once": True,
            "cleaned_alias_conflicts": 0,
            "local_mapping_exactly_matches_cluster_assignments": True,
            "source_and_inventory_pins_verified": enforce_source_pins,
            "generation_provenance_verified": verify_generation_provenance,
            "review_history_verified": True,
            "global_gpt_pass_disabled": True,
            "human_approved": False,
        },
    }
    _write_json(output / "reconciliation_manifest.json", manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--assignments",
        default=str(DEFAULT_ROOT / "provenance/cluster_assignments.parquet"),
    )
    prepare.add_argument("--output-dir", default=str(DEFAULT_ROOT / "global_review_packets"))
    prepare.add_argument("--max-packet-bytes", type=int, default=750_000)
    catalog = subparsers.add_parser("prepare-catalog")
    catalog.add_argument("--assignments", default=str(DEFAULT_ROOT / "provenance/cluster_assignments.parquet"))
    catalog.add_argument("--primary", default=str(DEFAULT_ROOT / "reviews/primary_review.jsonl"))
    catalog.add_argument("--checker", default=str(DEFAULT_ROOT / "reviews/checker_review.jsonl"))
    catalog.add_argument("--adjudication", default=str(DEFAULT_ROOT / "reviews/adjudication_review.jsonl"))
    catalog.add_argument("--output-dir", default=str(DEFAULT_ROOT / "global_catalog_review_packets"))
    catalog.add_argument("--owned-labels-per-packet", type=int, default=500)
    propose = subparsers.add_parser("propose")
    propose.add_argument("--assignments", default=str(DEFAULT_ROOT / "provenance/cluster_assignments.parquet"))
    propose.add_argument("--local-mapping", default=str(DEFAULT_ROOT / "provisional/local_endpoint_mapping.json"))
    propose.add_argument("--primary", default=str(DEFAULT_ROOT / "reviews/primary_review.jsonl"))
    propose.add_argument("--checker", default=str(DEFAULT_ROOT / "reviews/checker_review.jsonl"))
    propose.add_argument("--adjudication", default=str(DEFAULT_ROOT / "reviews/adjudication_review.jsonl"))
    propose.add_argument("--catalog-primary", default=str(DEFAULT_ROOT / "reviews/catalog_primary_review.jsonl"))
    propose.add_argument("--catalog-checker", default=str(DEFAULT_ROOT / "reviews/catalog_checker_review.jsonl"))
    propose.add_argument("--catalog-adjudication", default=str(DEFAULT_ROOT / "reviews/catalog_adjudication_review.jsonl"))
    propose.add_argument("--generation-audit", default=str(DEFAULT_ROOT / "provisional/local_endpoint_mapping.json.generation.json"))
    propose.add_argument("--provisional-manifest", default=str(DEFAULT_ROOT / "provenance/provisional_manifest.json"))
    propose.add_argument("--review-packet-manifest", default=str(DEFAULT_ROOT / "global_review_packets/manifest.json"))
    propose.add_argument("--catalog-manifest", default=str(DEFAULT_ROOT / "global_catalog_review_packets/manifest.json"))
    propose.add_argument("--output-dir", default=str(DEFAULT_ROOT / "proposal"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prepare":
        result = build_review_packets(
            assignments_path=args.assignments,
            output_dir=args.output_dir,
            max_packet_bytes=args.max_packet_bytes,
        )
        print(json.dumps(result["coverage"], indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "prepare-catalog":
        result = build_global_catalog(
            assignments_path=args.assignments,
            primary_path=args.primary,
            checker_path=args.checker,
            adjudication_path=args.adjudication,
            output_dir=args.output_dir,
            owned_labels_per_packet=args.owned_labels_per_packet,
        )
        print(json.dumps(result["counts"], indent=2, sort_keys=True), flush=True)
        return 0
    result = consolidate(
        assignments_path=args.assignments,
        local_mapping_path=args.local_mapping,
        primary_path=args.primary,
        checker_path=args.checker,
        adjudication_path=args.adjudication,
        catalog_primary_path=args.catalog_primary,
        catalog_checker_path=args.catalog_checker,
        catalog_adjudication_path=args.catalog_adjudication,
        generation_audit_path=args.generation_audit,
        provisional_manifest_path=args.provisional_manifest,
        review_packet_manifest_path=args.review_packet_manifest,
        catalog_manifest_path=args.catalog_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_global_catalog", "build_review_packets", "consolidate", "main"]
