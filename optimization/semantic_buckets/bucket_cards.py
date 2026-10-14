"""Scientific cards, task targets and Jinja template rendering."""

from __future__ import annotations
from .storage import _stable_id
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import json
from typing import Any
import pandas as pd
import argparse
import gzip
from pathlib import Path
from collections.abc import Mapping
from predict.utils.json import sha256_file
from predict.utils.json import write_json_atomic


def _bucket_payload(
    bucket: str,
    atom_ids: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
    *,
    policy: Mapping,
) -> dict[str, Any]:
    atoms = [lookup[atom] for atom in atom_ids]
    columns = policy["PROMPT_DIMENSION_COLUMNS"][str(atoms[0]["source_id"])]
    dimensions = {
        column: sorted({str(atom["values"][column]) for atom in atoms})
        for column in columns
    }
    samples = [
        {
            column: sample_cards[atom][column]
            for column in columns
            if column in sample_cards[atom]
        }
        for atom in sorted(atom_ids)[: policy["SAMPLE_LIMIT"]]
    ]
    return {
        "bucket_id": bucket,
        "pair_bucket_count": len(atom_ids),
        "record_count": sum((int(atom["record_count"]) for atom in atoms)),
        "canonical_dimensions": dimensions,
        "sample_records": samples,
    }


def _value_payload(
    bucket: str,
    atom_ids: Sequence[str],
    column: str,
    lookup: Mapping[str, Mapping[str, Any]],
    *,
    context_columns: Iterable[str] = (),
    policy: Mapping,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    by_value: dict[str, list[str]] = {}
    for atom_id in atom_ids:
        value = str(lookup[atom_id]["values"][column])
        by_value.setdefault(value, []).append(atom_id)
    value_items: dict[str, list[str]] = {}
    values = []
    for index, (value, members) in enumerate(sorted(by_value.items())):
        value_id = f"v{index:06d}"
        value_items[value_id] = sorted(members)
        values.append(
            {
                "value_id": value_id,
                "value": value,
                "pair_bucket_count": len(members),
                "record_count": sum(
                    (int(lookup[item]["record_count"]) for item in members)
                ),
            }
        )
    identity = {
        context: sorted({str(lookup[atom]["values"][context]) for atom in atom_ids})
        for context in context_columns
        if context != column
    }
    return (
        {
            "bucket_id": bucket,
            "current_bucket_identity": identity,
            "column": column,
            "column_description": policy["COLUMN_DESCRIPTIONS"][column],
            "values": values,
        },
        value_items,
    )


def _column_selection_payload(
    level: str,
    source: str,
    buckets: Mapping[str, Sequence[str]],
    lookup: Mapping[str, Mapping[str, Any]],
    consumed: Iterable[str],
    *,
    policy: Mapping,
) -> dict[str, Any]:
    consumed_set = set(consumed)
    profiled = dict(sorted(buckets.items()))
    if policy["SELECTOR_PROFILE_LIMIT"] is not None:
        profiled = dict(list(profiled.items())[: policy["SELECTOR_PROFILE_LIMIT"]])
    active_buckets = []
    for bucket, atom_ids in profiled.items():
        active_buckets.append(
            {
                "bucket_id": bucket,
                "current_bucket_identity": {
                    column: sorted(
                        {str(lookup[atom]["values"][column]) for atom in atom_ids}
                    )
                    for column in sorted(consumed_set)
                },
                "pair_bucket_count": len(atom_ids),
                "record_count": sum(
                    (int(lookup[atom]["record_count"]) for atom in atom_ids)
                ),
            }
        )
    candidates = []
    for column in policy["REFINEMENT_COLUMNS"][source]:
        if column in consumed_set:
            continue
        distinct_counts = [
            len({str(lookup[atom]["values"][column]) for atom in atom_ids})
            for atom_ids in buckets.values()
        ]
        profiles = []
        for bucket, atom_ids in profiled.items():
            values = Counter((str(lookup[atom]["values"][column]) for atom in atom_ids))
            profiles.append(
                {
                    "bucket_id": bucket,
                    "distinct_values": len(values),
                    "example_values": sorted(values)
                    if policy["SELECTOR_VALUE_LIMIT"] is None
                    else sorted(values)[: policy["SELECTOR_VALUE_LIMIT"]],
                }
            )
        candidates.append(
            {
                "column": column,
                "description": policy["COLUMN_DESCRIPTIONS"][column],
                "buckets_with_multiple_values": sum(
                    (count > 1 for count in distinct_counts)
                ),
                "maximum_distinct_values_in_one_bucket": max(distinct_counts),
                "bucket_profiles": profiles,
            }
        )
    return {
        "task": policy["TASK_NAME"],
        "level": level,
        "source_id": source,
        "active_bucket_count": len(buckets),
        "profiled_bucket_count": len(profiled),
        "active_buckets": active_buckets,
        "consumed_columns": sorted(consumed_set),
        "candidate_columns": candidates,
    }


def _merge_bucket_payload(
    level: str,
    source: str,
    buckets: Mapping[str, Sequence[str]],
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
    *,
    policy: Mapping,
) -> dict[str, Any]:
    return {
        "task": policy["TASK_NAME"],
        "level": level,
        "source_id": source,
        "buckets": [
            _bucket_payload(bucket, members, lookup, sample_cards, policy=policy)
            for bucket, members in sorted(buckets.items())
        ],
    }


DIMENSION_VALUE_LIMIT = 8
TEXT_LIMIT = 240


def _sample_atoms(bucket: str, atoms: pd.DataFrame) -> list[str]:
    ordered = sorted(
        atoms.atom_id.astype(str),
        key=lambda atom: (_stable_id("weight-sample", bucket, atom), atom),
    )
    sources = dict(zip(atoms.atom_id.astype(str), atoms.source_id.astype(str)))
    selected: list[str] = []
    seen: set[str] = set()
    for new_source_only in (True, False):
        for atom in ordered:
            source = sources[atom]
            if atom in selected or (new_source_only and source in seen):
                continue
            selected.append(atom)
            seen.add(source)
            if len(selected) == 5:
                return selected
    return selected


def _component(
    source: str, rows: pd.DataFrame, *, source_labels: Mapping[str, str] | None = None
) -> dict[str, Any]:
    values = [json.loads(value) for value in rows.values_json]
    fields = sorted({field for item in values for field in item})
    dimensions = {}
    for field in fields:
        distinct = {str(item.get(field, "__unknown__")) for item in values}
        displayed = sorted(
            distinct,
            key=lambda value: (
                _stable_id("weight-dimension", source, field, value),
                value,
            ),
        )[:DIMENSION_VALUE_LIMIT]
        displayed.sort()
        if len(distinct) > DIMENSION_VALUE_LIMIT:
            displayed.append(
                f"[{len(distinct) - DIMENSION_VALUE_LIMIT} additional values omitted]"
            )
        dimensions[field] = displayed
    return {
        "source_id": source,
        "source_label": (source_labels or {}).get(source, source),
        "canonical_dimensions": dimensions,
    }


def _usable(value: Any) -> bool:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return (
        bool(text)
        and (not text.startswith("__not_present"))
        and (text != "not_applicable")
    )


def _display_value(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("examples", value)
    if isinstance(value, list):
        text = " | ".join((_display_value(item) for item in value if _usable(item)))
    else:
        text = " ".join(str(value).split())
    if text.lower() in {"__unknown__", "unknown"}:
        text = "not reported"
    return text if len(text) <= TEXT_LIMIT else text[: TEXT_LIMIT - 1].rstrip() + "…"


def _card(
    label: str,
    payload: Mapping[str, Any],
    *,
    source_labels: Mapping[str, str] | None = None,
) -> str:
    labels = source_labels or {}
    lines = [label]
    for component in payload["identity"]["source_components"]:
        lines.extend(
            [
                f"  Evidence family: {component.get('source_label') or labels.get(component.get('source_id', ''), 'not reported')}",
                "  Canonical bucket identity:",
            ]
        )
        for field, values in component["canonical_dimensions"].items():
            if shown := _display_value(values):
                lines.append(f"    - {field.replace('_', ' ')}: {shown}")
    lines.append("  Representative experimental records:")
    for index, sample in enumerate(payload["sample_records"], 1):
        lines.append(
            f"    Record {index} ({labels.get(sample['source_id'], sample['source_id'])}):"
        )
        for field, value in sample.items():
            if field != "source_id" and (shown := _display_value(value)):
                lines.append(f"      - {field.replace('_', ' ')}: {shown}")
    return "\n".join(lines)


def _anchor_card(
    index: int,
    payload: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    source_labels: Mapping[str, str] | None = None,
) -> str:
    heading = f"Anchor {index} — locked weight {score['weight']}\n  Prior rationale: {' '.join(str(score['rationale']).split())}"
    return "\n".join(
        [
            heading,
            *_card(
                "  Scientific profile", payload, source_labels=source_labels
            ).splitlines()[1:],
        ]
    )


TASKS = {
    "skin_reaction": {
        "target": "skin sensitization or allergic contact dermatitis",
        "source_labels": {
            "direct_skin_reaction": "Direct skin sensitization/contact-allergy evidence",
            "sensitization_aop": "Skin-sensitization AOP and mechanistic evidence",
            "skin_exposure": "Dermal exposure evidence",
        },
    },
    "ames": {
        "target": "Ames bacterial mutagenicity under the reported assay conditions",
        "source_labels": {
            "fixed_mutation": "Fixed-mutation evidence",
            "mutagenicity_mechanism": "Mutagenicity mechanism evidence",
            "mutagenicity_outcomes": "Mutagenicity outcome evidence",
            "premutagenic_damage": "Pre-mutagenic damage evidence",
        },
    },
    "dili": {
        "target": "drug-induced liver injury",
        "source_labels": {
            "dili_base": "DILI evidence source base",
            "dili_v1": "DILI evidence source v1",
            "dili_v2": "DILI evidence source v2",
            "dili_v3": "DILI evidence source v3",
            "dili_v4": "DILI evidence source v4",
            "dili_v5": "DILI evidence source v5",
        },
    },
    "carcinogens": {
        "target": "carcinogenicity",
        "source_labels": {
            "carcinogens_base": "CARCINOGENS evidence source base",
            "carcinogens_v1": "CARCINOGENS evidence source v1",
            "carcinogens_v2": "CARCINOGENS evidence source v2",
            "carcinogens_v3": "CARCINOGENS evidence source v3",
            "carcinogens_v4": "CARCINOGENS evidence source v4",
            "carcinogens_v5": "CARCINOGENS evidence source v5",
        },
    },
    "bbb_martins": {"target": "blood-brain barrier access", "source_labels": {}},
    "bioavailability_ma": {
        "target": "oral bioavailability under the reported condition",
        "source_labels": {},
    },
}
AMES_ENRICHED_COLUMNS = (
    "source_name",
    "evidence_basis",
    "experimental_context",
    "test_system",
    "biological_test_system",
    "metabolic_activation",
    "metabolic_activation_system",
    "dose_or_concentration",
    "endpoint_class",
    "genetic_locus_or_chromosome_target",
    "assay_method_and_endpoint",
    "result_call",
    "result_direction",
    "cytotoxicity_status",
    "mechanism_category",
    "qualifying_conditions",
)
ENRICHED_COLUMNS = {
    "ames": AMES_ENRICHED_COLUMNS,
    "dili": (
        "source_name",
        "endpoint_name",
        "canonical_endpoint_concept",
        "canonical_assay_context",
        "canonical_species_context",
        "canonical_reference_scope",
        "canonical_measurement_text",
        "canonical_unit_text",
        "assay_and_readout",
        "biological_model_context",
        "human_evidence_basis",
        "reaction_type",
        "biochemical_pattern",
        "clinical_phenotype",
        "maximum_reported_severity",
        "qualifying_conditions",
        "support_text",
    ),
    "carcinogens": (
        "source_name",
        "endpoint_name",
        "canonical_endpoint_concept",
        "canonical_assay_context",
        "canonical_species_context",
        "canonical_reference_scope",
        "canonical_measurement_text",
        "canonical_unit_text",
        "assay_method",
        "biological_test_system",
        "evidence_basis",
        "evidence_scope",
        "carcinogenic_role",
        "classification_authority",
        "classification_label",
        "effect_direction",
        "qualifying_conditions",
        "support_text",
    ),
}


def prepare_scoring_inputs(
    task: str,
    world_path: Path,
    semantic_root: Path,
    output: Path,
    records_path: Path | None = None,
) -> dict:
    """Render the canonical sampled cards from an explicit frozen bucket world."""
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "world": world_path,
        "semantic_map": semantic_root / "semantic_bucket_map.parquet",
        "atoms": semantic_root / "input_atoms.parquet",
        "samples": semantic_root / "sample_cards.json.gz",
    }
    world = pd.read_parquet(world_path, columns=["level", "semantic_bucket_id"])
    if world.empty or world.duplicated(["level", "semantic_bucket_id"]).any():
        raise ValueError(
            "scoring world must contain unique task-level bucket identities"
        )
    mapping = pd.read_parquet(sources["semantic_map"])
    if mapping.atom_id.duplicated().any():
        raise ValueError("semantic map repeats intrinsic atoms")
    wanted = set(zip(world.level, world.semantic_bucket_id))
    mapping = mapping[
        [key in wanted for key in zip(mapping.level, mapping.semantic_bucket_id)]
    ]
    if set(zip(mapping.level, mapping.semantic_bucket_id)) != wanted:
        raise ValueError("semantic map does not cover the scoring world")
    atoms = pd.read_parquet(sources["atoms"]).set_index("atom_id", drop=False)
    if atoms.index.duplicated().any() or not set(mapping.atom_id) <= set(atoms.index):
        raise ValueError("semantic map has missing or repeated atom inputs")
    samples = {}
    if records_path is None:
        with gzip.open(sources["samples"], "rt", encoding="utf-8") as handle:
            samples = json.load(handle)
    if records_path is not None:
        del sources["samples"]
        if task not in ENRICHED_COLUMNS:
            raise ValueError("this task uses construction sample cards")
        sources["enriched_records"] = records_path
        sources["record_map"] = semantic_root / "record_semantic_bucket_map.parquet"
        record_map = pd.read_parquet(sources["record_map"])
        sampled = {
            atom
            for bucket, members in mapping.groupby("semantic_bucket_id")
            for atom in _sample_atoms(str(bucket), atoms.loc[members.atom_id])
        }
        representatives = (
            record_map[record_map.atom_id.isin(sampled)]
            .sort_values(["atom_id", "canonical_record_id"])
            .drop_duplicates("atom_id")
        )
        records = pd.read_parquet(
            records_path,
            columns=["source_row_uid", *ENRICHED_COLUMNS[task]],
            filters=[("source_row_uid", "in", representatives.source_row_uid.tolist())],
        )
        records = records.drop_duplicates("source_row_uid").set_index("source_row_uid")
        samples = {
            str(row.atom_id): {
                field: records.loc[row.source_row_uid, field]
                for field in ENRICHED_COLUMNS[task]
                if _usable(records.loc[row.source_row_uid, field])
            }
            for row in representatives.itertuples(index=False)
        }
        if set(samples) != sampled:
            raise ValueError("enriched representatives do not cover sampled atoms")
    labels = TASKS[task]["source_labels"]
    payloads = {}
    for bucket, members in mapping.groupby("semantic_bucket_id", sort=True):
        if members.level.nunique() != 1:
            raise ValueError("bucket identity spans levels")
        rows = atoms.loc[members.atom_id]
        sampled = _sample_atoms(str(bucket), rows)
        if not set(sampled) <= samples.keys():
            raise ValueError("representative cards omit sampled atoms")
        payloads[str(bucket)] = {
            "identity": {
                "source_components": [
                    _component(str(source), group, source_labels=labels)
                    for source, group in rows.groupby("source_id")
                ]
            },
            "sample_records": [
                {
                    "source_id": str(atoms.loc[atom, "source_id"]),
                    **{
                        key: value
                        for key, value in samples[atom].items()
                        if key not in {"canonical_record_id", "source_id"}
                    },
                }
                for atom in sampled
            ],
        }
    pins = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sources.items()
    }
    output.mkdir(parents=True)
    world.to_parquet(output / "world.parquet", index=False)
    with gzip.open(
        output / "bucket_prompt_payloads.json.gz", "wt", encoding="utf-8"
    ) as handle:
        json.dump(payloads, handle, sort_keys=True, ensure_ascii=False)
    descriptor = {
        "tasks": {
            task: {
                "target": TASKS[task]["target"],
                "source_labels": labels,
                "world": "world.parquet",
                "payloads": "bucket_prompt_payloads.json.gz",
            }
        },
        "source_bindings": pins,
    }
    write_json_atomic(output / "inputs.json", descriptor)
    return descriptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare frozen world/card inputs for semantic utility scoring."
    )
    parser.add_argument("--task", required=True, choices=tuple(TASKS))
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enriched-records", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_scoring_inputs(
                args.task,
                args.world,
                args.semantic_root,
                args.output,
                args.enriched_records,
            ),
            indent=2,
        )
    )
