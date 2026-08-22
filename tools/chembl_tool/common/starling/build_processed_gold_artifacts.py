"""Build record-complete numeric views of the frozen Starling gold lineages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping

from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.starling.benchmark_dataset import parse_numeric_interval
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    SOURCE_DATASET as BBB_SOURCE_DATASET,
    SOURCE_REVISION as BBB_SOURCE_REVISION,
    label_record as label_bbb_record,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
    load_label_decisions as load_bioavailability_decisions,
)
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import (
    load_label_decisions as load_skin_decisions,
)


SCHEMA_VERSION = "processed_starling_gold.v1"
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    task: str
    lineage: str
    source_path: Path
    gold_root: Path
    output_root: Path
    source_kind: str
    expected_source_sha256: str
    source_git_revision: str = ""
    source_manifest_path: Path | None = None
    expected_manifest_sha256: str = ""


@dataclass(frozen=True)
class ResolvedSource:
    path: Path
    manifest_path: Path | None
    locator: str


@dataclass
class VotingRecord:
    source_index: int
    source_id: str
    source_record_id: str
    pmid: str
    label_method: str
    raw_value: str
    vote: int
    molecule_key: str
    source_record: dict[str, Any] = field(default_factory=dict)
    value_type: str = "categorical"
    observed_value: float | None = None
    contribution: float | None = None
    contribution_method: str = ""


TASK_SPECS = {
    "bbb_martins": TaskSpec(
        key="bbb_martins",
        task="BBB_Martins",
        lineage="experimental_meaningful_cns_access_v2",
        source_path=Path("data/starling_data/bbb_martins/Direct_BBB/records.parquet"),
        gold_root=Path(
            "data/processed_starling_experimental_meaningful_cns_access_v2/"
            "BBB_Martins/scaffold"
        ),
        output_root=Path(
            "data/experimental_meaningful_cns_access_v2_processed/"
            "BBB_Martins/scaffold"
        ),
        source_kind="bbb",
        expected_source_sha256="14c01314377b6c31f18e6ec9c1ee72a8605a06800d74385fc8a7273f2334df57",
    ),
    "bioavailability_ma": TaskSpec(
        key="bioavailability_ma",
        task="Bioavailability_Ma",
        lineage="record_supported_v2",
        source_path=Path(
            "data/starling_data/bioavailability_ma/canonical_direct_v2/"
            "direct_claims.parquet"
        ),
        gold_root=Path(
            "data/processed_starling_record_supported_v2/"
            "Bioavailability_Ma/scaffold"
        ),
        output_root=Path(
            "data/record_supported_v2_processed/Bioavailability_Ma/scaffold"
        ),
        source_kind="bioavailability",
        expected_source_sha256="045261cbda785092143eeadd636f78399f7b02f951b23480b16fb8dde22661c5",
        source_git_revision="5a6484f156edb2612b61e95b8f7c38116a8ea021",
        source_manifest_path=Path(
            "data/starling_data/bioavailability_ma/canonical_direct_v2/merge_manifest.json"
        ),
        expected_manifest_sha256="b986a214491f3b04c6f3ae6fc71cbd4eacad08d390b9b62de53f953966b0a257",
    ),
    "skin_reaction": TaskSpec(
        key="skin_reaction",
        task="Skin_Reaction",
        lineage="record_supported_v2",
        source_path=Path(
            "data/starling_data/skin_reaction/direct_skin_reaction/extractions.parquet"
        ),
        gold_root=Path(
            "data/processed_starling_record_supported_v2/Skin_Reaction/scaffold"
        ),
        output_root=Path("data/record_supported_v2_processed/Skin_Reaction/scaffold"),
        source_kind="skin",
        expected_source_sha256="e7c4819c98af47eae4dc0563f40dc39f8b09eb8e03c9626daee603147125f9c1",
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_gold_rows(spec: TaskSpec) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = spec.gold_root / f"{split}_molecule_labels.jsonl"
        for row in _read_jsonl(path):
            key = str(row["molecule_identity_key"])
            if key in rows:
                raise ValueError(f"duplicate gold molecule identity: {key}")
            rows[key] = {**row, "split": split}
    return rows


def _identity_key(smiles: str) -> str | None:
    identity = normalize_molecule_identity(smiles)
    if identity.status != "ok" or not identity.parent_smiles:
        return None
    return identity.parent_inchi_key or identity.parent_smiles


def _record_from_decision(
    source_index: int,
    decision: Any,
    gold_rows: Mapping[str, Mapping[str, Any]],
) -> VotingRecord | None:
    record = decision.record
    if record is None:
        return None
    molecule_key = _identity_key(record.smiles)
    if molecule_key not in gold_rows:
        return None
    return VotingRecord(
        source_index=source_index,
        source_id=record.source_id,
        source_record_id=record.source_record_id,
        pmid=record.pmid,
        label_method=record.label_method,
        raw_value=record.raw_value,
        vote=int(record.label),
        molecule_key=str(molecule_key),
    )


def _collect_adapter_votes(
    spec: TaskSpec,
    gold_rows: Mapping[str, Mapping[str, Any]],
    source: ResolvedSource,
) -> list[VotingRecord]:
    loader = (
        load_bioavailability_decisions
        if spec.source_kind == "bioavailability"
        else load_skin_decisions
    )
    loader_kwargs = {"source_path": source.path}
    if spec.source_kind == "bioavailability":
        loader_kwargs["manifest_path"] = source.manifest_path
    decisions, _ = loader(**loader_kwargs)
    votes = []
    for source_index, decision in enumerate(decisions):
        vote = _record_from_decision(source_index, decision, gold_rows)
        if vote is not None:
            if spec.source_kind == "bioavailability":
                vote.source_id = f"canonical:{spec.source_path}"
            votes.append(vote)
    _attach_source_rows(source.path, votes)
    return votes


def _collect_bbb_votes(
    spec: TaskSpec,
    gold_rows: Mapping[str, Mapping[str, Any]],
    source: ResolvedSource,
) -> list[VotingRecord]:
    manifest_path = spec.source_path.with_name("source_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["resolved_revision"] != BBB_SOURCE_REVISION:
        raise ValueError("BBB local source revision does not match the frozen gold revision")
    votes = []
    for source_index, row in _iter_parquet_rows(source.path):
        stored_index = int(row["source_index"])
        if stored_index != source_index:
            raise ValueError("BBB source_index is not aligned with Parquet row order")
        label, method = label_bbb_record(row, source_index=source_index)
        molecule_key = _identity_key(str(row.get("smiles") or "")) if label is not None else None
        if molecule_key not in gold_rows:
            continue
        votes.append(
            VotingRecord(
                source_index=source_index,
                source_id=BBB_SOURCE_DATASET,
                source_record_id=f"row:{source_index}",
                pmid=str(row.get("pmid") or ""),
                label_method=method,
                raw_value=_bbb_raw_value(row),
                vote=int(label),
                molecule_key=str(molecule_key),
                source_record=_json_safe(row),
            )
        )
    return votes


def _bbb_raw_value(row: Mapping[str, Any]) -> str:
    fields = (
        row.get("bbb_permeability_label"),
        row.get("quant_metric"),
        row.get("quant_value"),
        row.get("quant_units"),
    )
    return " | ".join(str(value) for value in fields if _has_value(value))


def _iter_parquet_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    source_index = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192):
        columns = batch.to_pydict()
        for offset in range(batch.num_rows):
            yield source_index, {
                name: values[offset] for name, values in columns.items()
            }
            source_index += 1


def _attach_source_rows(path: Path, votes: list[VotingRecord]) -> None:
    by_index = {record.source_index: record for record in votes}
    remaining = set(by_index)
    for source_index, row in _iter_parquet_rows(path):
        if source_index not in remaining:
            continue
        by_index[source_index].source_record = _json_safe(row)
        remaining.remove(source_index)
    if remaining:
        raise ValueError(f"missing {len(remaining)} source rows from {path}")


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in {"", "nan", "none", "null"}


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return _json_safe(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _git_materialize(revision: str, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        result = subprocess.run(
            ["git", "show", f"{revision}:{source}"],
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))


def _validate_resolved_source(spec: TaskSpec, source: ResolvedSource) -> None:
    actual = sha256_file(source.path)
    if actual != spec.expected_source_sha256:
        raise ValueError(
            f"{spec.task} source hash mismatch: expected "
            f"{spec.expected_source_sha256}, found {actual}"
        )
    if source.manifest_path is None:
        return
    manifest_hash = sha256_file(source.manifest_path)
    if manifest_hash != spec.expected_manifest_sha256:
        raise ValueError(
            f"{spec.task} source manifest hash mismatch: expected "
            f"{spec.expected_manifest_sha256}, found {manifest_hash}"
        )


@contextmanager
def _resolve_source(spec: TaskSpec) -> Iterable[ResolvedSource]:
    current_hash = sha256_file(spec.source_path)
    current_manifest = spec.source_manifest_path
    manifest_matches = (
        current_manifest is None
        or sha256_file(current_manifest) == spec.expected_manifest_sha256
    )
    if current_hash == spec.expected_source_sha256 and manifest_matches:
        source = ResolvedSource(spec.source_path, current_manifest, str(spec.source_path))
        _validate_resolved_source(spec, source)
        yield source
        return
    if not spec.source_git_revision:
        raise ValueError(f"{spec.task} frozen source is unavailable in the current checkout")
    with TemporaryDirectory(prefix=f"{spec.key}_frozen_source_") as temporary:
        root = Path(temporary)
        source_path = root / spec.source_path.name
        _git_materialize(spec.source_git_revision, spec.source_path, source_path)
        manifest_path = None
        if spec.source_manifest_path is not None:
            manifest_path = root / spec.source_manifest_path.name
            _git_materialize(
                spec.source_git_revision,
                spec.source_manifest_path,
                manifest_path,
            )
        locator = f"git:{spec.source_git_revision}:{spec.source_path}"
        source = ResolvedSource(source_path, manifest_path, locator)
        _validate_resolved_source(spec, source)
        yield source


def scalarize_bioavailability_vote(record: VotingRecord) -> float | None:
    """Return a defensible central percentage for a numeric voting record."""
    if "numeric_20_percent_threshold" not in record.label_method:
        return None
    method = record.label_method.rsplit(":", 1)[-1]
    if method in {"lower_bound", "upper_bound"}:
        return None
    interval = parse_numeric_interval(record.raw_value, fraction_to_percent=True)
    if interval is None:
        raise ValueError(f"cannot reparse accepted numeric vote: {record.raw_value!r}")
    scalar_methods = {"reported_point", "reported_mean_plus_minus", "reported_range"}
    if method not in scalar_methods:
        raise ValueError(f"unsupported accepted numeric method: {method}")
    if not math.isfinite(interval.lower) or not math.isfinite(interval.upper):
        raise ValueError("scalarizable Bioavailability interval is not finite")
    return (interval.lower + interval.upper) / 2.0


def _prepare_contributions(
    spec: TaskSpec,
    votes: list[VotingRecord],
    gold_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if spec.source_kind != "bioavailability":
        for record in votes:
            record.contribution = float(record.vote)
            record.contribution_method = "categorical_vote"
        return {"unit": "positive_vote_fraction", "class_means": None}
    for record in votes:
        record.value_type = (
            "continuous"
            if "numeric_20_percent_threshold" in record.label_method
            else "categorical"
        )
        record.observed_value = scalarize_bioavailability_vote(record)
    class_means, calibration_counts = _bioavailability_class_means(votes, gold_rows)
    for record in votes:
        if record.observed_value is not None:
            record.contribution = record.observed_value
            record.contribution_method = _observed_method(record)
        else:
            record.contribution = class_means[record.vote]
            record.contribution_method = (
                "train_class_mean_categorical"
                if record.value_type == "categorical"
                else "train_class_mean_one_sided_bound"
            )
    return {
        "unit": "percent_oral_bioavailability",
        "positive_threshold": 20.0,
        "class_means": {str(label): value for label, value in class_means.items()},
        "calibration_counts": {
            str(label): count for label, count in calibration_counts.items()
        },
        "calibration_split": "train",
    }


def _bioavailability_class_means(
    votes: Iterable[VotingRecord],
    gold_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[int, float], Counter[int]]:
    values: dict[int, list[float]] = defaultdict(list)
    for record in votes:
        if gold_rows[record.molecule_key]["split"] != "train":
            continue
        if record.observed_value is not None:
            values[record.vote].append(record.observed_value)
    if not values[0] or not values[1]:
        raise ValueError("Bioavailability train calibration requires both vote classes")
    means = {label: math.fsum(items) / len(items) for label, items in values.items()}
    if not means[0] < 20.0 <= means[1]:
        raise ValueError(f"Bioavailability class means cross the frozen threshold: {means}")
    return means, Counter({label: len(items) for label, items in values.items()})


def _observed_method(record: VotingRecord) -> str:
    method = record.label_method.rsplit(":", 1)[-1]
    return {
        "reported_point": "observed_point",
        "reported_mean_plus_minus": "observed_reported_mean",
        "reported_range": "observed_closed_range_midpoint",
    }[method]


def _validate_votes(
    gold_rows: Mapping[str, Mapping[str, Any]],
    votes: Iterable[VotingRecord],
) -> dict[str, list[VotingRecord]]:
    grouped: dict[str, list[VotingRecord]] = defaultdict(list)
    for record in votes:
        grouped[record.molecule_key].append(record)
    if set(grouped) != set(gold_rows):
        missing = len(set(gold_rows) - set(grouped))
        extra = len(set(grouped) - set(gold_rows))
        raise ValueError(f"voting-record molecule mismatch: missing={missing}, extra={extra}")
    for key, gold in gold_rows.items():
        records = grouped[key]
        counts = Counter(record.vote for record in records)
        expected = {int(label): int(count) for label, count in gold["label_counts"].items()}
        if len(records) != int(gold["source_record_count"]) or counts != expected:
            raise ValueError(f"voting-record reconstruction failed for {key}")
    return grouped


def _record_key(spec: TaskSpec, record: VotingRecord) -> str:
    return f"{spec.task}:source_row:{record.source_index}"


def _enrich_molecule(
    spec: TaskSpec,
    gold: Mapping[str, Any],
    records: list[VotingRecord],
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record.source_index)
    value_types = {record.value_type for record in ordered}
    record_type = (
        "categorical_only"
        if value_types == {"categorical"}
        else "continuous_only"
        if value_types == {"continuous"}
        else "mixed"
    )
    contributions = [float(record.contribution) for record in ordered]
    method = (
        "mean_observed_and_train_class_mean_imputed_votes"
        if spec.source_kind == "bioavailability"
        else "mean_binary_record_votes"
    )
    return {
        **gold,
        "voting_record_type": record_type,
        "voting_record_count": len(ordered),
        "continuous_value_mean": math.fsum(contributions) / len(contributions),
        "continuous_value_unit": (
            "percent_oral_bioavailability"
            if spec.source_kind == "bioavailability"
            else "positive_vote_fraction"
        ),
        "continuous_value_method": method,
        "voting_record_keys": [_record_key(spec, record) for record in ordered],
    }


def _serialize_record(
    spec: TaskSpec,
    record: VotingRecord,
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "voting_record_key": _record_key(spec, record),
        "task": spec.task,
        "source_lineage": spec.lineage,
        "split": gold["split"],
        "drug": gold["drug"],
        "molecule_identity_key": record.molecule_key,
        "molecule_Y": int(gold["Y"]),
        "record_vote": record.vote,
        "voting_value_type": record.value_type,
        "observed_continuous_value": record.observed_value,
        "continuous_value_contribution": record.contribution,
        "continuous_value_unit": (
            "percent_oral_bioavailability"
            if spec.source_kind == "bioavailability"
            else "positive_vote_fraction"
        ),
        "continuous_value_contribution_method": record.contribution_method,
        "source_id": record.source_id,
        "source_record_id": record.source_record_id,
        "source_row_index": record.source_index,
        "pmid": record.pmid,
        "label_method": record.label_method,
        "source_record": record.source_record,
    }


def _write_task_artifacts(
    spec: TaskSpec,
    gold_rows: Mapping[str, Mapping[str, Any]],
    votes: list[VotingRecord],
    value_policy: Mapping[str, Any],
    source: ResolvedSource,
) -> dict[str, Any]:
    grouped = _validate_votes(gold_rows, votes)
    enriched = {
        key: _enrich_molecule(spec, gold, grouped[key])
        for key, gold in gold_rows.items()
    }
    paths: dict[str, Path] = {}
    for split in SPLITS:
        rows = sorted(
            (row for row in enriched.values() if row["split"] == split),
            key=lambda row: row["molecule_identity_key"],
        )
        paths[f"{split}_molecule_labels"] = spec.output_root / f"{split}_molecule_labels.jsonl"
        write_jsonl_atomic(paths[f"{split}_molecule_labels"], rows)
    heldout = sorted(
        (row for row in enriched.values() if row["split"] in {"valid", "test"}),
        key=lambda row: row["molecule_identity_key"],
    )
    paths["heldout_molecule_labels"] = spec.output_root / "heldout_molecule_labels.jsonl"
    write_jsonl_atomic(paths["heldout_molecule_labels"], heldout)
    record_rows = [
        _serialize_record(spec, record, gold_rows[record.molecule_key])
        for record in votes
    ]
    record_rows.sort(
        key=lambda row: (
            SPLITS.index(row["split"]),
            row["molecule_identity_key"],
            row["source_row_index"],
        )
    )
    paths["voting_records"] = spec.output_root / "voting_records.jsonl"
    write_jsonl_atomic(paths["voting_records"], record_rows)
    return _task_summary(spec, enriched, record_rows, value_policy, paths, source)


def _task_summary(
    spec: TaskSpec,
    enriched: Mapping[str, Mapping[str, Any]],
    record_rows: list[Mapping[str, Any]],
    value_policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    source: ResolvedSource,
) -> dict[str, Any]:
    source_files = {
        split: spec.gold_root / f"{split}_molecule_labels.jsonl" for split in SPLITS
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task": spec.task,
        "source_lineage": spec.lineage,
        "source_gold_root": str(spec.gold_root),
        "source_parquet": str(spec.source_path),
        "source_parquet_locator": source.locator,
        "source_parquet_sha256": sha256_file(source.path),
        "source_gold_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "n_molecules": len(enriched),
        "n_voting_records": len(record_rows),
        "split_molecule_counts": dict(Counter(row["split"] for row in enriched.values())),
        "split_voting_record_counts": dict(Counter(row["split"] for row in record_rows)),
        "voting_record_type_counts": dict(
            Counter(row["voting_record_type"] for row in enriched.values())
        ),
        "continuous_value_policy": dict(value_policy),
        "paths": {name: str(path) for name, path in paths.items()},
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    summary_path = spec.output_root / "summary.json"
    write_json_atomic(summary_path, summary)
    return summary


def build_task(spec: TaskSpec) -> dict[str, Any]:
    gold_rows = _load_gold_rows(spec)
    with _resolve_source(spec) as source:
        votes = (
            _collect_bbb_votes(spec, gold_rows, source)
            if spec.source_kind == "bbb"
            else _collect_adapter_votes(spec, gold_rows, source)
        )
        value_policy = _prepare_contributions(spec, votes, gold_rows)
        return _write_task_artifacts(spec, gold_rows, votes, value_policy, source)


def _write_lineage_summaries(specs: Iterable[TaskSpec]) -> None:
    roots: dict[Path, list[TaskSpec]] = defaultdict(list)
    for spec in specs:
        roots[spec.output_root.parents[1]].append(spec)
    for root, root_specs in roots.items():
        tasks = {}
        for summary_path in sorted(root.glob("*/scaffold/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            tasks[str(summary["task"])] = summary
        write_json_atomic(
            root / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_lineage": root_specs[0].lineage,
                "tasks": tasks,
            },
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", *TASK_SPECS],
        default=["all"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    keys = list(TASK_SPECS) if args.tasks == ["all"] else args.tasks
    specs = [TASK_SPECS[key] for key in keys]
    summaries = {}
    for spec in specs:
        print(f"[processed-gold] building {spec.task}", flush=True)
        summaries[spec.task] = build_task(spec)
        print(
            f"[processed-gold] {spec.task}: "
            f"molecules={summaries[spec.task]['n_molecules']:,} "
            f"records={summaries[spec.task]['n_voting_records']:,}",
            flush=True,
        )
    _write_lineage_summaries(specs)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
