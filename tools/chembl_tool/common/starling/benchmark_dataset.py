"""Common machinery for building binary benchmark splits from Starling records."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    normalize_molecule_identity,
)


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"


@dataclass(frozen=True)
class NumericInterval:
    """A reported numeric value represented without inventing a point estimate."""

    lower: float
    upper: float
    method: str
    normalized_text: str


@dataclass(frozen=True)
class LabeledSourceRecord:
    """One source extraction that can be mapped to the task's binary label."""

    smiles: str
    label: int
    source_id: str
    source_record_id: str = ""
    pmid: str = ""
    label_method: str = ""
    raw_value: str = ""
    context: str = ""


@dataclass(frozen=True)
class LabelDecision:
    """Accepted label or an auditable reason why a source row was not used."""

    record: LabeledSourceRecord | None
    reason: str = ""
    example: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSplit:
    """One deterministic train/test partition over accepted parent molecules."""

    method: str
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
    target_test_size: int


def accepted(record: LabeledSourceRecord) -> LabelDecision:
    return LabelDecision(record=record)


def rejected(reason: str, **example: Any) -> LabelDecision:
    return LabelDecision(record=None, reason=reason, example=example)


def parse_numeric_interval(
    value: Any,
    *,
    fraction_to_percent: bool = False,
) -> NumericInterval | None:
    """Parse the first reported value/range while preserving threshold ambiguity."""
    text = normalize_numeric_text(value)
    if not text:
        return None
    first_number = re.search(NUMBER_PATTERN, text)
    if first_number is None:
        return None

    prefix = text[max(0, first_number.start() - 24) : first_number.start()].lower()
    first_value = float(first_number.group(0))
    scale = _percent_scale(text, first_value, fraction_to_percent=fraction_to_percent)

    range_match = re.match(
        rf"\s*({NUMBER_PATTERN})\s*(?:-|to)\s*({NUMBER_PATTERN})",
        text[first_number.start() :],
        flags=re.IGNORECASE,
    )
    if range_match:
        left = float(range_match.group(1)) * scale
        right = float(range_match.group(2)) * scale
        return NumericInterval(min(left, right), max(left, right), "reported_range", text)

    relation = _relation_from_prefix(prefix, text[first_number.start() : first_number.start() + 2])
    value_scaled = first_value * scale
    if relation == "lower_bound":
        return NumericInterval(value_scaled, math.inf, relation, text)
    if relation == "upper_bound":
        return NumericInterval(-math.inf, value_scaled, relation, text)

    plus_minus = re.match(
        rf"\s*({NUMBER_PATTERN})\s*(?:±|\+/-|\+-|plus/minus)\s*({NUMBER_PATTERN})",
        text[first_number.start() :],
        flags=re.IGNORECASE,
    )
    if plus_minus:
        center = float(plus_minus.group(1)) * scale
        error = abs(float(plus_minus.group(2)) * scale)
        return NumericInterval(center - error, center + error, "reported_mean_plus_minus", text)
    return NumericInterval(value_scaled, value_scaled, "reported_point", text)


def classify_interval(interval: NumericInterval, *, threshold: float) -> int | None:
    """Apply a >= positive threshold only when the entire interval is one-sided."""
    if interval.lower >= threshold:
        return 1
    if interval.upper < threshold:
        return 0
    return None


def normalize_numeric_text(value: Any) -> str:
    text = str(value or "").strip()
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
        "％": "%",
        "﹪": "%",
        "per cent": "%",
        "percent": "%",
        "approximately": "about",
        "approx.": "about",
        "approx": "about",
    }
    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def has_reported_text(value: Any) -> bool:
    """Return whether a nullable dataframe/dataset field contains real text."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in {"", "nan", "none", "null", "n/a", "na"}


def build_benchmark_dataset(
    *,
    task_name: str,
    decisions: Iterable[LabelDecision],
    source_metadata: Mapping[str, Any],
    output_dir: str | Path,
    max_test_size: int = 500,
    test_fraction: float = 0.2,
    seed: int = 20260723,
    max_rejection_examples: int = 20,
) -> dict[str, Any]:
    """Aggregate consistent parent labels and write random/scaffold split artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped: dict[str, list[tuple[LabeledSourceRecord, dict[str, Any]]]] = defaultdict(list)

    for decision in decisions:
        source_counts["n_source_rows_considered"] += 1
        if decision.record is None:
            reason = decision.reason or "unspecified_rejection"
            rejection_counts[reason] += 1
            if len(rejection_examples[reason]) < max_rejection_examples:
                rejection_examples[reason].append(dict(decision.example))
            continue

        source_counts["n_source_rows_labeled"] += 1
        record = decision.record
        identity = normalize_molecule_identity(record.smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            rejection_counts["invalid_or_unresolved_smiles"] += 1
            if len(rejection_examples["invalid_or_unresolved_smiles"]) < max_rejection_examples:
                rejection_examples["invalid_or_unresolved_smiles"].append(
                    {"source_id": record.source_id, "source_record_id": record.source_record_id}
                )
            continue
        identity_key = identity.parent_inchi_key or identity.parent_smiles
        grouped[identity_key].append((record, identity.to_dict()))

    molecule_rows: list[dict[str, Any]] = []
    conflicting_rows: list[dict[str, Any]] = []
    for identity_key, items in sorted(grouped.items()):
        labels = sorted({record.label for record, _ in items})
        label_counts = Counter(record.label for record, _ in items)
        identity = items[0][1]
        base = {
            "drug": identity["parent_smiles"],
            "molecule_identity_key": identity_key,
            "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
            "source_record_count": len(items),
            "source_ids": sorted({record.source_id for record, _ in items}),
            "label_methods": dict(Counter(record.label_method for record, _ in items)),
            "source_pmids": _unique_limited((record.pmid for record, _ in items), 50),
            "source_record_ids": _unique_limited((record.source_record_id for record, _ in items), 50),
            "raw_value_examples": _unique_limited((record.raw_value for record, _ in items), 20),
            "context_examples": _unique_limited((record.context for record, _ in items), 10),
            "molecule_identity": identity,
        }
        if len(labels) != 1:
            conflicting_rows.append({**base, "drop_reason": "conflicting_parent_level_labels"})
            continue
        molecule_rows.append(
            {
                **base,
                "Y": labels[0],
                "bemis_murcko_scaffold": bemis_murcko_scaffold(identity["parent_smiles"]),
            }
        )

    target_test_size = calculate_test_size(
        len(molecule_rows),
        max_test_size=max_test_size,
        test_fraction=test_fraction,
    )
    random_train, random_test = stratified_hash_split(
        molecule_rows,
        test_size=target_test_size,
        seed=seed,
    )
    scaffold_train, scaffold_test = scaffold_group_split(
        molecule_rows,
        test_size=target_test_size,
        seed=seed,
    )
    splits = {
        "random": DatasetSplit(
            method="label_stratified_stable_hash",
            train=random_train,
            test=random_test,
            target_test_size=target_test_size,
        ),
        "scaffold": DatasetSplit(
            method="bemis_murcko_scaffold_group_subset_sum",
            train=scaffold_train,
            test=scaffold_test,
            target_test_size=target_test_size,
        ),
    }
    split_test_keys = {
        name: {row["molecule_identity_key"] for row in split.test}
        for name, split in splits.items()
    }
    labeled_rows = [
        {
            **row,
            "split_assignments": {
                name: (
                    "test"
                    if row["molecule_identity_key"] in split_test_keys[name]
                    else "train"
                )
                for name in splits
            },
        }
        for row in molecule_rows
    ]
    _write_jsonl(output_path / "molecule_labels.jsonl", labeled_rows)
    _write_jsonl(output_path / "conflicting_molecules.jsonl", conflicting_rows)
    _write_jsonl(
        output_path / "source_rejection_examples.jsonl",
        (
            {"reason": reason, "example": example}
            for reason in sorted(rejection_examples)
            for example in rejection_examples[reason]
        ),
    )

    split_summaries = {
        name: _write_split_artifacts(output_path, name, split)
        for name, split in splits.items()
    }
    summary = {
        "task": task_name,
        "protocol_version": "starling_binary_benchmark.v2",
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "seed": seed,
        "test_size_policy": {
            "formula": "min(max_test_size, floor(test_fraction * n_binary_molecules))",
            "max_test_size": max_test_size,
            "test_fraction": test_fraction,
            "target_test_size": target_test_size,
        },
        "n_source_rows_considered": source_counts["n_source_rows_considered"],
        "n_source_rows_labeled_before_structure_normalization": source_counts["n_source_rows_labeled"],
        "source_rejection_counts": dict(sorted(rejection_counts.items())),
        "n_parent_groups_with_any_label": len(grouped),
        "n_conflicting_parent_groups": len(conflicting_rows),
        "n_binary_molecules": len(molecule_rows),
        "all_label_counts": _label_counts(molecule_rows),
        "n_unique_bemis_murcko_scaffolds": len(
            {row["bemis_murcko_scaffold"] for row in molecule_rows}
        ),
        "n_acyclic_molecules": sum(
            not row["bemis_murcko_scaffold"] for row in molecule_rows
        ),
        "splits": split_summaries,
        "cross_split_test_identity_overlap": len(
            split_test_keys["random"] & split_test_keys["scaffold"]
        ),
        "source_metadata": dict(source_metadata),
        "paths": {
            "molecule_labels": str(output_path / "molecule_labels.jsonl"),
            "conflicting_molecules": str(output_path / "conflicting_molecules.jsonl"),
            "source_rejection_examples": str(output_path / "source_rejection_examples.jsonl"),
        },
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_path / "report_zh.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def calculate_test_size(
    n_molecules: int,
    *,
    max_test_size: int = 500,
    test_fraction: float = 0.2,
) -> int:
    """Calculate floor(min(max_test_size, fraction * accepted molecules))."""
    if n_molecules < 2:
        raise ValueError("at least two binary molecules are required")
    if max_test_size <= 0:
        raise ValueError("max_test_size must be positive")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    test_size = min(max_test_size, math.floor(test_fraction * n_molecules))
    if test_size <= 0:
        raise ValueError(
            f"test-size policy produced {test_size} for {n_molecules} molecules"
        )
    return test_size


def stratified_hash_split(
    rows: list[dict[str, Any]],
    *,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return an exact-size, label-stratified, deterministic molecule split."""
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if len(rows) < test_size:
        raise ValueError(f"need at least {test_size} binary molecules, found {len(rows)}")

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["Y"])].append(row)
    if set(by_label) != {0, 1}:
        raise ValueError(f"both binary labels are required, found {sorted(by_label)}")

    positive_target = round(test_size * len(by_label[1]) / len(rows))
    positive_target = min(max(1, positive_target), len(by_label[1]), test_size - 1)
    negative_target = test_size - positive_target
    if negative_target > len(by_label[0]):
        shift = negative_target - len(by_label[0])
        negative_target -= shift
        positive_target += shift
    if positive_target > len(by_label[1]):
        shift = positive_target - len(by_label[1])
        positive_target -= shift
        negative_target += shift

    test_keys: set[str] = set()
    for label, target in ((0, negative_target), (1, positive_target)):
        ordered = sorted(
            by_label[label],
            key=lambda row: (
                _stable_hash(f"{seed}\0{row['molecule_identity_key']}"),
                row["molecule_identity_key"],
            ),
        )
        test_keys.update(row["molecule_identity_key"] for row in ordered[:target])

    train = sorted(
        (row for row in rows if row["molecule_identity_key"] not in test_keys),
        key=lambda row: row["molecule_identity_key"],
    )
    test = sorted(
        (row for row in rows if row["molecule_identity_key"] in test_keys),
        key=lambda row: row["molecule_identity_key"],
    )
    if len(test) != test_size:
        raise AssertionError(f"expected {test_size} test molecules, found {len(test)}")
    return train, test


def bemis_murcko_scaffold(smiles: str) -> str:
    """Return the canonical Bemis-Murcko scaffold; acyclic molecules map to empty."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"cannot calculate scaffold for invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )


def scaffold_group_split(
    rows: list[dict[str, Any]],
    *,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seed-order whole scaffolds and subset-sum toward the test target from below."""
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if len(rows) < test_size:
        raise ValueError(f"need at least {test_size} binary molecules, found {len(rows)}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["bemis_murcko_scaffold"])].append(row)

    ordered_groups = sorted(
        (
            (scaffold, group)
            for scaffold, group in groups.items()
            if len(group) <= test_size
        ),
        key=lambda item: (
            _stable_hash(f"{seed}\0scaffold\0{item[0]}"),
            item[0],
        ),
    )
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for scaffold, group in ordered_groups:
        group_size = len(group)
        for current_size in sorted(tuple(reachable), reverse=True):
            new_size = current_size + group_size
            if new_size > test_size or new_size in reachable:
                continue
            reachable[new_size] = (*reachable[current_size], scaffold)
        if test_size in reachable:
            break

    actual_test_size = max(reachable)
    if actual_test_size <= 0:
        raise ValueError("no scaffold group fits within the requested test size")
    test_scaffolds = set(reachable[actual_test_size])
    train = sorted(
        (row for row in rows if row["bemis_murcko_scaffold"] not in test_scaffolds),
        key=lambda row: row["molecule_identity_key"],
    )
    test = sorted(
        (row for row in rows if row["bemis_murcko_scaffold"] in test_scaffolds),
        key=lambda row: row["molecule_identity_key"],
    )
    train_scaffolds = {row["bemis_murcko_scaffold"] for row in train}
    observed_test_scaffolds = {row["bemis_murcko_scaffold"] for row in test}
    if train_scaffolds & observed_test_scaffolds:
        raise AssertionError("scaffold leakage detected between train and test")
    if len(test) != actual_test_size:
        raise AssertionError(
            f"expected {actual_test_size} scaffold-test molecules, found {len(test)}"
        )
    return train, test


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relation_from_prefix(prefix: str, immediate: str) -> str:
    text = f"{prefix} {immediate}".lower()
    if re.search(r"(?:>=|≥|at least|not less than|greater than|more than|above)", text):
        return "lower_bound"
    if re.search(r"(?:<=|≤|at most|not more than|less than|lower than|below|up to)", text):
        return "upper_bound"
    return ""


def _percent_scale(text: str, first_value: float, *, fraction_to_percent: bool) -> float:
    if "%" in text:
        return 1.0
    if fraction_to_percent and 0.0 <= first_value <= 1.5:
        return 100.0
    return 1.0


def _unique_limited(values: Iterable[Any], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
            if len(output) >= limit:
                break
    return output


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _minimal_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {"drug": row["drug"], "Y": int(row["Y"])}


def _label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(int(row["Y"])) for row in rows)
    return {label: counts.get(label, 0) for label in ("0", "1")}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _write_split_artifacts(
    output_path: Path,
    name: str,
    split: DatasetSplit,
) -> dict[str, Any]:
    split_path = output_path / name
    split_path.mkdir(parents=True, exist_ok=True)
    train_keys = {row["molecule_identity_key"] for row in split.train}
    test_keys = {row["molecule_identity_key"] for row in split.test}
    if train_keys & test_keys:
        raise AssertionError(f"{name} split has molecule-identity overlap")

    train_scaffolds = {row["bemis_murcko_scaffold"] for row in split.train}
    test_scaffolds = {row["bemis_murcko_scaffold"] for row in split.test}
    scaffold_overlap = train_scaffolds & test_scaffolds
    if name == "scaffold" and scaffold_overlap:
        raise AssertionError("scaffold split has train/test scaffold overlap")

    train_details = (
        {**row, "split": "train", "split_method": split.method}
        for row in split.train
    )
    test_details = (
        {**row, "split": "test", "split_method": split.method}
        for row in split.test
    )
    _write_jsonl(split_path / "train.jsonl", _minimal_rows(split.train))
    _write_jsonl(split_path / "test.jsonl", _minimal_rows(split.test))
    _write_jsonl(split_path / "train_molecule_labels.jsonl", train_details)
    _write_jsonl(split_path / "test_molecule_labels.jsonl", test_details)

    summary = {
        "method": split.method,
        "target_test_size": split.target_test_size,
        "actual_test_size": len(split.test),
        "test_size_shortfall": split.target_test_size - len(split.test),
        "n_train": len(split.train),
        "n_test": len(split.test),
        "train_label_counts": _label_counts(split.train),
        "test_label_counts": _label_counts(split.test),
        "n_train_scaffolds": len(train_scaffolds),
        "n_test_scaffolds": len(test_scaffolds),
        "train_test_scaffold_overlap": len(scaffold_overlap),
        "train_test_identity_overlap": 0,
        "paths": {
            "train": str(split_path / "train.jsonl"),
            "test": str(split_path / "test.jsonl"),
            "train_molecule_labels": str(split_path / "train_molecule_labels.jsonl"),
            "test_molecule_labels": str(split_path / "test_molecule_labels.jsonl"),
        },
    }
    (split_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['task']} Starling 二分类数据构建报告",
        "",
        f"- 协议：`{summary['protocol_version']}`",
        f"- 分子身份：`{summary['identity_normalizer_version']}`",
        f"- seed：{summary['seed']}",
        f"- 可用二分类 parent：{summary['n_binary_molecules']:,}",
        f"- test target：{summary['test_size_policy']['target_test_size']:,}",
        f"- 因 parent-level 标签冲突而丢弃：{summary['n_conflicting_parent_groups']:,}",
        "",
        "## Splits",
        "",
        "| split | train | test | test Y=0 / Y=1 | scaffold overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, split in summary["splits"].items():
        lines.append(
            f"| {name} | {split['n_train']:,} | {split['n_test']:,} | "
            f"{split['test_label_counts']['0']:,} / {split['test_label_counts']['1']:,} | "
            f"{split['train_test_scaffold_overlap']:,} |"
        )
    lines.extend(
        [
            "",
            "## Source-row rejection",
            "",
            "```json",
            json.dumps(summary["source_rejection_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "完整 provenance 见 `molecule_labels.jsonl`；冲突分子和 source-row rejection 示例分别见",
            "`conflicting_molecules.jsonl` 与 `source_rejection_examples.jsonl`。",
            "",
            "`random/test_molecule_labels.jsonl` 与 `scaffold/test_molecule_labels.jsonl`",
            "分别是两套 retrieval 泄漏隔离清单。现有 full-source Starling index 不能直接用于这些 test。",
            "",
        ]
    )
    return "\n".join(lines)
