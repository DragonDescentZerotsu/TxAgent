"""Build an oral-bioavailability transfer benchmark from a HuggingFace dataset.

The source dataset contains literature-extracted oral bioavailability statements.
This script keeps a cleaned molecule-level audit table and builds pair-level
transfer examples from molecules measured under the same normalized condition key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma import (
    oral_bioavailability as oral_cleaning,
)


DEFAULT_DATASET = oral_cleaning.ORAL_BIOAVAILABILITY_DATASET
DEFAULT_SPLIT = "train"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf"
DEFAULT_RUN_ID = "absolute_broad_condition_v1"
LABEL_TO_COMPLETION = {"similar": "A", "different": "B"}
CONDITION_COLUMNS = [
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
EXCLUDED_FROM_CONDITION_TEXT = {
    "pmid",
    "support_text",
    "molecule_name",
    "oral_bioavailability_value",
    "bioavailability_report_type",
    "smiles",
}
SIMILARITY_BUCKETS = [
    ("very_close_analog", 0.95, 1.01),
    ("close_analog", 0.80, 0.95),
    ("moderate_analog", 0.60, 0.80),
    ("weak_analog", 0.40, 0.60),
    ("distant_analog", 0.20, 0.40),
    ("very_distant_analog", -0.01, 0.20),
]
NULL_STRINGS = {"", "none", "null", "nan", "n/a", "na", "not specified", "unknown"}
QUALITATIVE_REJECT_WORDS = {
    "high",
    "low",
    "moderate",
    "excellent",
    "complete",
    "negligible",
    "similar",
    "reduced",
    "increased",
    "increase",
    "decrease",
    "unchanged",
    "variable",
    "unity",
}
NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|\.\d+)"


@dataclass(frozen=True)
class CleanRow:
    source_index: int
    molecule_id: str
    molecule_name: str
    canonical_smiles: str
    oral_bioavailability_value_percent: float
    condition_text: str
    condition_key: str
    condition_key_hash: str
    parse_method: str
    parse_modifier: str
    raw_row: dict[str, Any]


@dataclass
class AggregateMolecule:
    aggregate_id: str
    condition_key: str
    condition_key_hash: str
    canonical_smiles: str
    molecule_names: list[str]
    value_percent: float
    n_source_rows: int
    source_indices: list[int]
    condition_text: str
    raw_metadata: list[dict[str, Any]]
    fp: Any


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(args.seed)

    out_dir = Path(args.out_root) / args.run_id
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, revision=args.revision or None, split=args.split)
    rows, dropped = clean_rows(dataset, args)
    aggregates = aggregate_rows(rows, args)
    candidate_pairs = build_candidate_pairs(aggregates, args, rng)
    eval_pairs = select_eval_pairs(candidate_pairs, args, rng)

    write_jsonl(out_dir / "molecule_records.jsonl", [clean_row_to_dict(row) for row in rows])
    write_tsv(out_dir / "molecule_records.tsv", [clean_row_to_flat_dict(row) for row in rows])
    write_jsonl(out_dir / "dropped_rows.jsonl", dropped)
    write_jsonl(out_dir / "aggregate_molecules.jsonl", [aggregate_to_dict(row) for row in aggregates])
    write_jsonl(out_dir / "pair_candidates.jsonl", candidate_pairs)
    write_tsv(out_dir / "pair_candidates.tsv", flatten_pair_rows(candidate_pairs))
    write_jsonl(out_dir / "eval_pairs.jsonl", eval_pairs)
    write_tsv(out_dir / "eval_pairs.tsv", flatten_pair_rows(eval_pairs))
    write_jsonl(out_dir / "hf_prompt_completion.jsonl", [pair_to_hf_record(row, args) for row in eval_pairs])
    plot_value_distribution(rows, figures_dir / "value_distribution.svg", figures_dir / "value_distribution.png")

    summary = summarize(
        dataset=dataset,
        rows=rows,
        dropped=dropped,
        aggregates=aggregates,
        candidate_pairs=candidate_pairs,
        eval_pairs=eval_pairs,
        args=args,
        elapsed_s=time.time() - started,
    )
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report_zh.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=oral_cleaning.ORAL_BIOAVAILABILITY_REVISION)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--allowed-report-types",
        default=",".join(sorted(oral_cleaning.ALLOWED_REPORT_TYPES)),
        help="Comma-separated bioavailability_report_type values to retain.",
    )
    parser.add_argument(
        "--condition-key-mode",
        choices=("exact_condition", "broad_condition"),
        default="broad_condition",
        help="exact_condition uses the full condition text; broad_condition excludes dose and extra_details from the key.",
    )
    parser.add_argument("--similar-delta", type=float, default=10.0)
    parser.add_argument("--different-delta", type=float, default=30.0)
    parser.add_argument("--min-value-percent", type=float, default=0.0)
    parser.add_argument("--max-value-percent", type=float, default=100.0)
    parser.add_argument("--max-pairs-per-condition", type=int, default=3000)
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument("--n-eval-pairs", type=int, default=3000)
    parser.add_argument("--balance-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args(argv)


def clean_rows(dataset: Any, args: argparse.Namespace) -> tuple[list[CleanRow], list[dict[str, Any]]]:
    allowed_report_types = {item.strip() for item in args.allowed_report_types.split(",") if item.strip()}
    cleaned, dropped = oral_cleaning.clean_oral_bioavailability_rows(
        dataset, allowed_report_types=allowed_report_types,
        min_value_percent=args.min_value_percent, max_value_percent=args.max_value_percent,
    )
    rows = []
    for item in cleaned:
        condition_key = build_condition_key(item.raw_row, item.condition_text, args.condition_key_mode)
        rows.append(
            CleanRow(
                source_index=item.source_index,
                molecule_id=f"hf_ob_row_{item.source_index}",
                molecule_name=item.molecule_name,
                canonical_smiles=item.canonical_smiles,
                oral_bioavailability_value_percent=item.value_percent,
                condition_text=item.condition_text,
                condition_key=condition_key,
                condition_key_hash=stable_hash(condition_key),
                parse_method=item.parse_method,
                parse_modifier=item.parse_modifier,
                raw_row=item.raw_row,
            )
        )
    return rows, dropped


def parse_bioavailability_value(value: Any) -> tuple[float, str, str] | None:
    return oral_cleaning.parse_bioavailability_value(value)


def normalize_value_text(value: Any) -> str:
    text = clean_text(value)
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
        "·": ".",
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


def has_only_qualitative_signal(text: str) -> bool:
    lowered = text.lower()
    if "%" not in lowered and re.search(
        r"\b(?:auc|fold|higher|lower|comparable|similar|increased|reduced|increase|decrease|unchanged)\b",
        lowered,
    ):
        return True
    if re.search(r"\d", lowered):
        return False
    return any(word in lowered for word in QUALITATIVE_REJECT_WORDS)


def parse_float_token(value: str) -> float:
    cleaned = re.sub(r"[<>≤≥~≈\s]", "", value)
    return float(cleaned)


def extract_numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in re.finditer(rf"(?<![A-Za-z]){NUMBER_PATTERN}", text)]


def extract_ranges(text: str) -> list[tuple[float, float]]:
    patterns = [
        rf"({NUMBER_PATTERN})\s*(?:-|to|and)\s*({NUMBER_PATTERN})\s*%?",
        rf"between\s+({NUMBER_PATTERN})\s+and\s+({NUMBER_PATTERN})",
        rf"range(?:d|s)?(?:\s+between|\s+of|\s*[:=])?\s*({NUMBER_PATTERN})\s*(?:-|to|and)\s*({NUMBER_PATTERN})",
    ]
    ranges = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            lo = float(match.group(1))
            hi = float(match.group(2))
            if lo <= hi:
                ranges.append((lo, hi))
            else:
                ranges.append((hi, lo))
    return ranges


def convert_to_percent(number: float, source_text: str) -> float:
    if re.search(r"%|\bper\s*cent\b|percent", source_text, flags=re.IGNORECASE):
        return number
    if 0.0 <= number <= 1.5:
        return number * 100.0
    return number


def canonicalize_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def build_condition_text(row: dict[str, Any]) -> str:
    lines = []
    for column in CONDITION_COLUMNS:
        value = clean_text(row.get(column)) or "not specified"
        lines.append(f"{column}: {value}")
    return "\n".join(lines)


def build_condition_key(row: dict[str, Any], condition_text: str, mode: str) -> str:
    if mode == "exact_condition":
        return normalize_key_text(condition_text)
    key_columns = ["species_or_population", "oral_exposure_mode", "qualifying_conditions", "comparator"]
    parts = [normalize_key_text(row.get(column)) or "not_specified" for column in key_columns]
    return "|".join(parts)


def normalize_key_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9%./+ -]+", "", text)
    return text.strip()


def aggregate_rows(rows: list[CleanRow], args: argparse.Namespace) -> list[AggregateMolecule]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    grouped: dict[tuple[str, str], list[CleanRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.condition_key, row.canonical_smiles)].append(row)

    aggregates = []
    for index, ((condition_key, canonical_smiles), items) in enumerate(sorted(grouped.items())):
        values = [item.oral_bioavailability_value_percent for item in items]
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        condition_text = Counter(item.condition_text for item in items).most_common(1)[0][0]
        molecule_names = sorted({item.molecule_name for item in items if item.molecule_name})
        condition_hash = stable_hash(condition_key)
        aggregates.append(
            AggregateMolecule(
                aggregate_id=f"hf_ob_mol_{index:07d}",
                condition_key=condition_key,
                condition_key_hash=condition_hash,
                canonical_smiles=canonical_smiles,
                molecule_names=molecule_names,
                value_percent=sum(values) / len(values),
                n_source_rows=len(items),
                source_indices=[item.source_index for item in items],
                condition_text=condition_text,
                raw_metadata=[item.raw_row for item in items],
                fp=fp,
            )
        )
    return aggregates


def build_candidate_pairs(
    aggregates: list[AggregateMolecule], args: argparse.Namespace, rng: random.Random
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[AggregateMolecule]] = defaultdict(list)
    for molecule in aggregates:
        by_condition[molecule.condition_key].append(molecule)

    candidate_pairs: list[dict[str, Any]] = []
    for condition_index, condition_key in enumerate(sorted(by_condition)):
        molecules = by_condition[condition_key]
        if len(molecules) < 2:
            continue
        local_pairs = sample_condition_pairs(molecules, args.max_pairs_per_condition, rng)
        for reference, query in local_pairs:
            label = label_pair(reference.value_percent, query.value_percent, args.similar_delta, args.different_delta)
            if label is None:
                continue
            pair = build_pair_record(
                pair_index=len(candidate_pairs),
                condition_index=condition_index,
                reference=reference,
                query=query,
                label=label,
                args=args,
            )
            candidate_pairs.append(pair)
            if len(candidate_pairs) >= args.max_candidate_pairs:
                return candidate_pairs
    return candidate_pairs


def sample_condition_pairs(
    molecules: list[AggregateMolecule], max_pairs: int, rng: random.Random
) -> list[tuple[AggregateMolecule, AggregateMolecule]]:
    n = len(molecules)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return [(molecules[i], molecules[j]) for i in range(n) for j in range(i + 1, n)]

    pairs = []
    seen = set()
    attempts = 0
    max_attempts = max_pairs * 20
    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(n), 2)
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((molecules[i], molecules[j]))
    return pairs


def label_pair(reference_value: float, query_value: float, similar_delta: float, different_delta: float) -> str | None:
    delta = abs(reference_value - query_value)
    if delta <= similar_delta:
        return "similar"
    if delta >= different_delta:
        return "different"
    return None


def build_pair_record(
    *,
    pair_index: int,
    condition_index: int,
    reference: AggregateMolecule,
    query: AggregateMolecule,
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tanimoto = DataStructs.TanimotoSimilarity(reference.fp, query.fp)
    abs_delta = abs(reference.value_percent - query.value_percent)
    condition_id = f"HF_ORAL_BIOAVAILABILITY_{reference.condition_key_hash}"
    return {
        "pair_index": pair_index,
        "label": label,
        "label_mode": "oral_bioavailability_percent_delta",
        "task_name": "oral_bioavailability_hf",
        "assay_chembl_id": condition_id,
        "assay_id": condition_index,
        "standard_type": "oral_bioavailability_percent",
        "target_chembl_id": "",
        "target_pref_name": "oral bioavailability",
        "target_type": "PK endpoint",
        "target_organism": "",
        "assay_type": "oral_bioavailability_literature",
        "assay_description": reference.condition_text,
        "condition_key_mode": args.condition_key_mode,
        "condition_key": reference.condition_key,
        "condition_key_hash": reference.condition_key_hash,
        "reference_molecule_chembl_id": reference.aggregate_id,
        "query_molecule_chembl_id": query.aggregate_id,
        "reference_smiles": reference.canonical_smiles,
        "query_smiles": query.canonical_smiles,
        "reference_molecule_name": "; ".join(reference.molecule_names[:5]),
        "query_molecule_name": "; ".join(query.molecule_names[:5]),
        "reference_activity_value": round(reference.value_percent, 6),
        "hidden_query_activity_value": round(query.value_percent, 6),
        "reference_oral_bioavailability_percent": round(reference.value_percent, 6),
        "hidden_query_oral_bioavailability_percent": round(query.value_percent, 6),
        "abs_activity_delta": round(abs_delta, 6),
        "similar_delta_threshold": args.similar_delta,
        "different_delta_threshold": args.different_delta,
        "tanimoto": round(float(tanimoto), 6),
        "similarity_bucket": similarity_bucket(float(tanimoto)),
        "reference_condition_text": reference.condition_text,
        "query_condition_text": query.condition_text,
        "reference_source_indices": reference.source_indices,
        "query_source_indices": query.source_indices,
        "reference_n_source_rows": reference.n_source_rows,
        "query_n_source_rows": query.n_source_rows,
        "baseline_tanimoto_0_50_prediction": "similar" if tanimoto >= 0.50 else "different",
        "baseline_tanimoto_0_48_prediction": "similar" if tanimoto >= 0.48 else "different",
    }


def select_eval_pairs(candidate_pairs: list[dict[str, Any]], args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    if args.n_eval_pairs <= 0 or len(candidate_pairs) <= args.n_eval_pairs:
        selected = list(candidate_pairs)
    elif args.balance_labels:
        selected = balanced_sample(candidate_pairs, args.n_eval_pairs, rng)
    else:
        selected = rng.sample(candidate_pairs, args.n_eval_pairs)
    rng.shuffle(selected)
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index
    return selected


def balanced_sample(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["label"], row["similarity_bucket"])].append(row)

    keys = sorted(by_key)
    if not keys:
        return []
    target_per_key = max(1, n // len(keys))
    selected = []
    leftovers = []
    for key in keys:
        bucket_rows = list(by_key[key])
        rng.shuffle(bucket_rows)
        selected.extend(bucket_rows[:target_per_key])
        leftovers.extend(bucket_rows[target_per_key:])
    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    elif len(selected) > n:
        selected = rng.sample(selected, n)
    return selected


def pair_to_hf_record(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metadata = {
        "sample_id": row.get("sample_index"),
        "pair_id": row["pair_index"],
        "source": args.dataset,
        "task_name": "oral_bioavailability_hf",
        "label_mode": row["label_mode"],
        "assay_type": row["assay_type"],
        "condition_key_mode": row["condition_key_mode"],
        "condition_key_hash": row["condition_key_hash"],
        "similarity_bucket": row["similarity_bucket"],
        "weighted_tanimoto": row["tanimoto"],
        "abs_activity_delta": row["abs_activity_delta"],
        "similar_delta_threshold": row["similar_delta_threshold"],
        "different_delta_threshold": row["different_delta_threshold"],
        "reference_activity_value": row["reference_activity_value"],
        "hidden_query_activity_value": row["hidden_query_activity_value"],
        "reference_source_indices": row["reference_source_indices"],
        "query_source_indices": row["query_source_indices"],
        "eval_subset": "oral_bioavailability_hf_eval",
    }
    return {
        "prompt": build_prompt(row),
        "completion": LABEL_TO_COMPLETION[row["label"]],
        "metadata": metadata,
    }


def build_prompt(row: dict[str, Any]) -> str:
    same_condition = row["reference_condition_text"] == row["query_condition_text"]
    if same_condition:
        condition_block = "## Experimental Context\n" + row["reference_condition_text"]
    else:
        condition_block = (
            "## Reference Experimental Context\n"
            + row["reference_condition_text"]
            + "\n\n## Query Experimental Context\n"
            + row["query_condition_text"]
        )
    return (
        "You are given two molecules and oral bioavailability evidence. Decide whether the reference "
        "molecule's oral bioavailability should transfer to the query molecule under the reported "
        "experimental context. Use (A) transfer if the expected absolute difference is <= "
        f"{row['similar_delta_threshold']:.1f} percentage points. Use (B) not transfer if it is >= "
        f"{row['different_delta_threshold']:.1f} percentage points. Ambiguous middle cases are not in this dataset.\n\n"
        + condition_block
        + "\n\n## Molecule A: reference\n"
        + f"- SMILES: {row['reference_smiles']}\n"
        + f"- Known oral bioavailability: {row['reference_activity_value']:.2f}%\n\n"
        + "## Molecule B: query\n"
        + f"- SMILES: {row['query_smiles']}\n"
        + "- Oral bioavailability: hidden for evaluation\n\n"
        + "Should the oral bioavailability behavior transfer from Molecule A to Molecule B?\n"
        + "(A) transfer\n"
        + "(B) not transfer\n\n"
        + "Answer:"
    )


def summarize(
    *,
    dataset: Any,
    rows: list[CleanRow],
    dropped: list[dict[str, Any]],
    aggregates: list[AggregateMolecule],
    candidate_pairs: list[dict[str, Any]],
    eval_pairs: list[dict[str, Any]],
    args: argparse.Namespace,
    elapsed_s: float,
) -> dict[str, Any]:
    values = [row.oral_bioavailability_value_percent for row in rows]
    condition_counts = Counter(row.condition_key_hash for row in rows)
    aggregate_condition_counts = Counter(row.condition_key_hash for row in aggregates)
    summary = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "split": args.split,
        "elapsed_s": round(elapsed_s, 3),
        "parameters": vars(args),
        "source_rows": len(dataset),
        "clean_rows": len(rows),
        "dropped_rows": len(dropped),
        "drop_reasons": dict(Counter(row["drop_reason"] for row in dropped)),
        "report_type_counts_source": dict(Counter(dataset["bioavailability_report_type"])),
        "parse_method_counts": dict(Counter(row.parse_method for row in rows)),
        "parse_modifier_counts": dict(Counter(row.parse_modifier or "exact" for row in rows)),
        "value_percent_summary": summarize_numbers(values),
        "condition_groups_line_level": summarize_group_counts(condition_counts),
        "aggregate_molecules": len(aggregates),
        "condition_groups_aggregate_level": summarize_group_counts(aggregate_condition_counts),
        "pair_candidates": summarize_pairs(candidate_pairs),
        "eval_pairs": summarize_pairs(eval_pairs),
        "files": {
            "molecule_records_jsonl": "molecule_records.jsonl",
            "molecule_records_tsv": "molecule_records.tsv",
            "dropped_rows_jsonl": "dropped_rows.jsonl",
            "aggregate_molecules_jsonl": "aggregate_molecules.jsonl",
            "pair_candidates_jsonl": "pair_candidates.jsonl",
            "pair_candidates_tsv": "pair_candidates.tsv",
            "eval_pairs_jsonl": "eval_pairs.jsonl",
            "eval_pairs_tsv": "eval_pairs.tsv",
            "hf_prompt_completion_jsonl": "hf_prompt_completion.jsonl",
            "value_distribution_svg": "figures/value_distribution.svg",
            "value_distribution_png": "figures/value_distribution.png",
        },
    }
    return summary


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "n": len(values),
        "min": round(min(values), 6),
        "p05": round(quantile(sorted_values, 0.05), 6),
        "p25": round(quantile(sorted_values, 0.25), 6),
        "median": round(quantile(sorted_values, 0.5), 6),
        "mean": round(sum(values) / len(values), 6),
        "p75": round(quantile(sorted_values, 0.75), 6),
        "p95": round(quantile(sorted_values, 0.95), 6),
        "max": round(max(values), 6),
        "std": round(statistics.pstdev(values), 6),
    }


def summarize_group_counts(counter: Counter) -> dict[str, Any]:
    counts = list(counter.values())
    if not counts:
        return {"n_groups": 0}
    return {
        "n_groups": len(counts),
        "min_size": min(counts),
        "median_size": quantile(sorted(counts), 0.5),
        "mean_size": sum(counts) / len(counts),
        "max_size": max(counts),
        "groups_with_at_least_2": sum(1 for count in counts if count >= 2),
        "groups_with_at_least_5": sum(1 for count in counts if count >= 5),
    }


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "similarity_bucket_counts": dict(Counter(row["similarity_bucket"] for row in rows)),
        "label_similarity_bucket_counts": {
            f"{label}|{bucket}": count
            for (label, bucket), count in Counter((row["label"], row["similarity_bucket"]) for row in rows).items()
        },
        "baseline_metrics": {
            "tanimoto_0_50": compute_prediction_metrics(rows, "baseline_tanimoto_0_50_prediction"),
            "tanimoto_0_48": compute_prediction_metrics(rows, "baseline_tanimoto_0_48_prediction"),
        },
    }


def compute_prediction_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred_similar = row[prediction_key] == "similar"
        true_similar = row["label"] == "similar"
        if pred_similar and true_similar:
            tp += 1
        elif pred_similar and not true_similar:
            fp += 1
        elif not pred_similar and true_similar:
            fn += 1
        else:
            tn += 1
    precision_similar = safe_div(tp, tp + fp)
    recall_similar = safe_div(tp, tp + fn)
    precision_different = safe_div(tn, tn + fn)
    recall_different = safe_div(tn, tn + fp)
    f1_similar = safe_div(2 * precision_similar * recall_similar, precision_similar + recall_similar)
    f1_different = safe_div(2 * precision_different * recall_different, precision_different + recall_different)
    return {
        "n": tp + fp + tn + fn,
        "accuracy": safe_div(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": (recall_similar + recall_different) / 2.0,
        "macro_f1": (f1_similar + f1_different) / 2.0,
        "precision_similar": precision_similar,
        "recall_similar": recall_similar,
        "precision_different": precision_different,
        "recall_different": recall_different,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def plot_value_distribution(rows: list[CleanRow], svg_path: Path, png_path: Path) -> None:
    values = [row.oral_bioavailability_value_percent for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=80, color="#2f6f73", edgecolor="white", linewidth=0.4)
    ax.set_title("Oral bioavailability value distribution")
    ax.set_xlabel("Oral bioavailability (%)")
    ax.set_ylabel("Cleaned row count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    value_summary = summary["value_percent_summary"]
    candidate = summary["pair_candidates"]
    eval_pairs = summary["eval_pairs"]
    lines = [
        "# Oral Bioavailability HF transfer dataset",
        "",
        f"- source dataset: `{summary['dataset']}` / `{summary['split']}`",
        f"- source rows: {summary['source_rows']:,}",
        f"- clean numeric rows: {summary['clean_rows']:,}",
        f"- dropped rows: {summary['dropped_rows']:,}",
        f"- aggregate molecules: {summary['aggregate_molecules']:,}",
        f"- candidate pairs: {candidate['n']:,}",
        f"- eval pairs: {eval_pairs['n']:,}",
        f"- eval label counts: {eval_pairs['label_counts']}",
        "",
        "## Value Distribution",
        "",
        f"- min / median / max: {value_summary.get('min', 0):.2f} / {value_summary.get('median', 0):.2f} / {value_summary.get('max', 0):.2f}",
        f"- mean / std: {value_summary.get('mean', 0):.2f} / {value_summary.get('std', 0):.2f}",
        "- figure: `figures/value_distribution.svg`",
        "",
        "## Parsing",
        "",
        f"- report type counts in source: {summary['report_type_counts_source']}",
        f"- drop reasons: {summary['drop_reasons']}",
        f"- parse methods: {summary['parse_method_counts']}",
        f"- parse modifiers: {summary['parse_modifier_counts']}",
        "",
        "## Baselines on Eval Pairs",
        "",
        "| baseline | accuracy | balanced accuracy | macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in eval_pairs["baseline_metrics"].items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_row_to_dict(row: CleanRow) -> dict[str, Any]:
    return {
        "source_index": row.source_index,
        "molecule_id": row.molecule_id,
        "molecule_name": row.molecule_name,
        "smiles": row.canonical_smiles,
        "oral_bioavailability_value": row.oral_bioavailability_value_percent,
        "oral_bioavailability_value_percent": row.oral_bioavailability_value_percent,
        "condition_text": row.condition_text,
        "condition_key": row.condition_key,
        "condition_key_hash": row.condition_key_hash,
        "parse_method": row.parse_method,
        "parse_modifier": row.parse_modifier,
        "metadata": row.raw_row,
    }


def clean_row_to_flat_dict(row: CleanRow) -> dict[str, Any]:
    out = clean_row_to_dict(row)
    out["metadata"] = json.dumps(out["metadata"], ensure_ascii=False, sort_keys=True)
    return out


def aggregate_to_dict(row: AggregateMolecule) -> dict[str, Any]:
    return {
        "aggregate_id": row.aggregate_id,
        "condition_key": row.condition_key,
        "condition_key_hash": row.condition_key_hash,
        "smiles": row.canonical_smiles,
        "molecule_names": row.molecule_names,
        "oral_bioavailability_value_percent": row.value_percent,
        "n_source_rows": row.n_source_rows,
        "source_indices": row.source_indices,
        "condition_text": row.condition_text,
        "metadata": row.raw_metadata,
    }


def flatten_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        flat = dict(row)
        for key in ("reference_source_indices", "query_source_indices"):
            flat[key] = ",".join(str(item) for item in flat.get(key, []))
        flattened.append(flat)
    return flattened


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def drop_record(source_index: int, row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"source_index": source_index, "drop_reason": reason, "raw_row": row}


def similarity_bucket(value: float) -> str:
    for name, lo, hi in SIMILARITY_BUCKETS:
        if lo <= value < hi:
            return name
    return "unknown"


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in NULL_STRINGS:
        return ""
    return re.sub(r"\s+", " ", text)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
