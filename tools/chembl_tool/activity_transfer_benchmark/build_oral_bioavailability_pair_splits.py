"""Build large HF prompt/completion splits for oral-bioavailability transfer."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from tools.chembl_tool.activity_transfer_benchmark.build_oral_bioavailability_transfer_dataset import (
    LABEL_TO_COMPLETION,
    similarity_bucket,
)


DEFAULT_SOURCE_DIR = (
    "outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf/"
    "absolute_unspecified_systemic_broad_condition_full_text_v1"
)
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits"
DEFAULT_RUN_ID = "absolute_unspecified_systemic_directed_max_7_1_2_v1"
SPLITS = ("train", "validation", "test")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(args.seed)

    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregates_path = Path(args.source_dir) / "aggregate_molecules.jsonl"
    aggregates = load_aggregates(aggregates_path)
    groups = group_aggregates(aggregates)
    ratios = parse_ratios(args.ratios)
    molecule_to_split = build_molecule_split_map(aggregates, ratios, args.seed) if args.split_mode == "molecule_disjoint" else {}
    max_counts = count_possible_pairs(groups, args, molecule_to_split=molecule_to_split)
    target_pairs = min(args.target_pairs, max_counts["directed_nonambiguous"]) if args.target_pairs > 0 else max_counts[
        "directed_nonambiguous"
    ]

    split_paths = {split: out_dir / f"{split}.jsonl" for split in SPLITS}
    counters = write_pair_splits(
        groups=groups,
        split_paths=split_paths,
        target_pairs=target_pairs,
        ratios=ratios,
        split_mode=args.split_mode,
        molecule_to_split=molecule_to_split,
        seed=args.seed,
        similar_delta=args.similar_delta,
        different_delta=args.different_delta,
        label_mode=args.label_mode,
        min_ordered_delta=args.min_ordered_delta,
        progress_every=args.progress_every,
        rng=rng,
    )
    summary = {
        "run_id": args.run_id,
        "created_at_unix": time.time(),
        "elapsed_s": round(time.time() - started, 3),
        "source_dir": args.source_dir,
        "aggregate_molecules": len(aggregates),
        "condition_groups": len(groups),
        "unique_molecules": len({row["smiles"] for row in aggregates}),
        "parameters": vars(args),
        "molecule_split_counts": dict(Counter(molecule_to_split.values())) if molecule_to_split else {},
        "max_possible": max_counts,
        "target_pairs": target_pairs,
        "written": counters,
        "files": {split: str(path) for split, path in split_paths.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir / "report_zh.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--target-pairs", type=int, default=30_000_000, help="0 means write every possible pair.")
    parser.add_argument("--ratios", default="0.7,0.1,0.2")
    parser.add_argument(
        "--split-mode",
        choices=("unordered_pair_random", "molecule_disjoint"),
        default="unordered_pair_random",
        help="molecule_disjoint assigns each canonical SMILES to exactly one split and only writes split-internal pairs.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("percent_delta", "higher_lower"),
        default="percent_delta",
        help="percent_delta predicts transfer/non-transfer; higher_lower predicts whether Molecule B has higher or lower F%.",
    )
    parser.add_argument("--similar-delta", type=float, default=10.0)
    parser.add_argument("--different-delta", type=float, default=30.0)
    parser.add_argument(
        "--min-ordered-delta",
        type=float,
        default=0.0,
        help="For higher_lower mode, exclude pairs with |delta F%| <= this margin.",
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser.parse_args(argv)


def parse_ratios(raw: str) -> dict[str, float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--ratios must contain train,validation,test values.")
    total = sum(values)
    if total <= 0:
        raise ValueError("--ratios must sum to a positive value.")
    return {split: value / total for split, value in zip(SPLITS, values)}


def load_aggregates(path: Path) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    aggregates = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            row["fp"] = generator.GetFingerprint(mol)
            aggregates.append(row)
    return aggregates


def group_aggregates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition_key"])].append(row)
    return groups


def build_molecule_split_map(rows: list[dict[str, Any]], ratios: dict[str, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    molecules = sorted({str(row["smiles"]) for row in rows})
    rng.shuffle(molecules)
    n_total = len(molecules)
    n_train = int(round(n_total * ratios["train"]))
    n_validation = int(round(n_total * ratios["validation"]))
    out = {}
    for index, smiles in enumerate(molecules):
        if index < n_train:
            split = "train"
        elif index < n_train + n_validation:
            split = "validation"
        else:
            split = "test"
        out[smiles] = split
    return out


def count_possible_pairs(
    groups: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    *,
    molecule_to_split: dict[str, str],
) -> dict[str, Any]:
    unordered_total = unordered_nonambiguous = similar = different = ambiguous = 0
    split_counts: Counter = Counter()
    label_split_counts: Counter = Counter()
    cross_split_nonambiguous = 0
    for rows in groups.values():
        n = len(rows)
        unordered_total += n * (n - 1) // 2
        values = [float(row["oral_bioavailability_value_percent"]) for row in rows]
        for i, row_a in enumerate(rows):
            value_a = values[i]
            split_a = molecule_to_split.get(str(row_a["smiles"]))
            for j, row_b in enumerate(rows[i + 1 :], start=i + 1):
                value_b = values[j]
                split_b = molecule_to_split.get(str(row_b["smiles"]))
                labels = directed_labels_for_pair(
                    value_a,
                    value_b,
                    label_mode=args.label_mode,
                    similar_delta=args.similar_delta,
                    different_delta=args.different_delta,
                    min_ordered_delta=args.min_ordered_delta,
                )
                if not labels:
                    ambiguous += 1
                    continue
                unordered_nonambiguous += 1
                if args.label_mode == "percent_delta":
                    if labels[0] == "similar":
                        similar += 1
                    else:
                        different += 1
                else:
                    similar += int(labels[0] == "query_higher") + int(labels[1] == "query_higher")
                    different += int(labels[0] == "query_lower") + int(labels[1] == "query_lower")
                if molecule_to_split:
                    if split_a and split_a == split_b:
                        split_counts[split_a] += 2
                        for label in labels:
                            label_split_counts[f"{split_a}|{label}"] += 1
                    else:
                        cross_split_nonambiguous += 1
                else:
                    split_counts["all"] += 2
                    for label in labels:
                        label_split_counts[f"all|{label}"] += 1
    return {
        "unordered_total": unordered_total,
        "unordered_nonambiguous": unordered_nonambiguous,
        "directed_nonambiguous": sum(split_counts.values()) if molecule_to_split else unordered_nonambiguous * 2,
        "directed_nonambiguous_by_split": dict(split_counts),
        "directed_label_counts_by_split": dict(label_split_counts),
        "cross_split_unordered_nonambiguous_excluded": cross_split_nonambiguous,
        "unordered_similar_or_query_higher": similar,
        "unordered_different_or_query_lower": different,
        "unordered_ambiguous": ambiguous,
    }


def write_pair_splits(
    *,
    groups: dict[str, list[dict[str, Any]]],
    split_paths: dict[str, Path],
    target_pairs: int,
    ratios: dict[str, float],
    split_mode: str,
    molecule_to_split: dict[str, str],
    seed: int,
    similar_delta: float,
    different_delta: float,
    label_mode: str,
    min_ordered_delta: float,
    progress_every: int,
    rng: random.Random,
) -> dict[str, Any]:
    for path in split_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = {split: path.open("w", encoding="utf-8") for split, path in split_paths.items()}
    split_counts: Counter = Counter()
    label_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    started = time.time()
    written = 0
    pair_index = 0
    unordered_pair_index = 0
    try:
        group_items = list(groups.items())
        rng.shuffle(group_items)
        for condition_index, (condition_key, rows) in enumerate(group_items):
            if len(rows) < 2:
                continue
            order = list(range(len(rows)))
            rng.shuffle(order)
            for left_pos, i in enumerate(order):
                row_a = rows[i]
                for j in order[left_pos + 1 :]:
                    row_b = rows[j]
                    labels = directed_labels_for_pair(
                        float(row_a["oral_bioavailability_value_percent"]),
                        float(row_b["oral_bioavailability_value_percent"]),
                        label_mode=label_mode,
                        similar_delta=similar_delta,
                        different_delta=different_delta,
                        min_ordered_delta=min_ordered_delta,
                    )
                    if not labels:
                        continue
                    if split_mode == "molecule_disjoint":
                        split_a = molecule_to_split.get(str(row_a["smiles"]))
                        split_b = molecule_to_split.get(str(row_b["smiles"]))
                        if not split_a or split_a != split_b:
                            unordered_pair_index += 1
                            continue
                        split = split_a
                    else:
                        split = assign_split(unordered_pair_index, ratios, seed)
                    for (reference, query), label in zip(((row_a, row_b), (row_b, row_a)), labels):
                        record = build_hf_record(
                            pair_index=pair_index,
                            unordered_pair_index=unordered_pair_index,
                            condition_index=condition_index,
                            condition_key=condition_key,
                            reference=reference,
                            query=query,
                            label=label,
                            label_mode=label_mode,
                            similar_delta=similar_delta,
                            different_delta=different_delta,
                            min_ordered_delta=min_ordered_delta,
                        )
                        handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        split_counts[split] += 1
                        label_counts[f"{split}|{label}"] += 1
                        bucket_counts[f"{split}|{record['metadata']['similarity_bucket']}"] += 1
                        written += 1
                        pair_index += 1
                        if progress_every > 0 and written % progress_every == 0:
                            elapsed = max(time.time() - started, 1e-9)
                            rate = written / elapsed
                            remaining = max(target_pairs - written, 0)
                            print(
                                json.dumps(
                                    {
                                        "stage": "write_oral_pair_splits",
                                        "written": written,
                                        "target_pairs": target_pairs,
                                        "split_counts": dict(split_counts),
                                        "rate_per_s": round(rate, 2),
                                        "elapsed_s": round(elapsed, 1),
                                        "eta_s": round(remaining / rate, 1) if rate > 0 else None,
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        if written >= target_pairs:
                            return {
                                "total": written,
                                "split_counts": dict(split_counts),
                                "label_counts": dict(label_counts),
                                "similarity_bucket_counts": dict(bucket_counts),
                            }
                    unordered_pair_index += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "total": written,
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "similarity_bucket_counts": dict(bucket_counts),
    }


def directed_labels_for_pair(
    value_a: float,
    value_b: float,
    *,
    label_mode: str,
    similar_delta: float,
    different_delta: float,
    min_ordered_delta: float,
) -> tuple[str, str] | None:
    if label_mode == "higher_lower":
        delta_ab = value_b - value_a
        if abs(delta_ab) <= min_ordered_delta:
            return None
        first = "query_higher" if delta_ab > 0 else "query_lower"
        second = "query_lower" if delta_ab > 0 else "query_higher"
        return first, second

    delta = abs(value_a - value_b)
    if delta <= similar_delta:
        return "similar", "similar"
    if delta >= different_delta:
        return "different", "different"
    return None


def assign_split(pair_index: int, ratios: dict[str, float], seed: int) -> str:
    del seed
    bucket = pair_index % 10
    train_cutoff = int(round(ratios["train"] * 10))
    validation_cutoff = train_cutoff + int(round(ratios["validation"] * 10))
    if bucket < train_cutoff:
        return "train"
    if bucket < validation_cutoff:
        return "validation"
    return "test"


def build_hf_record(
    *,
    pair_index: int,
    unordered_pair_index: int,
    condition_index: int,
    condition_key: str,
    reference: dict[str, Any],
    query: dict[str, Any],
    label: str,
    label_mode: str,
    similar_delta: float,
    different_delta: float,
    min_ordered_delta: float,
) -> dict[str, Any]:
    tanimoto = float(DataStructs.TanimotoSimilarity(reference["fp"], query["fp"]))
    reference_value = float(reference["oral_bioavailability_value_percent"])
    query_value = float(query["oral_bioavailability_value_percent"])
    abs_delta = abs(reference_value - query_value)
    endpoint_id = f"oral_bioavailability:{reference['condition_key_hash']}"
    completion = LABEL_TO_COMPLETION[label] if label_mode == "percent_delta" else ("A" if label == "query_higher" else "B")
    metadata = {
        "sample_id": pair_index,
        "pair_id": pair_index,
        "unordered_pair_id": unordered_pair_index,
        "source": "starling-labs/Oral_Bioavailability",
        "task_name": "oral_bioavailability_hf",
        "label_mode": f"oral_bioavailability_{label_mode}",
        "semantic_label": label,
        "completion_a_label": "similar" if label_mode == "percent_delta" else "query_higher",
        "completion_b_label": "different" if label_mode == "percent_delta" else "query_lower",
        "endpoint_id": endpoint_id,
        "assay_type": "oral_bioavailability_literature",
        "condition_key": condition_key,
        "condition_key_hash": reference["condition_key_hash"],
        "condition_index": condition_index,
        "similarity_bucket": similarity_bucket(tanimoto),
        "weighted_tanimoto": round(tanimoto, 6),
        "abs_activity_delta": round(abs_delta, 6),
        "similar_delta_threshold": similar_delta,
        "different_delta_threshold": different_delta,
        "min_ordered_delta_threshold": min_ordered_delta,
        "reference_activity_value": round(reference_value, 6),
        "hidden_query_activity_value": round(query_value, 6),
        "reference_aggregate_id": reference["aggregate_id"],
        "query_aggregate_id": query["aggregate_id"],
        "reference_source_indices": reference.get("source_indices", []),
        "query_source_indices": query.get("source_indices", []),
        "direction": "directed_reference_to_query",
        "eval_subset": f"oral_bioavailability_hf_{label_mode}",
    }
    return {
        "prompt": build_prompt(reference, query, reference_value, label_mode, similar_delta, different_delta, min_ordered_delta),
        "completion": completion,
        "metadata": metadata,
    }


def build_prompt(
    reference: dict[str, Any],
    query: dict[str, Any],
    reference_value: float,
    label_mode: str,
    similar_delta: float,
    different_delta: float,
    min_ordered_delta: float,
) -> str:
    if label_mode == "higher_lower":
        margin_line = (
            f"- Equal/tie exclusion margin: absolute difference <= {min_ordered_delta:.1f} percentage points\n"
            if min_ordered_delta > 0
            else "- Equal/tie exclusion: exactly equal values are excluded from the dataset\n"
        )
        return (
            "You are given two molecules and an oral bioavailability endpoint. Predict whether Molecule B has "
            "higher or lower oral bioavailability than Molecule A under the same broad experimental condition.\n\n"
            "## Endpoint\n"
            "- Measurement type: oral_bioavailability_percent\n"
            "- Endpoint description: absolute oral bioavailability or explicitly numeric oral/systemic availability\n"
            + margin_line
            + "- Reference experimental context:\n"
            + indent_block(str(reference.get("condition_text") or "not specified"))
            + "\n"
            "- Query experimental context:\n"
            + indent_block(str(query.get("condition_text") or "not specified"))
            + "\n\n## Molecule A\n"
            + f"- SMILES: {reference['smiles']}\n"
            + f"- Known oral bioavailability: {reference_value:.2f}%\n\n"
            + "## Molecule B\n"
            + f"- SMILES: {query['smiles']}\n"
            + "- Oral bioavailability: hidden for evaluation\n\n"
            + "Compared with Molecule A, is Molecule B expected to have higher or lower oral bioavailability?\n"
            + "(A) higher\n"
            + "(B) lower\n\n"
            + "Answer:"
        )
    return (
        "You are given two molecules and an oral bioavailability endpoint. Decide whether the endpoint behavior "
        "should transfer between the molecules.\n\n"
        "## Endpoint\n"
        "- Measurement type: oral_bioavailability_percent\n"
        "- Endpoint description: absolute oral bioavailability or explicitly numeric oral/systemic availability\n"
        "- Similar label threshold: absolute difference <= "
        f"{similar_delta:.1f} percentage points\n"
        "- Different label threshold: absolute difference >= "
        f"{different_delta:.1f} percentage points\n"
        "- Reference experimental context:\n"
        + indent_block(str(reference.get("condition_text") or "not specified"))
        + "\n"
        "- Query experimental context:\n"
        + indent_block(str(query.get("condition_text") or "not specified"))
        + "\n\n## Molecule A\n"
        + f"- SMILES: {reference['smiles']}\n"
        + f"- Known oral bioavailability: {reference_value:.2f}%\n\n"
        + "## Molecule B\n"
        + f"- SMILES: {query['smiles']}\n"
        + "- Oral bioavailability: hidden for evaluation\n\n"
        + "Should this oral bioavailability endpoint transfer between Molecule A and Molecule B?\n"
        + "(A) transfer\n"
        + "(B) not transfer\n\n"
        + "Answer:"
    )


def indent_block(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Oral bioavailability pair splits",
        "",
        f"- source: `{summary['source_dir']}`",
        f"- aggregate molecules: {summary['aggregate_molecules']:,}",
        f"- condition groups: {summary['condition_groups']:,}",
        f"- max directed non-ambiguous pairs: {summary['max_possible']['directed_nonambiguous']:,}",
        f"- written pairs: {summary['written']['total']:,}",
        f"- split counts: {summary['written']['split_counts']}",
        f"- label counts: {summary['written']['label_counts']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
