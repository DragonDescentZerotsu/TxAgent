"""Audit whether BBB-compatible analogs exist just below Morgan top-3.

The selector simulation is label-blind. Valid labels are used only after
selection for a descriptive KNN-vote diagnostic; no LLM calls are made.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.bbb_property_compatibility import (
    FEATURE_NAMES,
    ION_FEATURES,
    budget_key,
    decorate_candidates,
    ionization_class,
    morgan_top_n,
    robust_feature_scales,
    select_compatible_neighbors,
    selection_summary,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent.molecule_property_profiles import (
    load_or_fetch_profiles,
)
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)


SCHEMA_VERSION = "bbb_property_compatibility_availability.v1"
DEFAULT_DATA_DIR = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold"
)
DEFAULT_KNN_PREDICTIONS = Path(
    "outputs/baselines/structure_knn_experimental_meaningful_cns_access_v2/"
    "BBB_Martins/scaffold/valid_predictions.jsonl"
)
DEFAULT_MATCHED_BATCH = Path(
    "outputs/paper/matched_train_label_direct_agent_meaningful_cns_adjudication_v3_"
    "scaffold_valid_gpt_oss_120b_formal/runs_identity_blind_parent_disjoint/"
    "bbb_martins/bbb_martins__matched_train_label_direct"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/paper/bbb_property_compatibility_availability_"
    "experimental_meaningful_cns_access_v2_valid"
)
DEFAULT_SIMILARITY_BUDGETS = (0.02, 0.05, 0.10)


def run_audit(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    knn_predictions_path: Path = DEFAULT_KNN_PREDICTIONS,
    matched_batch: Path = DEFAULT_MATCHED_BATCH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tool_service_url: str = "http://127.0.0.1:8765",
    top_n: int = 20,
    similarity_budgets: tuple[float, ...] = DEFAULT_SIMILARITY_BUDGETS,
    batch_size: int = 64,
) -> dict[str, Any]:
    train_path, valid_path = data_dir / "train.jsonl", data_dir / "valid.jsonl"
    train, valid = _read_jsonl(train_path), _read_jsonl(valid_path)
    baseline = _read_jsonl(knn_predictions_path)
    if len(valid) != len(baseline):
        raise ValueError("Valid and KNN prediction counts differ")

    expected_profiles = [
        (f"{split}:{index}", str(row["drug"]))
        for split, rows in (("train", train), ("valid", valid))
        for index, row in enumerate(rows)
    ]
    property_path = output_dir / "molecule_properties.jsonl"
    profiles, profile_status_counts = load_or_fetch_profiles(
        expected_profiles,
        path=property_path,
        required_features=set(FEATURE_NAMES) | set(ION_FEATURES),
        tool_service_url=tool_service_url,
        batch_size=batch_size,
        fallback=_rdkit_fallback_features,
    )
    train_features = [profiles[f"train:{index}"] for index in range(len(train))]
    valid_features = [profiles[f"valid:{index}"] for index in range(len(valid))]
    scales = robust_feature_scales(train_features)
    transferability = _load_transferability(matched_batch, len(valid))
    candidate_sets = morgan_top_n(train, valid, top_n=top_n)
    query_rows, predictions_by_budget = [], {value: [] for value in similarity_budgets}

    for query_index, (gold, baseline_row, candidates) in enumerate(
        zip(valid, baseline, candidate_sets, strict=True)
    ):
        _validate_top3_parity(query_index, candidates, baseline_row)
        current = decorate_candidates(
            candidates[:3], valid_features[query_index], train_features, scales
        )
        row: dict[str, Any] = {
            "query_index": query_index,
            "Y": int(gold["Y"]),
            "transferability": transferability[query_index],
            "query_ionization_class": ionization_class(valid_features[query_index]),
            "current_top3": current,
            "current": selection_summary(current),
            "budgets": {},
        }
        cutoff = float(candidates[2]["similarity"])
        for budget in similarity_budgets:
            selected = select_compatible_neighbors(
                candidates,
                query_features=valid_features[query_index],
                train_features=train_features,
                scales=scales,
                top_k=3,
                similarity_floor=cutoff - budget,
            )
            candidate = selection_summary(selected)
            candidate.update(
                {
                    "similarity_budget": budget,
                    "similarity_floor": cutoff - budget,
                    "neighbor_set_changed": {item["train_index"] for item in selected}
                    != {item["train_index"] for item in current},
                    "mean_property_distance_delta": candidate["mean_property_distance"]
                    - row["current"]["mean_property_distance"],
                    "mean_similarity_delta": candidate["mean_similarity"]
                    - row["current"]["mean_similarity"],
                    "same_ionization_count_delta": candidate["same_ionization_count"]
                    - row["current"]["same_ionization_count"],
                }
            )
            row["budgets"][budget_key(budget)] = candidate
            predictions_by_budget[budget].append(candidate["prediction"])
        query_rows.append(row)

    labels = [int(row["Y"]) for row in valid]
    baseline_predictions = [int(row["prediction"]) for row in baseline]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_train": len(train),
        "n_valid": len(valid),
        "top_n": top_n,
        "selection_uses_labels": False,
        "selection_order": [
            "ionization_class_match",
            "train-IQR-standardized_property_distance",
            "Morgan_similarity",
        ],
        "features": list(FEATURE_NAMES),
        "feature_scales": scales,
        "property_profile_status_counts": profile_status_counts,
        "transferability_counts": dict(sorted(Counter(transferability).items())),
        "budgets": {
            budget_key(budget): _budget_summary(
                query_rows,
                budget,
                labels,
                baseline_predictions,
                predictions_by_budget[budget],
            )
            for budget in similarity_budgets
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256_file(valid_path)},
            "knn_predictions": {
                "path": str(knn_predictions_path),
                "sha256": sha256_file(knn_predictions_path),
            },
            "matched_v3_batch": str(matched_batch),
        },
        "tool": "molecule_properties.v1",
        "tool_service_url": tool_service_url,
        "selection_uses_labels": False,
        "top_n": top_n,
        "similarity_budgets": list(similarity_budgets),
    }
    write_jsonl_atomic(output_dir / "query_audit.jsonl", query_rows)
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "manifest.json", manifest)
    (output_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _budget_summary(
    rows: list[dict[str, Any]],
    budget: float,
    labels: list[int],
    baseline: list[int],
    candidate: list[int],
) -> dict[str, Any]:
    key = budget_key(budget)
    low = [index for index, row in enumerate(rows) if row["transferability"] == "low"]
    improvements = [-float(row["budgets"][key]["mean_property_distance_delta"]) for row in rows]
    costs = [-float(row["budgets"][key]["mean_similarity_delta"]) for row in rows]
    return {
        "n_neighbor_sets_changed": sum(row["budgets"][key]["neighbor_set_changed"] for row in rows),
        "n_prediction_flips": sum(left != right for left, right in zip(baseline, candidate, strict=True)),
        "mean_property_distance_improvement": statistics.fmean(improvements),
        "median_property_distance_improvement": statistics.median(improvements),
        "mean_similarity_cost": statistics.fmean(costs),
        "median_similarity_cost": statistics.median(costs),
        "mean_same_ionization_count_delta": statistics.fmean(
            float(row["budgets"][key]["same_ionization_count_delta"]) for row in rows
        ),
        "all_valid_vote_diagnostic": paired_binary_summary(labels, baseline, candidate),
        "low_transferability": {
            "n": len(low),
            "n_neighbor_sets_changed": sum(rows[i]["budgets"][key]["neighbor_set_changed"] for i in low),
            "mean_property_distance_improvement": statistics.fmean(improvements[i] for i in low),
            "mean_similarity_cost": statistics.fmean(costs[i] for i in low),
            "vote_diagnostic": paired_binary_summary(
                [labels[i] for i in low], [baseline[i] for i in low], [candidate[i] for i in low]
            ),
        },
    }


def _validate_top3_parity(
    query_index: int, candidates: list[dict[str, Any]], baseline: dict[str, Any]
) -> None:
    expected = [(int(row["train_index"]), float(row["similarity"])) for row in baseline.get("neighbors") or []]
    observed = [(int(row["train_index"]), float(row["similarity"])) for row in candidates[:3]]
    if len(expected) != 3 or any(
        left[0] != right[0] or abs(left[1] - right[1]) > 1e-12
        for left, right in zip(expected, observed, strict=True)
    ):
        raise ValueError(f"Morgan top-3 parity failure at query {query_index}")


def _load_transferability(batch: Path, n_queries: int) -> list[str]:
    values = []
    for index in range(n_queries):
        path = batch / "runs" / f"{batch.name}_idx{index:05d}" / "group_reasoning_outputs.jsonl"
        outputs = _read_jsonl(path)
        if len(outputs) != 1:
            raise ValueError(f"Expected one matched group output at query {index}")
        values.append(str((outputs[0].get("llm") or {}).get("content", {}).get("transferability")))
    return values


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rdkit_fallback_features(smiles: str) -> dict[str, float | None]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit fallback could not parse {smiles!r}")
    return {
        "pka__fraction_neutral": None,
        "pka__logd_estimate": None,
        "pka__num_acidic_sites": None,
        "pka__num_basic_sites": None,
        "rdkit__MolWt": float(Descriptors.MolWt(molecule)),
        "rdkit__TPSA": float(rdMolDescriptors.CalcTPSA(molecule)),
        "rdkit__NumHDonors": float(Lipinski.NumHDonors(molecule)),
        "rdkit__NumRotatableBonds": float(Lipinski.NumRotatableBonds(molecule)),
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# BBB property-compatibility availability audit",
        "",
        "Valid-only, no-LLM diagnostic. Neighbor selection does not use labels.",
        "",
        "| max similarity cost | changed | property gain | similarity cost | ion-match delta | vote F1 delta | low-transfer vote delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["budgets"].items():
        paired, low = row["all_valid_vote_diagnostic"], row["low_transferability"]["vote_diagnostic"]
        lines.append(
            f"| {key.rsplit('_', 1)[-1]} | {row['n_neighbor_sets_changed']}/{summary['n_valid']} "
            f"| {row['mean_property_distance_improvement']:.3f} | {row['mean_similarity_cost']:.3f} "
            f"| {row['mean_same_ionization_count_delta']:+.2f} | {paired['delta_macro_f1']:+.4f} "
            f"| {low['delta_macro_f1']:+.4f} |"
        )
    return "\n".join(lines + ["", "Vote metrics are descriptive, not agent estimates.", ""])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--knn-predictions", type=Path, default=DEFAULT_KNN_PREDICTIONS)
    parser.add_argument("--matched-batch", type=Path, default=DEFAULT_MATCHED_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_audit(
        data_dir=args.data_dir,
        knn_predictions_path=args.knn_predictions,
        matched_batch=args.matched_batch,
        output_dir=args.output_dir,
        tool_service_url=args.tool_service_url,
        top_n=args.top_n,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
