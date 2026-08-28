"""Build the versioned experimental meaningful-CNS-access BBB benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.chembl_tool.common.starling.benchmark_dataset import build_benchmark_dataset
from tools.chembl_tool.common.starling.build_record_supported_benchmark import (
    build_task,
    build_task_preserving_split,
)
from tools.chembl_tool.common.json_utils import atomic_output_path, write_json_atomic
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    CONTRACT_VERSION,
    SOURCE_REVISION,
    load_label_decisions,
)


LINEAGE = "experimental_meaningful_cns_access_v2"
PROTOCOL_VERSION = "starling_experimental_meaningful_cns_access_benchmark.v2"
DEFAULT_OUTPUT_ROOT = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v2"
)
SOURCE_ARTIFACT_NAMES = (
    "molecule_labels.jsonl",
    "conflicting_molecules.jsonl",
    "rejected_parent_molecules.jsonl",
    "source_rejection_examples.jsonl",
)


@dataclass(frozen=True)
class BBBGoldBuildSpec:
    lineage: str
    protocol_version: str
    contract_version: str
    default_output_root: Path
    preserve_split_root: Path | None = None
    allow_new_parents_in_preserved_split: bool = False


V2_BUILD_SPEC = BBBGoldBuildSpec(
    lineage=LINEAGE,
    protocol_version=PROTOCOL_VERSION,
    contract_version=CONTRACT_VERSION,
    default_output_root=DEFAULT_OUTPUT_ROOT,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv, default_output_root=V2_BUILD_SPEC.default_output_root)
    _validate_build_scope(args, V2_BUILD_SPEC)
    decisions, metadata = load_label_decisions(
        revision=args.bbb_revision,
        max_rows=args.max_rows,
    )
    return run_build(args, decisions, metadata, spec=V2_BUILD_SPEC)


def _validate_build_scope(
    args: argparse.Namespace,
    spec: BBBGoldBuildSpec = V2_BUILD_SPEC,
) -> None:
    if args.bbb_revision != SOURCE_REVISION:
        raise ValueError(
            "The manual source exclusions are bound to the frozen BBB source revision; "
            f"expected {SOURCE_REVISION}."
        )
    if args.max_rows and Path(args.output_root) == spec.default_output_root:
        raise ValueError(
            "A partial --max-rows build requires an explicit non-canonical --output-root."
        )


def run_build(
    args: argparse.Namespace,
    decisions,
    metadata,
    *,
    spec: BBBGoldBuildSpec,
) -> int:
    metadata = {**metadata, "benchmark_lineage": spec.lineage}
    output_root = Path(args.output_root)
    with TemporaryDirectory(prefix="bbb_meaningful_cns_access_build_") as temporary:
        temporary_root = Path(temporary)
        source_task_root = temporary_root / "BBB_Martins"
        build_benchmark_dataset(
            task_name="BBB_Martins",
            decisions=decisions,
            source_metadata=metadata,
            output_dir=source_task_root,
            max_eval_size=args.max_eval_size,
            valid_fraction=args.valid_fraction,
            test_fraction=args.test_fraction,
            agreement_threshold=args.agreement_threshold,
            seed=args.seed,
            max_rejection_examples=args.max_rejection_examples,
        )
        task_root = output_root / "BBB_Martins"
        task_root.mkdir(parents=True, exist_ok=True)
        for artifact_name in SOURCE_ARTIFACT_NAMES:
            source_path = source_task_root / artifact_name
            with atomic_output_path(task_root / artifact_name) as temporary_path:
                temporary_path.write_bytes(source_path.read_bytes())
        build_kwargs = {
            "source_root": temporary_root,
            "output_root": output_root,
            "lineage": spec.lineage,
            "protocol_version": spec.protocol_version,
            "seed": args.seed,
        }
        summary = (
            build_task_preserving_split(
                "BBB_Martins",
                reference_split_root=spec.preserve_split_root,
                allow_new_parents=spec.allow_new_parents_in_preserved_split,
                **build_kwargs,
            )
            if spec.preserve_split_root is not None
            else build_task("BBB_Martins", **build_kwargs)
        )

    summary["paths"].update(
        {
            name.removesuffix(".jsonl"): str(output_root / "BBB_Martins" / name)
            for name in SOURCE_ARTIFACT_NAMES
        }
    )
    summary.pop("cross_method_eval_identity_overlap", None)
    summary["record_support_policy"].update(
        {
            "target_valid_size": summary["split_size_policy"]["target_valid_size"],
            "target_test_size": summary["split_size_policy"]["target_test_size"],
        }
    )
    summary["gold_distribution"] = _gold_distribution(output_root / "BBB_Martins")
    summary["artifact_sha256"] = _artifact_hashes(output_root / "BBB_Martins")
    summary["build_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "contract": spec.contract_version,
                "lineage": spec.lineage,
                "artifacts": summary["artifact_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_json_atomic(output_root / "BBB_Martins" / "summary.json", summary)
    with atomic_output_path(output_root / "BBB_Martins" / "report_zh.md") as temporary:
        temporary.write_text(_render_report(summary, spec), encoding="utf-8")
    root_summary = {
        "lineage": spec.lineage,
        "gold_contract_version": spec.contract_version,
        "tasks": {"bbb_martins": summary},
    }
    write_json_atomic(output_root / "summary.json", root_summary)
    print(json.dumps(root_summary, ensure_ascii=False, indent=2, default=str))
    return 0


def _render_report(summary: dict, spec: BBBGoldBuildSpec) -> str:
    split = summary["splits"]["scaffold"]
    rejection_counts = summary["source_rejection_counts"]
    return "\n".join(
        [
            "# BBB experimental meaningful-CNS-access gold build",
            "",
            f"- lineage: `{spec.lineage}`",
            f"- gold contract: `{spec.contract_version}`",
            f"- frozen source rows: {summary['n_source_rows_considered']:,}",
            f"- accepted experimental source rows: "
            f"{summary['n_source_rows_labeled_before_structure_normalization']:,}",
            f"- binary parent molecules: {summary['n_binary_molecules']:,}",
            f"- parent labels Y=0/Y=1: "
            f"{summary['all_label_counts'].get('0', 0):,}/"
            f"{summary['all_label_counts'].get('1', 0):,}",
            f"- scaffold train/valid/test: "
            f"{split['n_train']:,}/{split['n_valid']:,}/{split['n_test']:,}",
            f"- valid/test singleton parents: "
            f"{split['valid_record_tier_counts'].get('singleton', 0):,}/"
            f"{split['test_record_tier_counts'].get('singleton', 0):,}",
            "- parent identity overlap and scaffold overlap: 0 by construction",
            "",
            "## Gold scope",
            "",
            "Gold contains experimentally observed brain, unbound-brain, brain/systemic-ratio, "
            "CSF, PET/autoradiography, or explicit in-vivo BBB outcomes after systemic "
            "administration. The target is meaningful or adequate CNS access versus "
            "restricted or poor access, independent of entry mechanism. Passive/in-vitro "
            "permeability and transporter results remain mechanism evidence and do not vote.",
            "",
            "Low but nonzero exposure can remain negative when the experiment supports "
            "restricted or poor access; mere detectability is not the positive-label rule.",
            "No endpoint or mechanism quota is used to balance the gold. Mechanism-family "
            "coverage is a separate downstream retrieval-index audit.",
            "",
            "CSF is retained as an explicitly tagged proxy outcome; it is not represented as a "
            "brain-parenchyma measurement.",
            "",
            "## Included endpoint-family distribution",
            "",
            "| family | accepted records | parent molecules | Y=0 parents | Y=1 parents |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {family} | {values['accepted_records']:,} | "
                f"{values['parent_molecules']:,} | {values['y0_parents']:,} | "
                f"{values['y1_parents']:,} |"
                for family, values in summary["gold_distribution"][
                    "endpoint_families"
                ].items()
            ],
            "",
            "## Major source-row exclusions",
            "",
            *[
                f"- `{reason}`: {count:,}"
                for reason, count in sorted(
                    rejection_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "",
        ]
    )


def _gold_distribution(task_root: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (task_root / "molecule_labels.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    accepted_records: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    parent_labels: Counter[tuple[str, int]] = Counter()
    record_tiers: Counter[str] = Counter()
    for row in rows:
        record_tiers[
            "singleton" if int(row["source_record_count"]) == 1 else "multi_record"
        ] += 1
        families: set[str] = set()
        for method, count in row["label_methods"].items():
            family = method.split(":", maxsplit=2)[1]
            accepted_records[family] += int(count)
            families.add(family)
        for family in families:
            parent_counts[family] += 1
            parent_labels[(family, int(row["Y"]))] += 1
    return {
        "record_tier_counts": dict(sorted(record_tiers.items())),
        "endpoint_families": {
            family: {
                "accepted_records": accepted_records[family],
                "parent_molecules": parent_counts[family],
                "y0_parents": parent_labels[(family, 0)],
                "y1_parents": parent_labels[(family, 1)],
            }
            for family in sorted(
                accepted_records,
                key=lambda name: (-accepted_records[name], name),
            )
        },
    }


def _artifact_hashes(task_root: Path) -> dict[str, str]:
    relative_paths = [
        Path(name) for name in SOURCE_ARTIFACT_NAMES
    ] + [
        Path("scaffold") / name
        for name in (
            "train.jsonl",
            "valid.jsonl",
            "test.jsonl",
            "train_molecule_labels.jsonl",
            "valid_molecule_labels.jsonl",
            "test_molecule_labels.jsonl",
            "heldout_molecule_labels.jsonl",
        )
    ]
    return {
        str(path): hashlib.sha256((task_root / path).read_bytes()).hexdigest()
        for path in relative_paths
    }


def _parse_args(
    argv: list[str] | None,
    *,
    default_output_root: Path = DEFAULT_OUTPUT_ROOT,
    allow_source_arrow: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(default_output_root))
    parser.add_argument("--bbb-revision", default=SOURCE_REVISION)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-eval-size", type=int, default=500)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--agreement-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--max-rejection-examples", type=int, default=20)
    if allow_source_arrow:
        parser.add_argument(
            "--source-arrow",
            default="",
            help="Offline copy of the pinned Hugging Face Arrow split.",
        )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
