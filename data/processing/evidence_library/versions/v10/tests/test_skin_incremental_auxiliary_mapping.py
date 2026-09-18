from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing import (
    build_embedding_bucket_mapping as builder,
)


def test_selected_cluster_split_preserves_exact_membership():
    selected = builder.Cluster(
        cluster_id="cluster_selected",
        values=("a", "b", "c", "d", "e"),
    )
    untouched = builder.Cluster(cluster_id="cluster_untouched", values=("z",))

    result = builder._split_selected_clusters(
        [selected, untouched], {"cluster_selected": 2}
    )

    assert sorted(value for cluster in result for value in cluster.values) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "z",
    ]
    assert sorted(len(cluster.values) for cluster in result) == [1, 1, 2, 2]
    assert untouched in result


def test_selected_cluster_split_can_refine_one_child_and_ignore_other_extractions():
    selected = builder.Cluster(
        cluster_id="cluster_selected",
        values=("a", "b", "c", "d", "e"),
    )
    first_pass = builder._split_selected_clusters(
        [selected], {"cluster_selected": 3}
    )
    child = next(cluster for cluster in first_pass if len(cluster.values) == 3)

    result = builder._split_selected_clusters(
        [selected],
        {"cluster_selected": 3, child.cluster_id: 1},
    )

    assert sorted(value for cluster in result for value in cluster.values) == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]
    assert sorted(len(cluster.values) for cluster in result) == [1, 1, 1, 2]
    assert builder._split_selected_clusters(
        [builder.Cluster(cluster_id="cluster_other", values=("z",))],
        {"cluster_selected": 3},
    ) == [builder.Cluster(cluster_id="cluster_other", values=("z",))]


def test_unknown_label_gate_does_not_reject_sodium_symbol():
    assert (
        builder._clean_bucket(
            "Na+/K+-ATPase activity assay",
            null_sentinel=None,
            bucket_pattern=None,
        )
        == "na+/k+-atpase activity assay"
    )
    for label in ("NA", "N/A assay", "unknown assay"):
        try:
            builder._clean_bucket(
                label,
                null_sentinel=None,
                bucket_pattern=None,
            )
        except ValueError as exc:
            assert "unknown-like" in str(exc)
        else:
            raise AssertionError(f"expected {label!r} to be rejected")


def test_incremental_inventory_filters_source_and_excludes_reviewed_tuples(tmp_path):
    records = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "source_id": "sensitization_aop",
                "assay_type": "LLNA",
                "experimental_conditions": "mouse",
                "support_text": "existing",
            },
            {
                "source_id": "sensitization_aop",
                "assay_type": "DPRA",
                "experimental_conditions": "in vitro",
                "support_text": "new",
            },
            {
                "source_id": "direct_skin_reaction",
                "assay_type": "ignored",
                "experimental_conditions": "ignored",
                "support_text": "ignored",
            },
        ]
    ).to_parquet(records, index=False)
    existing = builder._tuple_key(("LLNA", "mouse", "existing"))
    base = {
        "sources": {
            "sensitization_aop": {
                "global_species_context": {
                    "source_columns": [
                        "assay_type",
                        "experimental_conditions",
                        "support_text",
                    ],
                    "mapping": {existing: "mouse"},
                }
            }
        }
    }

    inventory = builder._load_incremental_inventory(
        records,
        source_id="sensitization_aop",
        output_names={"global_species_context"},
        base_mapping=base,
    )

    assert inventory == {
        "global_species_context": [
            builder._tuple_key(("DPRA", "in vitro", "new"))
        ]
    }


def test_incremental_candidate_is_delta_only_and_does_not_modify_base(tmp_path):
    base_path = tmp_path / "reviewed.json"
    base_bytes = b'{"mapping_version":"reviewed","sources":{"keep":"unchanged"}}\n'
    base_path.write_bytes(base_bytes)
    records = tmp_path / "records.parquet"
    pd.DataFrame({"source_id": ["direct_skin_reaction"]}).to_parquet(
        records, index=False
    )
    args = SimpleNamespace(
        base_mapping=str(base_path),
        incremental_records=str(records),
        model="deepseek/deepseek-v4-flash-0731",
        reasoning_effort="high",
        embedding_model=builder.EMBEDDING_MODEL,
    )
    first_stage = {
        "direct_skin_reaction": {
            "assay_or_test": {"canonical_context": {"new assay": "patch test"}}
        }
    }
    ledger = builder.TokenLedger(
        tmp_path / "ledger.json",
        model=args.model,
        reasoning_effort="high",
    )
    budget = builder.TokenBudget(limit=100, ledger=ledger)

    manifest = builder._publish_incremental_candidate(
        args=args,
        first_stage=first_stage,
        inventories={"direct_skin_reaction": {"global_context": ["new assay"]}},
        selected_outputs={"direct_skin_reaction": {"global_context"}},
        budget=budget,
        work_dir=tmp_path,
    )

    candidate = json.loads(
        (tmp_path / "incremental_delta_candidate.json").read_text(encoding="utf-8")
    )
    assert base_path.read_bytes() == base_bytes
    assert candidate["publication_status"] == "unpublished_requires_agent_reconciliation"
    assert candidate["sources"]["direct_skin_reaction"]["global_context"][
        "mapping"
    ] == {'["new assay"]': "patch test"}
    assert manifest["new_tuple_counts"] == {
        "direct_skin_reaction/global_context": 1
    }
