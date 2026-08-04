import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.chembl_tool.tasks.bbb_martins.data_processing import (
    auxiliary_reconciliation,
)
from tools.chembl_tool.common.starling.clustered_auxiliary_mapping import (
    DEFAULT_CLUSTER_RANDOM_SEED,
    DEFAULT_CLUSTER_TARGET_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    Cluster,
    _cache_identity,
    _cluster_cache_identity,
)
from tools.chembl_tool.tasks.bbb_martins.data_processing.auxiliary_reconciliation import (
    reconstruct_provisional,
    write_provisional_artifacts,
)
from tools.chembl_tool.tasks.bbb_martins.data_processing.build_embedding_bucket_mapping import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    PROMPT_VERSION,
    extraction_specs,
)
from tools.chembl_tool.tasks.bbb_martins.starling_auxiliary_metadata import (
    MAPPING_VERSION,
)


ARTIFACT_NAMES = (
    "provisional_cluster_mapping.json",
    "cluster_assignments.parquet",
    "pre_reconciliation_mapping.json",
    "provisional_manifest.json",
)


@pytest.fixture(autouse=True)
def _synthetic_expected_counts(monkeypatch):
    """Keep production freeze gates active while permitting small fixtures."""
    monkeypatch.setattr(auxiliary_reconciliation, "EXPECTED_SECTION_CLUSTERS", 7)
    monkeypatch.setattr(auxiliary_reconciliation, "EXPECTED_NON_NULL_ASSIGNMENTS", 8)
    monkeypatch.setattr(
        auxiliary_reconciliation, "EXPECTED_CURRENT_PUBLISHED_DELTAS", 3
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _usage() -> dict[str, int]:
    return {
        "cached_input_tokens": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "total_tokens": 15,
    }


def _generation(source_id: str, output_field: str, cluster_id: str) -> dict:
    response_sha256 = hashlib.sha256(
        f"{source_id}/{output_field}/{cluster_id}".encode()
    ).hexdigest()
    usage = _usage()
    return {
        "attempt_count": 1,
        "attempts": [
            {
                "attempt": 1,
                "response_sha256": response_sha256,
                "served_model": "gpt-5.4-mini-2026-03-17",
                "status": "valid",
                "usage": usage,
            }
        ],
        "usage": usage,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    inventories = {
        ("direct_bbb", "assay_model"): (
            Cluster("cluster_direct_a", ("GdET1WI", "PAMPA-BBB")),
            Cluster("cluster_direct_b", ("GdET₁WI",)),
        ),
        ("direct_bbb", "species"): (
            Cluster("cluster_direct_species", ("Rattus norvegicus",)),
        ),
        ("passive_permeability", "biological_system"): (
            Cluster("cluster_passive", ("PAMPA-BBB",)),
        ),
        ("efflux_transport", "assay_system"): (
            Cluster(
                "cluster_efflux",
                ("mouse MDCK-MDR1 bidirectional transport",),
            ),
        ),
    }
    provisional_labels = {
        ("direct_bbb", "global_context"): {
            "GdET1WI": "gadolinium-enhanced t1wi",
            "PAMPA-BBB": "pampa-bbb",
            "GdET₁WI": "mri",
        },
        ("direct_bbb", "global_species_context"): {
            "Rattus norvegicus": "rat",
        },
        ("passive_permeability", "global_context"): {
            "PAMPA-BBB": "pampa-bbb",
        },
        ("passive_permeability", "global_species_context"): {
            "PAMPA-BBB": None,
        },
        ("efflux_transport", "global_context"): {
            "mouse MDCK-MDR1 bidirectional transport": (
                "mdck-mdr1 bidirectional transport"
            ),
        },
        ("efflux_transport", "global_species_context"): {
            "mouse MDCK-MDR1 bidirectional transport": "mice",
        },
    }
    published_labels = {
        **provisional_labels,
        ("direct_bbb", "global_context"): {
            "GdET1WI": "gadolinium-enhanced mri",
            "PAMPA-BBB": "pampa-bbb",
            "GdET₁WI": "gadolinium-enhanced mri",
        },
        ("efflux_transport", "global_species_context"): {
            "mouse MDCK-MDR1 bidirectional transport": "mouse",
        },
    }

    for (source_id, input_column), clusters in inventories.items():
        values = sorted(
            {value for cluster in clusters for value in cluster.values},
            key=lambda value: (value.casefold(), value),
        )
        _write_json(
            cache_dir / f"clusters__{source_id}__{input_column}.json",
            {
                "identity": _cluster_cache_identity(
                    values,
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                    target_size=DEFAULT_CLUSTER_TARGET_SIZE,
                    random_seed=DEFAULT_CLUSTER_RANDOM_SEED,
                ),
                "clusters": [
                    {"cluster_id": cluster.cluster_id, "values": list(cluster.values)}
                    for cluster in clusters
                ],
            },
        )

    generation_sections = {}
    published_sources: dict[str, dict] = {}
    for spec in extraction_specs():
        clusters = inventories[(spec.source_id, spec.input_column)]
        labels = provisional_labels[(spec.source_id, spec.output_field)]
        cache_records = []
        cluster_generation = {}
        for cluster in clusters:
            generation = _generation(
                spec.source_id, spec.output_field, cluster.cluster_id
            )
            cache_records.append(
                {
                    "generation": generation,
                    "identity": _cache_identity(
                        spec=spec,
                        cluster=cluster,
                        model=DEFAULT_MODEL,
                        reasoning_effort=DEFAULT_REASONING_EFFORT,
                        prompt_version=PROMPT_VERSION,
                    ),
                    "mapping": {
                        f"v{index:04d}": labels[value]
                        for index, value in enumerate(cluster.values)
                    },
                }
            )
            cluster_generation[cluster.cluster_id] = generation
        cache_path = cache_dir / f"{spec.source_id}__{spec.output_field}.jsonl"
        cache_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in cache_records
            ),
            encoding="utf-8",
        )
        generation_sections[f"{spec.source_id}/{spec.output_field}"] = {
            "input_column": spec.input_column,
            "raw_values": sum(len(cluster.values) for cluster in clusters),
            "clusters": len(clusters),
            "null_mappings": sum(value is None for value in labels.values()),
            "usage": _usage(),
            "cluster_generation": cluster_generation,
        }

        mapping = {
            json.dumps([raw], ensure_ascii=False, separators=(",", ":")): value
            for raw, value in published_labels[
                (spec.source_id, spec.output_field)
            ].items()
        }
        mapping["[null]"] = None
        published_sources.setdefault(spec.source_id, {})[spec.output_field] = {
            "source_columns": [spec.input_column],
            "mapping": mapping,
        }

    published_path = tmp_path / "published_mapping.json"
    _write_json(
        published_path,
        {"mapping_version": MAPPING_VERSION, "sources": published_sources},
    )
    generation_path = tmp_path / "generation.json"
    _write_json(
        generation_path,
        {
            "mapping_version": MAPPING_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": DEFAULT_MODEL,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "cluster_target_size": DEFAULT_CLUSTER_TARGET_SIZE,
            "cluster_random_seed": DEFAULT_CLUSTER_RANDOM_SEED,
            "mapping_sha256": _sha256(published_path),
            "sections": generation_sections,
            "cleaned_key_conflict_reconciliation": {
                "contract_version": "globally_reconciled_cleaned_keys.v1",
                "conflicting_cleaned_keys": 1,
                "sections": {},
                "validations": {"all_conflicts_resolved_by_model": True},
            },
        },
    )
    return cache_dir, generation_path, published_path


def _reconstruct(tmp_path: Path):
    cache_dir, generation_path, published_path = _synthetic_bundle(tmp_path)
    result = reconstruct_provisional(
        cache_dir=cache_dir,
        generation_audit_path=generation_path,
        published_mapping_path=published_path,
    )
    return (*result, cache_dir, generation_path, published_path)


def _mapping_delta(before: dict, after: dict) -> set[tuple[str, str, str, object, object]]:
    output = set()
    for source_id, outputs in before["sources"].items():
        for output_field, section in outputs.items():
            after_mapping = after["sources"][source_id][output_field]["mapping"]
            for tuple_key, provisional_label in section["mapping"].items():
                final_label = after_mapping[tuple_key]
                if provisional_label != final_label:
                    output.add(
                        (
                            source_id,
                            output_field,
                            tuple_key,
                            provisional_label,
                            final_label,
                        )
                    )
    return output


def test_reconstructs_exact_cluster_and_item_coverage_with_explicit_null(tmp_path):
    provisional, assignments, audit, _, _, _ = _reconstruct(tmp_path)

    assert len(assignments) == 8
    assert set(
        (
            "source_id",
            "output_field",
            "input_column",
            "cluster_id",
            "item_id",
            "raw_value",
            "tuple_key",
            "provisional_label",
            "cache_identity",
            "served_model",
            "response_sha256",
        )
    ).issubset(assignments.columns)
    assert not assignments.duplicated(
        ["source_id", "output_field", "cluster_id", "item_id"]
    ).any()
    assert assignments["served_model"].eq("gpt-5.4-mini-2026-03-17").all()
    assert assignments["response_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    assert "provisional" in provisional["mapping_version"]
    assert set(provisional["sources"]) == {
        "direct_bbb",
        "passive_permeability",
        "efflux_transport",
    }
    for outputs in provisional["sources"].values():
        for section in outputs.values():
            assert section["mapping"]["[null]"] is None
    assert all(audit["validations"].values())


def test_preserves_the_exact_published_before_after_delta(tmp_path):
    provisional, assignments, _, _, _, published_path = _reconstruct(tmp_path)
    published = json.loads(published_path.read_text(encoding="utf-8"))

    assert _mapping_delta(provisional, published) == {
        (
            "direct_bbb",
            "global_context",
            '["GdET1WI"]',
            "gadolinium-enhanced t1wi",
            "gadolinium-enhanced mri",
        ),
        (
            "direct_bbb",
            "global_context",
            '["GdET₁WI"]',
            "mri",
            "gadolinium-enhanced mri",
        ),
        (
            "efflux_transport",
            "global_species_context",
            '["mouse MDCK-MDR1 bidirectional transport"]',
            "mice",
            "mouse",
        ),
    }
    changed_keys = {
        (row.source_id, row.output_field, row.tuple_key)
        for row in assignments.itertuples(index=False)
        if provisional["sources"][row.source_id][row.output_field]["mapping"][
            row.tuple_key
        ]
        != published["sources"][row.source_id][row.output_field]["mapping"][
            row.tuple_key
        ]
    }
    assert len(changed_keys) == 3


def test_artifact_writes_are_deterministic_and_replayable(tmp_path):
    cache_dir, generation_path, published_path = _synthetic_bundle(tmp_path)
    left = tmp_path / "left"
    right = tmp_path / "right"

    left_manifest = write_provisional_artifacts(
        output_dir=left,
        cache_dir=cache_dir,
        generation_audit_path=generation_path,
        published_mapping_path=published_path,
    )
    right_manifest = write_provisional_artifacts(
        output_dir=right,
        cache_dir=cache_dir,
        generation_audit_path=generation_path,
        published_mapping_path=published_path,
    )

    assert {_sha256(left / name) for name in ARTIFACT_NAMES} == {
        _sha256(right / name) for name in ARTIFACT_NAMES
    }
    assert {
        name: _sha256(left / name) for name in ARTIFACT_NAMES
    } == {
        name: _sha256(right / name) for name in ARTIFACT_NAMES
    }
    assert left_manifest == json.loads(
        (left / "provisional_manifest.json").read_text(encoding="utf-8")
    )
    assert right_manifest == json.loads(
        (right / "provisional_manifest.json").read_text(encoding="utf-8")
    )
    assert pd.read_parquet(left / "cluster_assignments.parquet").to_dict(
        orient="records"
    ) == pd.read_parquet(right / "cluster_assignments.parquet").to_dict(
        orient="records"
    )

    replay_manifest = write_provisional_artifacts(
        output_dir=left,
        cache_dir=cache_dir,
        generation_audit_path=generation_path,
        published_mapping_path=published_path,
        overwrite=True,
    )
    assert replay_manifest == left_manifest


def test_missing_cluster_response_is_rejected(tmp_path):
    cache_dir, generation_path, published_path = _synthetic_bundle(tmp_path)
    path = cache_dir / "direct_bbb__global_context.jsonl"
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n")

    with pytest.raises(ValueError, match="missing|coverage|cluster"):
        reconstruct_provisional(
            cache_dir=cache_dir,
            generation_audit_path=generation_path,
            published_mapping_path=published_path,
        )


def test_duplicate_cluster_response_is_rejected(tmp_path):
    cache_dir, generation_path, published_path = _synthetic_bundle(tmp_path)
    path = cache_dir / "direct_bbb__global_context.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")

    with pytest.raises(ValueError, match="duplicate"):
        reconstruct_provisional(
            cache_dir=cache_dir,
            generation_audit_path=generation_path,
            published_mapping_path=published_path,
        )
