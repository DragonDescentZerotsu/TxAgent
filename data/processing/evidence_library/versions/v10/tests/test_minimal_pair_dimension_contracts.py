from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from data.processing.evidence_library.shared.v2.auxiliary_metadata import (
    SourceUniverseAuxiliaryAttacher,
)
from data.processing.evidence_library.shared.v2.clustered_auxiliary_mapping import (
    AuxiliaryExtractionSpec,
    Cluster,
    _query_cluster,
    build_clustered_auxiliary_mapping,
    clean_bucket,
    cluster_values,
    distinct_source_values,
    reconciliation_plan,
    validate_response,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.pair_dimension_reconciliation import (
    _compose_global_mapping,
    establish_reviewed_mapping,
    extraction_specs,
    global_reconciliation_plan,
    prepare_agent_review,
)
from data.processing.evidence_library.versions.v10 import pair_dimension_reconciliation
from data.processing.evidence_library.versions.v10.tasks.ames.build_starling_downstream_artifacts import (
    get_spec as ames_stage3_spec,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_schema import (
    PAIR_BUCKETS as AMES_PAIR_BUCKETS,
)
from data.processing.evidence_library.versions.v10.tasks.carcinogens.starling_policy import (
    PAIR_BUCKETS as CARCINOGENS_PAIR_BUCKETS,
)
from data.processing.evidence_library.versions.v10.tasks.carcinogens.build_starling_downstream_artifacts import (
    get_spec as carcinogens_stage3_spec,
)
from data.processing.evidence_library.versions.v10.tasks.dili.build_starling_downstream_artifacts import (
    get_spec as dili_stage3_spec,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_schema import (
    PAIR_BUCKETS as DILI_PAIR_BUCKETS,
)


CORE = (
    "canonical_endpoint_name",
    "canonical_unit_text",
    "canonical_measurement_scale_id",
)


def test_pair_contracts_use_only_minimal_declared_dimensions() -> None:
    expected_ames = {source: (*CORE, "canonical_assay_context") for source in AMES_PAIR_BUCKETS}
    expected_dili = {
        source: (
            *CORE,
            "canonical_assay_context",
            *(("canonical_species_context",) if source in {"dili_v1", "dili_v2"} else ()),
        )
        for source in DILI_PAIR_BUCKETS
    }
    expected_carcinogens = {
        source: (
            *CORE,
            "canonical_assay_context",
            *(
                ("canonical_species_context",)
                if source in {"carcinogens_base", "carcinogens_v3"}
                else ()
            ),
        )
        for source in CARCINOGENS_PAIR_BUCKETS
    }
    assert {key: value.canonical_dimensions for key, value in AMES_PAIR_BUCKETS.items()} == expected_ames
    assert {key: value.canonical_dimensions for key, value in DILI_PAIR_BUCKETS.items()} == expected_dili
    assert {
        key: value.canonical_dimensions for key, value in CARCINOGENS_PAIR_BUCKETS.items()
    } == expected_carcinogens
    forbidden = {
        "gold_label",
        "split",
        "parent",
        "effect_direction",
        "exposure_regimen",
    }
    all_fields = {
        field
        for specs in (AMES_PAIR_BUCKETS, DILI_PAIR_BUCKETS, CARCINOGENS_PAIR_BUCKETS)
        for spec in specs.values()
        for field in spec.canonical_dimensions
    }
    assert not forbidden & all_fields


def test_composite_auxiliary_inventory_uses_stable_tuple_keys(tmp_path) -> None:
    path = tmp_path / "records.parquet"
    frame = pd.DataFrame(
        {
            "assay": ["LC-MS", "LC-MS", None],
            "readout": ["ALT", "AST", "unknown"],
        }
    )
    frame.to_parquet(path, index=False)
    values = distinct_source_values(frame, ("assay", "readout"))
    assert values == ['["LC-MS","ALT"]', '["LC-MS","AST"]']

    spec = AuxiliaryExtractionSpec(
        source_id="dili_v4",
        input_path=path,
        input_column=None,
        input_columns=("assay", "readout"),
        output_field="canonical_assay_context",
        prompt="Map each tuple.",
    )
    plan = reconciliation_plan((spec,), cluster_target_size=50)
    assert plan["sections"]["dili_v4/canonical_assay_context"] == {
        "input_columns": ["assay", "readout"],
        "distinct_values": 2,
        "planned_clusters": 1,
    }
    assert json.loads(values[0]) == ["LC-MS", "ALT"]


def test_single_column_auxiliary_plan_keeps_v1_shape(tmp_path) -> None:
    path = tmp_path / "records.parquet"
    pd.DataFrame({"source_id": ["source"], "assay": ["Ames"]}).to_parquet(
        path, index=False
    )
    spec = AuxiliaryExtractionSpec(
        source_id="source",
        input_path=path,
        input_column="assay",
        output_field="canonical_assay_context",
        prompt="Map each assay.",
    )
    section = reconciliation_plan((spec,))["sections"][
        "source/canonical_assay_context"
    ]
    assert section == {
        "input_column": "assay",
        "distinct_values": 1,
        "planned_clusters": 1,
    }


def test_single_element_input_columns_publish_scalar_lookup_key(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "records.parquet"
    pd.DataFrame({"source_id": ["source"], "endpoint": ["tumor"]}).to_parquet(
        path, index=False
    )
    monkeypatch.setattr(
        "data.processing.evidence_library.shared.v2.clustered_auxiliary_mapping.embed_values",
        lambda values, **kwargs: np.ones((len(values), 1)),
    )

    class Pool:
        def chat_json(self, messages, *, max_tokens=None):
            return {"raw_content": '{"mapping":{"v0000":"neoplasm"}}'}

    output = tmp_path / "mapping.json"
    build_clustered_auxiliary_mapping(
        specs=(
            AuxiliaryExtractionSpec(
                source_id="source",
                input_path=path,
                input_column=None,
                input_columns=("endpoint",),
                output_field="canonical_endpoint",
                prompt="Map endpoints.",
            ),
        ),
        output_path=output,
        mapping_version="test.v1",
        prompt_version="test.v1",
        api_key_loader=None,
        client=Pool(),
        model="test-model",
        reasoning_effort="high",
    )

    section = json.loads(output.read_text())["sources"]["source"][
        "canonical_endpoint"
    ]
    assert section["mapping"] == {'["tumor"]': "neoplasm", "[null]": None}


def test_reconciliation_reads_local_stage1_source_tuples(tmp_path) -> None:
    specs = extraction_specs("ames", tmp_path / "stage1.parquet")
    assert {spec.source_columns for spec in specs} == {
        (
            "endpoint_name",
            "test_system",
            "metabolic_activation",
        ),
        (
            "endpoint_name",
            "study_context",
            "biological_test_system",
            "metabolic_activation_status",
        ),
        (
            "endpoint_name",
            "assay_version",
            "endpoint_subtype",
            "biological_system",
            "metabolic_activation_presence",
        ),
        (
            "endpoint_name",
            "mechanism_category",
            "assay_method_and_endpoint",
            "biological_system",
            "metabolic_activation_system",
        ),
    }
    assert {spec.input_source_id for spec in specs} == {None}


def test_agent_mapping_labels_are_accepted_verbatim() -> None:
    labels = {
        "a": " Unknown / N/A {free-form} → keep\nexactly ",
        "b": None,
        "c": "__null__",
    }
    expected = {
        "a": labels["a"],
        "b": None,
        "c": None,
    }
    for payload in (labels, {"mapping": labels, "explanation": "ignored"}):
        assert validate_response(
            json.dumps(payload),
            item_ids={key: key for key in labels},
            null_sentinel="__null__",
        ) == expected
    assert clean_bucket(" __null__ ", null_sentinel="__null__") == " __null__ "


def test_cluster_query_accepts_shared_provider_pool_client(tmp_path) -> None:
    class Pool:
        def chat_json(self, messages, *, max_tokens=None):
            assert max_tokens == 131_072
            return {
                "raw_content": '{"mapping":{"v0000":"bacterial mutation"}}',
                "model": "deepseek/deepseek-v4.1-flash",
                "usage": {"completion_tokens": 8, "total_tokens": 16},
                "execution_provider": {
                    "provider": "openrouter_wafer_fast",
                    "provider_pool_snapshot_sha256": "snapshot",
                },
            }

    mapping, audit = _query_cluster(
        Pool(),
        spec=AuxiliaryExtractionSpec(
            source_id="ames",
            input_path=tmp_path / "unused.parquet",
            input_column="assay",
            output_field="canonical_assay_context",
            prompt="Map assays.",
        ),
        cluster=Cluster("cluster", ("Ames test",)),
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        reasoning_effort="high",
        max_retries=1,
        retry_delay=0,
        max_tokens=131_072,
    )

    assert mapping == {"v0000": "bacterial mutation"}
    assert audit["attempts"][0]["served_model"] == "deepseek/deepseek-v4.1-flash"
    assert audit["attempts"][0]["execution_provider"][
        "provider_pool_snapshot_sha256"
    ] == "snapshot"


def test_pair_dimension_cli_adds_multiple_local_endpoints_to_pool(
    tmp_path, monkeypatch, capsys
) -> None:
    from predict.api_client import pool

    config = tmp_path / "providers.json"
    config.write_text(
        json.dumps(
            {
                "version": "openai_provider_pool.v1",
                "providers": [
                    {
                        "name": "openrouter",
                        "base_url": "https://openrouter.ai/api/v1",
                        "model": "deepseek/deepseek-v4-flash-0731",
                        "api_key_env": "OPEN_ROUTER_KEY",
                        "max_inflight": 1,
                    }
                ],
                "openrouter_ranked_profile": {
                    "profile": "0731",
                    "snapshot_sha256": "snapshot",
                },
            }
        )
    )
    captured = {}

    def build_provider_pool(configured, **kwargs):
        captured["configured"] = configured
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(pool, "build_provider_pool", build_provider_pool)
    monkeypatch.setattr(
        pair_dimension_reconciliation,
        "run_global_reconciliation",
        lambda **kwargs: {"status": "prepared"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pair_dimension_reconciliation",
            "globalize",
            "--task",
            "carcinogens",
            "--input",
            str(tmp_path / "candidate.json"),
            "--output",
            str(tmp_path / "global.json"),
            "--provider-pool-config",
            str(config),
            "--base-url",
            "http://dgx020:50002/v1",
            "--additional-local-base-url",
            "http://dgx009:50001/v1",
            "--local-concurrency",
            "2",
            "--workers",
            "5",
        ],
    )

    pair_dimension_reconciliation.main()

    providers = captured["configured"].providers
    assert [(provider.base_url, provider.max_inflight) for provider in providers] == [
        ("http://dgx020:50002/v1", 2),
        ("http://dgx009:50001/v1", 2),
        ("https://openrouter.ai/api/v1", 1),
    ]
    assert captured["kwargs"]["reasoning_effort"] == "high"
    assert json.loads(capsys.readouterr().out) == {"status": "prepared"}


def test_minibatch_clusters_are_deterministic_and_bounded() -> None:
    values = [f"value-{index}" for index in range(211)]
    embeddings = np.random.default_rng(7).normal(size=(len(values), 16))
    first = cluster_values(values, embeddings, target_size=20, max_size=25)
    second = cluster_values(values, embeddings, target_size=20, max_size=25)
    assert first == second
    assert max(len(cluster.values) for cluster in first) <= 25
    assert {value for cluster in first for value in cluster.values} == set(values)


def test_source_universe_auxiliary_mapping_joins_by_uid_and_source(tmp_path) -> None:
    records = tmp_path / "task" / "records.parquet"
    records.parent.mkdir()
    pd.DataFrame(
        {
            "source_row_uid": ["sr_one", "sr_two"],
            "source_id": ["upstream_a", "upstream_b"],
            "canonical_assay_context": ["Ames test", None],
        }
    ).to_parquet(records, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "main_source_universe.v1",
                "tasks": {
                    "task": {
                        "path": "task/records.parquet",
                        "rows": 2,
                        "sha256": file_sha256(records),
                    }
                },
            }
        )
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "mapping_version": "task_pair_dimensions.v1",
                "sources": {
                    "source_a": {
                        "canonical_assay_context": {
                            "source_columns": ["canonical_assay_context"],
                            "mapping": {'["Ames test"]': "bacterial mutation"},
                        }
                    },
                    "source_b": {
                        "canonical_assay_context": {
                            "source_columns": ["canonical_assay_context"],
                            "mapping": {"[null]": None},
                        }
                    },
                },
            }
        )
    )
    attacher = SourceUniverseAuxiliaryAttacher(
        mapping,
        mapping_version="task_pair_dimensions.v1",
        applicable_sources=("source_a", "source_b"),
        null_like=("unknown",),
        output_fields=("canonical_assay_context",),
        universe_records_path=records,
        universe_manifest_path=manifest,
        task_id="task",
        source_aliases={"upstream_a": "source_a", "upstream_b": "source_b"},
    )

    assert attacher.attach(
        {"source_row_uid": "sr_one", "source_id": "source_a"}
    )["canonical_assay_context"] == "bacterial mutation"
    assert attacher.attach(
        {"source_row_uid": "sr_two", "source_id": "source_b"}
    )["canonical_assay_context"] is None
    assert attacher.manifest()["source_universe_projection"]["records"] == 2
    with pytest.raises(ValueError, match="source mismatch"):
        attacher.attach({"source_row_uid": "sr_one", "source_id": "source_b"})
    with pytest.raises(ValueError, match="lacks UID"):
        attacher.attach({"source_row_uid": "sr_missing", "source_id": "source_a"})


def test_all_three_tasks_bind_the_shared_canonical_stage3() -> None:
    specs = (ames_stage3_spec(), dili_stage3_spec(), carcinogens_stage3_spec())
    assert [spec.task_id for spec in specs] == ["ames", "dili", "carcinogens"]
    assert all(spec.pair_bucket_version.endswith(".v10") for spec in specs)
    assert all(spec.direct_mapping_builder is None for spec in specs)


def test_global_plan_covers_every_first_pass_section(tmp_path) -> None:
    candidate = {"mapping_version": "candidate", "sources": {}}
    specs = extraction_specs("dili", tmp_path / "unused.parquet")
    for spec in specs:
        candidate["sources"].setdefault(spec.source_id, {})[spec.output_field] = {
            "source_columns": list(spec.source_columns),
            "mapping": {'["raw-a"]': "label-a", '["raw-b"]': "label-b", "[null]": None},
        }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    plan = global_reconciliation_plan("dili", path, cluster_size=1)
    assert set(plan["sections"]) == {
        f"{spec.source_id}/{spec.output_field}" for spec in specs
    }
    assert all(row["provisional_labels"] == 2 for row in plan["sections"].values())
    assert plan["planned_clusters"] == 2 * len(specs)


def test_global_label_mapping_is_composed_over_raw_keys() -> None:
    candidate = {
        "sources": {
            "source": {
                "field": {
                    "source_columns": ["raw"],
                    "mapping": {'["a"]': "alias-a", '["b"]': "alias-b", "[null]": None},
                }
            }
        }
    }
    labels = {
        "sources": {
            "source": {
                "field": {
                    "mapping": {
                        '["alias-a"]': "unified",
                        '["alias-b"]': "unified",
                        "[null]": None,
                    }
                }
            }
        }
    }
    result = _compose_global_mapping("dili", candidate, labels)
    section = result["sources"]["source"]["field"]
    assert result["mapping_version"] == "dili_pair_dimensions.v1"
    assert section["mapping"] == {
        '["a"]': "unified",
        '["b"]': "unified",
        "[null]": None,
    }


def test_global_label_mapping_normalizes_null_like_provisional_labels() -> None:
    candidate = {
        "sources": {
            "source": {
                "field": {
                    "source_columns": ["raw"],
                    "mapping": {
                        '["a"]': " unspecified ",
                        '["b"]': " active/cleaved? ",
                        "[null]": None,
                    },
                }
            }
        }
    }
    labels = {
        "sources": {
            "source": {
                "field": {
                    "mapping": {
                        '["active/cleaved?"]': "active or cleaved",
                        "[null]": None,
                    },
                }
            }
        }
    }

    result = _compose_global_mapping("dili", candidate, labels)

    assert result["sources"]["source"]["field"]["mapping"] == {
        '["a"]': None,
        '["b"]': "active or cleaved",
        "[null]": None,
    }


def test_global_endpoint_name_pass_publishes_stage2_endpoint_concept() -> None:
    candidate = {
        "sources": {
            "source": {
                "canonical_endpoint_name": {
                    "source_columns": ["canonical_endpoint_name"],
                    "mapping": {'["raw"]': "provisional"},
                }
            }
        }
    }
    labels = {
        "sources": {
            "source": {
                "canonical_endpoint_name": {
                    "mapping": {'["provisional"]': "reviewed concept"}
                }
            }
        }
    }
    result = _compose_global_mapping("dili", candidate, labels)
    assert set(result["sources"]["source"]) == {"canonical_endpoint_concept"}
    assert result["sources"]["source"]["canonical_endpoint_concept"]["mapping"] == {
        '["raw"]': "reviewed concept"
    }


def test_agent_review_establishes_hash_bound_successor(tmp_path) -> None:
    candidate = {
        "mapping_version": "dili_pair_dimensions.v1",
        "sources": {},
    }
    for spec in extraction_specs("dili", tmp_path / "unused.parquet"):
        field = (
            "canonical_endpoint_concept"
            if spec.output_field == "canonical_endpoint_name"
            else spec.output_field
        )
        candidate["sources"].setdefault(spec.source_id, {})[field] = {
            "source_columns": list(spec.source_columns),
            "mapping": {'["a"]': "alias-a", '["b"]': "alias-b"},
        }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    candidate_path.with_suffix(".json.generation.json").write_text(
        json.dumps(
            {
                "publication_status": "unpublished_requires_agent_review",
                "output": {"sha256": file_sha256(candidate_path)},
            }
        )
    )
    packet_path = tmp_path / "packet.json"
    packet = prepare_agent_review("dili", candidate_path, packet_path)
    assert packet["candidate"]["sha256"] == file_sha256(candidate_path)
    assert packet["mapping_id"] == "auxiliary_context"

    first_source = sorted(candidate["sources"])[0]
    first_field = sorted(candidate["sources"][first_source])[0]
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "version": "pair_dimension_agent_review.v1",
                "task_id": "dili",
                "candidate_sha256": file_sha256(candidate_path),
                "reviewer": "test-agent",
                "decision": "approve",
                "review_completion": {
                    "all_sections_reviewed": True,
                    "all_labels_reviewed": True,
                },
                "overrides": [
                    {
                        "source_id": first_source,
                        "field": first_field,
                        "from": "alias-b",
                        "to": "alias-a",
                        "rationale": "Equivalent labels in the reviewed context.",
                    }
                ],
            }
        )
    )
    output = tmp_path / "reviewed.json"
    receipt_path = tmp_path / "receipt.json"
    receipt = establish_reviewed_mapping(
        "dili", candidate_path, review_path, output, receipt_path
    )
    reviewed = json.loads(output.read_text())
    assert set(reviewed["sources"][first_source][first_field]["mapping"].values()) == {
        "alias-a"
    }
    assert receipt["mapping"]["sha256"] == file_sha256(output)
    assert receipt["review_completion"]["override_count"] == 1
