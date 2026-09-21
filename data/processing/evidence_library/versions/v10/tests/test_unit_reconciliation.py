from __future__ import annotations

import asyncio
import json

import pytest

from data.processing.evidence_library.versions.v10.canonical_reconciliation import (
    ATTEMPTS,
    LOCAL_MAX_TOKENS,
    OPENAI_MAX_TOKENS,
    _v3_clusters,
    _v3_guarded_unit,
    _v3_seed_resolved,
    _v3_validate_response,
    _load_review_decisions,
    consolidate,
    consolidate_draft,
    context_graph_partition_rows,
    partition_rows,
    prepare_review,
    resolve_partition,
    safe_canonical_unit,
    snapshot_v3_identities,
    terminal_events,
    validate,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)


def test_partitions_are_groups_of_fifty() -> None:
    assert [len(group) for group in partition_rows(list(range(101)))] == [50, 50, 1]


def test_context_graph_batching_is_deterministic_and_complete() -> None:
    rows = [
        {"id": "u_1", "unit": "one", "context_tokens": ["assay=a"]},
        {"id": "u_2", "unit": "two", "context_tokens": ["assay=b"]},
        {"id": "u_3", "unit": "three", "context_tokens": ["assay=a"]},
    ]
    first = context_graph_partition_rows(rows, size=2)
    assert first == context_graph_partition_rows(rows, size=2)
    assert [[row["id"] for row in group] for group in first] == [
        ["u_1", "u_3"],
        ["u_2"],
    ]


def test_retry_sequence_uses_third_attempt() -> None:
    rows = [{"id": "u_1", "unit": "µM"}]
    called = []

    async def request(spec, _prompt):
        called.append(spec)
        if len(called) < 3:
            raise RuntimeError("temporary failure")
        return (
            json.dumps({"mappings": [{"id": "u_1", "canonical_unit": "µM"}]}),
            {"completion_tokens": 10},
        )

    mappings, attempts, status = asyncio.run(resolve_partition("dili", rows, request))
    assert called == list(ATTEMPTS)
    assert [attempt["status"] for attempt in attempts] == ["failed", "failed", "ok"]
    assert [attempt.provider for attempt in called] == ["local", "local", "local"]
    assert [attempt.max_tokens for attempt in called] == [
        OPENAI_MAX_TOKENS,
        OPENAI_MAX_TOKENS,
        LOCAL_MAX_TOKENS,
    ]
    assert attempts[2]["usage"] == {"completion_tokens": 10}
    assert status == "resolved"
    assert mappings[0]["canonical_unit"] == "µM"


def test_retry_exhaustion_is_identity_and_complete() -> None:
    rows = [{"id": "u_1", "unit": "odd A"}, {"id": "u_2", "unit": "odd B"}]

    async def request(_spec, _prompt):
        raise RuntimeError("down")

    mappings, attempts, status = asyncio.run(resolve_partition("ames", rows, request))
    assert len(attempts) == 3
    assert status == "identity_after_retry_exhaustion"
    assert [(row["id"], row["canonical_unit"]) for row in mappings] == [
        ("u_1", "odd A"),
        ("u_2", "odd B"),
    ]


def test_scale_and_qualifier_guards() -> None:
    assert safe_canonical_unit("uM", "µM", "dili")[0] == "µM"
    assert safe_canonical_unit("nM", "µM", "dili")[0] == "nM"
    assert safe_canonical_unit("mM", "µM", "dili")[0] == "mM"
    assert safe_canonical_unit("10^-6 mol/L", "µM", "dili")[0] == "10^-6 mol/L"
    assert safe_canonical_unit("% inhibition", "%", "dili")[0] == "% inhibition"


def test_seed_cache_inherits_only_resolved_exact_partitions(tmp_path) -> None:
    rows = [[{"id": "u_1", "unit": "µM"}], [{"id": "u_2", "unit": "odd"}]]
    events = [
        {
            "task": "dili",
            "pass": "a",
            "input_sha256": "frozen",
            "partition": 0,
            "input_ids": ["u_1"],
            "status": "resolved",
            "attempts": [],
            "mappings": [
                {"id": "u_1", "canonical_unit": "µM", "guard_decision": "identity"}
            ],
        },
        {
            "task": "dili",
            "pass": "a",
            "input_sha256": "frozen",
            "partition": 1,
            "input_ids": ["u_2"],
            "status": "identity_after_retry_exhaustion",
            "attempts": [],
            "mappings": [
                {
                    "id": "u_2",
                    "canonical_unit": "odd",
                    "guard_decision": "identity_after_retry_exhaustion",
                }
            ],
        },
    ]
    cache = tmp_path / "terminal.jsonl"
    cache.write_text("".join(json.dumps(event) + "\n" for event in events))
    inherited = terminal_events(
        cache,
        task="dili",
        pass_name="a",
        input_sha256="frozen",
        partitions=rows,
        resolved_only=True,
    )
    assert set(inherited) == {0}

    events[0]["input_ids"] = ["wrong"]
    cache.write_text("".join(json.dumps(event) + "\n" for event in events))
    try:
        terminal_events(
            cache,
            task="dili",
            pass_name="a",
            input_sha256="frozen",
            partitions=rows,
            resolved_only=True,
        )
    except ValueError as error:
        assert "membership mismatch" in str(error)
    else:
        raise AssertionError("partition drift was accepted")


def test_agent_review_rejects_scale_change(tmp_path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "from_canonical_unit": "nM",
                "canonical_unit": "µM",
                "decision_type": "equivalent_alias",
                "rationale": "unsafe",
            }
        )
        + "\n"
    )
    try:
        _load_review_decisions(decisions, "dili", {"nM", "µM"})
    except ValueError as error:
        assert "unsafe agent review alias" in str(error)
    else:
        raise AssertionError("scale-changing agent review was accepted")


def test_agent_review_accepts_guarded_spelling_alias(tmp_path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "from_canonical_unit": "uM",
                "canonical_unit": "µM",
                "decision_type": "equivalent_alias",
                "rationale": "equivalent SI typography",
            }
        )
        + "\n"
    )
    aliases, rows = _load_review_decisions(decisions, "dili", {"uM", "µM"})
    assert aliases == {"uM": "µM"}
    assert len(rows) == 1


def test_legacy_single_pass_review_and_publication(tmp_path) -> None:
    task_root = tmp_path / "run/dili"
    pass_root = task_root / "pass_a"
    pass_root.mkdir(parents=True)
    universe = task_root / "universe.jsonl"
    universe.write_text(
        "".join(
            json.dumps({"id": identifier, "unit": unit}) + "\n"
            for identifier, unit in (("u_1", "uM"), ("u_2", "µM"))
        )
    )
    (task_root / "universe.manifest.json").write_text(
        json.dumps(
            {
                "version": "starling_unit_reconciliation_two_pass.v1",
                "task": "dili",
                "unit_count": 2,
                "universe_sha256": file_sha256(universe),
            }
        )
    )
    pass_mapping = pass_root / "mapping.jsonl"
    pass_mapping.write_text(
        "".join(
            json.dumps({"id": identifier, "canonical_unit": unit}) + "\n"
            for identifier, unit in (("u_1", "uM"), ("u_2", "µM"))
        )
    )
    (pass_root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "starling_unit_reconciliation_two_pass.v1",
                "task": "dili",
                "pass": "a",
                "mapping_count": 2,
                "mapping_sha256": file_sha256(pass_mapping),
            }
        )
    )
    base = tmp_path / "base/mapping.json"
    draft = consolidate_draft("dili", tmp_path / "run", base)
    assert draft["publication_status"] == "draft"
    with pytest.raises(ValueError, match="has not completed agent review"):
        validate("dili", tmp_path / "run", base)
    assert validate(
        "dili", tmp_path / "run", base, allow_draft=True
    )["publication_status"] == "draft"
    review_root = tmp_path / "review"
    review_manifest = prepare_review("dili", base, review_root, packet_size=1)
    decisions = review_root / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "from_canonical_unit": "uM",
                "canonical_unit": "µM",
                "decision_type": "equivalent_alias",
                "rationale": "equivalent SI typography",
            }
        )
        + "\n"
    )
    packets = [json.loads(line) for line in (review_root / "packets.jsonl").read_text().splitlines()]
    completion = review_root / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "version": "canonical_unit_agent_review_completion.v1",
                "task": "dili",
                "reviewer_id": "test-reviewer",
                "packet_manifest": {
                    "path": str((review_root / "manifest.json").resolve()),
                    "sha256": file_sha256(review_root / "manifest.json"),
                },
                "decisions_sha256": file_sha256(decisions),
                "reviewed_packet_ids": [row["packet_id"] for row in packets],
            }
        )
    )
    assert review_manifest["packet_count"] == 2
    output = tmp_path / "published/mapping.json"
    manifest = consolidate(
        "dili", tmp_path / "run", output, decisions, completion
    )
    assert manifest["canonical_unit_count"] == 1
    assert manifest["publication_status"] == "reviewed"
    assert manifest["agent_review_decisions"]["accepted_aliases"] == 1
    assert validate("dili", tmp_path / "run", output)["valid"] is True


def test_incremental_unit_successor_preserves_base_and_reviews_only_new_units(
    tmp_path,
) -> None:
    base = tmp_path / "predecessor/mapping.json"
    base.parent.mkdir(parents=True)
    base.write_text(
        json.dumps(
            {
                "version": "starling_exact_measurement_units.v2",
                "task": "dili",
                "entries": [
                    {
                        "task": "dili",
                        "canonical_endpoints": ["*"],
                        "input_unit": "nM",
                        "action": "map",
                        "canonical_unit": "nM",
                        "scale": "1",
                        "domain": "any",
                    },
                    {
                        "task": "dili",
                        "canonical_endpoints": ["cell_viability"],
                        "input_unit": "nM",
                        "action": "map",
                        "canonical_unit": "nM viability basis",
                        "scale": "1",
                        "domain": "any",
                    },
                ],
            }
        )
    )
    task_root = tmp_path / "run/dili"
    pass_root = task_root / "llm_pass"
    pass_root.mkdir(parents=True)
    universe = task_root / "universe.jsonl"
    universe.write_text(json.dumps({"id": "u_1", "unit": "uM"}) + "\n")
    (task_root / "universe.manifest.json").write_text(
        json.dumps(
            {
                "version": "starling_canonical_reconciliation_single_pass.v1",
                "task": "dili",
                "unit_count": 1,
                "universe_sha256": file_sha256(universe),
                "base_mapping": {
                    "path": str(base.resolve()),
                    "sha256": file_sha256(base),
                    "entry_count": 2,
                },
            }
        )
    )
    pass_mapping = pass_root / "mapping.jsonl"
    pass_mapping.write_text(
        json.dumps({"id": "u_1", "canonical_unit": "µM"}) + "\n"
    )
    (pass_root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "starling_canonical_reconciliation_single_pass.v1",
                "task": "dili",
                "mapping_count": 1,
                "mapping_sha256": file_sha256(pass_mapping),
            }
        )
    )

    output = tmp_path / "draft/mapping.json"
    manifest = consolidate_draft("dili", tmp_path / "run", output)
    assert manifest["entry_count"] == 3
    assert [
        row["input_unit"] for row in json.loads(output.read_text())["entries"]
    ] == ["nM", "nM", "uM"]
    assert validate("dili", tmp_path / "run", output, allow_draft=True)["valid"]

    review = prepare_review(
        "dili", output, tmp_path / "review", packet_size=1, base_mapping=base
    )
    assert review["entry_count"] == 3
    assert review["review_entry_count"] == 1
    assert review["packet_count"] == 1


def test_default_publication_requires_agent_review(tmp_path) -> None:
    with pytest.raises(ValueError, match="publication requires"):
        consolidate("dili", tmp_path)


def test_v3_grounded_reduction_preserves_basis_and_scale() -> None:
    canonical, decision = _v3_guarded_unit(
        "ames", "% overall yield of reactions", "% overall yield", "% yield"
    )
    assert canonical == "% yield"
    assert decision == "accepted_grounded_reduction"
    assert _v3_guarded_unit(
        "ames",
        "10^-7 sec^-1 (deamination rate constant)",
        "10^-7 sec^-1",
        "10^-7 s^-1",
    )[0] == "10^-7 s^-1"
    assert _v3_guarded_unit(
        "ames", "nkat mg-1 protein", "nkat mg-1 protein", "nkat/mg protein"
    )[0] == "nkat/mg protein"
    assert _v3_guarded_unit(
        "ames",
        "nmol H2O2 consumed/min/mg protein",
        "nmol H2O2 consumed/min/mg protein",
        "nmol/min/mg protein",
    )[0] == "nmol/min/mg protein"
    assert _v3_guarded_unit(
        "ames",
        "nmol H2O2 consumed/min/mg protein",
        "nmol H2O2 consumed/min/mg protein",
        "nmol/min/mg total protein",
    )[0] == "nmol/min/mg total protein"
    assert _v3_guarded_unit(
        "carcinogens",
        "% chromosome aberrations (0.1 % breaks, 0.2 % translocations) per 1000 nuclei",
        "% chromosome aberrations (0.1 % breaks, 0.2 % translocations) per 1000 nuclei",
        "% chromosome aberrations per 1000 nuclei",
    )[0] == "% chromosome aberrations per 1000 nuclei"
    assert _v3_guarded_unit(
        "carcinogens",
        "cells x 10^3/well",
        "cells x 10^3/well",
        "10^3 cells/well",
    )[0] == "10^3 cells/well"
    assert _v3_guarded_unit(
        "ames",
        "nmol/min/mg protein",
        "nmol/min/mg protein",
        "nmol min^-1 mg protein^-1",
    )[0] == "nmol min^-1 mg protein^-1"
    with pytest.raises(ValueError, match="scale|dimension|denominator"):
        _v3_guarded_unit("dili", "nM", "nM", "µM")
    with pytest.raises(ValueError, match="scale|dimension"):
        _v3_guarded_unit("ames", "mmol/L", "mmol/L", "nmol/L")
    with pytest.raises(ValueError, match="qualifier"):
        _v3_guarded_unit(
            "carcinogens", "% inhibition", "% inhibition", "inhibition"
        )


def test_v3_kmeans_is_endpoint_local_bounded_and_deterministic() -> None:
    rows = [
        {
            "id": f"a_{index}",
            "task": "dili",
            "endpoint": "endpoint_a",
            "unit": f"{index} nmol/min/mg protein",
        }
        for index in range(55)
    ] + [
        {
            "id": f"b_{index}",
            "task": "dili",
            "endpoint": "endpoint_b",
            "unit": f"{index} percent yield",
        }
        for index in range(2)
    ]
    clusters, manifest = _v3_clusters(rows, "a")
    repeated, repeated_manifest = _v3_clusters(rows, "a")
    assert clusters == repeated
    assert manifest == repeated_manifest
    assert max(len(cluster["rows"]) for cluster in clusters) <= 50
    assert all(len({row["endpoint"] for row in cluster["rows"]}) == 1 for cluster in clusters)
    assert sorted(row["id"] for cluster in clusters for row in cluster["rows"]) == sorted(
        row["id"] for row in rows
    )


def test_v3_response_requires_verbatim_span_and_complete_coverage() -> None:
    cluster = {
        "task": "ames",
        "endpoint": "yield",
        "rows": [{"id": "u_1", "unit": "% overall yield of reactions"}],
    }
    assert _v3_validate_response(
        json.dumps(
            {
                "mappings": [
                    {
                        "id": "u_1",
                        "unit_span": "% overall yield",
                        "canonical_unit": "%",
                    }
                ]
            }
        ),
        cluster,
    )[0]["canonical_unit"] == "%"
    with pytest.raises(ValueError, match="verbatim span"):
        _v3_validate_response(
            json.dumps(
                {
                    "mappings": [
                        {
                            "id": "u_1",
                            "unit_span": "% reaction yield",
                            "canonical_unit": "%",
                        }
                    ]
                }
            ),
            cluster,
        )


@pytest.mark.parametrize(
    "unit",
    (
        "% overall yield",
        "% inhibition",
        "% loss",
        "% breaks repaired",
        "% GSH depleted",
    ),
)
def test_v3_percent_outcome_semantics_may_reduce_to_percent(unit: str) -> None:
    canonical, decision = _v3_guarded_unit("ames", unit, unit, "%")
    assert canonical == "%"
    assert decision == "accepted_percent_semantic_reduction"


@pytest.mark.parametrize(
    "unit",
    (
        "% of control",
        "% inhibition relative to vehicle control",
        "% of total cells",
        "10^-3 % yield",
    ),
)
def test_v3_percent_reference_bases_and_scales_remain_protected(unit: str) -> None:
    with pytest.raises(ValueError, match="qualifier|basis|scale"):
        _v3_guarded_unit("ames", unit, unit, "%")


def test_v3_recovery_seed_inherits_only_resolved_clusters(tmp_path) -> None:
    clusters = [
        {
            "cluster_id": f"c_{index}",
            "rows": [{"id": f"u_{index}"}],
        }
        for index in range(2)
    ]
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        "".join(
            json.dumps(
                {
                    "cluster_id": f"c_{index}",
                    "input_ids": [f"u_{index}"],
                    "status": status,
                }
            )
            + "\n"
            for index, status in enumerate(
                ("resolved", "identity_after_retry_exhaustion")
            )
        )
    )
    terminal = tmp_path / "terminal.jsonl"
    receipt = _v3_seed_resolved(terminal, seed, clusters)
    inherited = [json.loads(line) for line in terminal.read_text().splitlines()]
    assert receipt["inherited_resolved_count"] == 1
    assert receipt["requeued_count"] == 1
    assert [row["cluster_id"] for row in inherited] == ["c_0"]


def test_v3_identity_snapshot_selects_only_terminal_fallbacks(tmp_path) -> None:
    pass_root = tmp_path / "run/v3/pass_a"
    pass_root.mkdir(parents=True)
    clusters = [
        {"cluster_id": f"c_{index}", "rows": [{"id": f"u_{index}"}]}
        for index in range(3)
    ]
    (pass_root / "clusters.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in clusters)
    )
    (pass_root / "terminal.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "cluster_id": f"c_{index}",
                    "input_ids": [f"u_{index}"],
                    "status": status,
                }
            )
            + "\n"
            for index, status in enumerate(
                ("resolved", "identity_after_retry_exhaustion")
            )
        )
    )
    output = tmp_path / "snapshot"
    manifest = snapshot_v3_identities(tmp_path / "run", "a", output)
    selected = [
        json.loads(line) for line in (output / "clusters.jsonl").read_text().splitlines()
    ]
    assert manifest["identity_cluster_count"] == 1
    assert [row["cluster_id"] for row in selected] == ["c_1"]
