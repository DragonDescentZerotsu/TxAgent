from __future__ import annotations

import asyncio
import json

from data.processing.evidence_library.versions.v10.unit_reconciliation import (
    ATTEMPTS,
    partition_rows,
    resolve_partition,
    safe_canonical_unit,
    shuffled_rows,
)


def test_partitions_are_groups_of_fifty() -> None:
    assert [len(group) for group in partition_rows(list(range(101)))] == [50, 50, 1]


def test_pass_b_shuffle_is_manifest_deterministic() -> None:
    rows = list(range(100))
    first = shuffled_rows(rows, "a" * 64)
    assert first == shuffled_rows(rows, "a" * 64)
    assert first != shuffled_rows(rows, "b" * 64)
    assert sorted(first) == rows


def test_retry_sequence_uses_third_attempt() -> None:
    rows = [{"id": "u_1", "unit": "µM"}]
    called = []

    async def request(spec, _prompt):
        called.append(spec)
        if len(called) < 3:
            raise RuntimeError("temporary failure")
        return json.dumps({"mappings": [{"id": "u_1", "canonical_unit": "µM"}]})

    mappings, attempts, status = asyncio.run(resolve_partition("dili", rows, request))
    assert called == list(ATTEMPTS)
    assert [attempt["status"] for attempt in attempts] == ["failed", "failed", "ok"]
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
