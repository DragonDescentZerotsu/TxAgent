from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

import pytest

from data.processing.evidence_library.shared.v2 import build_runtime


def _projection_fixture(bounds: tuple[int, int]):
    start, stop = bounds
    working = [{"kind": "working", "row": row} for row in range(start, stop)]
    persisted = [{"kind": "persisted", "row": row} for row in range(start, stop)]
    return working, persisted


def test_parallel_projection_honors_retain_working_false(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("ordered parallel projection requires fork")
    monkeypatch.setattr(
        build_runtime, "_normalize_and_project_range", _projection_fixture
    )
    first, persisted = build_runtime.normalize_and_project_records_ordered(
        [{}] * 2_000,
        hooks=SimpleNamespace(),
        policy=SimpleNamespace(task_id="runtime-test"),
        workers=2,
        chunk_rows=1_000,
        retain_working=False,
    )
    assert first == persisted
    assert {row["kind"] for row in first} == {"persisted"}
