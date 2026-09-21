from __future__ import annotations

import pytest

from semantic_buckets import source_local_semantic_v5 as workflow


@pytest.mark.parametrize("task", ("ames", "dili", "carcinogens"))
def test_semantic_contract_starts_from_endpoint_and_ignores_unit_species(task):
    pair_columns, refinement = workflow._semantic_columns(workflow._contract(task))
    assert set(pair_columns) == set(refinement)
    for source, columns in refinement.items():
        assert columns[0] == "canonical_endpoint_concept"
        assert "canonical_measurement_scale_id" in columns
        assert "canonical_unit_text" not in columns
        assert not any("species" in column or "population" in column for column in columns)
        assert set(columns) <= set(pair_columns[source])


def test_semantic_contract_retains_assay_context_for_every_source():
    for task in ("ames", "dili", "carcinogens"):
        _, refinement = workflow._semantic_columns(workflow._contract(task))
        assert all("canonical_assay_context" in columns for columns in refinement.values())


def test_endpoint_specs_preserve_independent_endpoint_capacity():
    specs = workflow._endpoint_specs(
        ("http://dgx017:50001/v1", "http://dgx020:50002/v1"), 128
    )
    assert [spec["name"] for spec in specs] == ["dgx017:50001", "dgx020:50002"]
    assert [spec["max_inflight"] for spec in specs] == [128, 128]
    assert all(spec["provider"] == "local" for spec in specs)


def test_endpoint_specs_reject_invalid_capacity():
    with pytest.raises(ValueError, match="positive"):
        workflow._endpoint_specs(("http://dgx017:50001/v1",), 0)
