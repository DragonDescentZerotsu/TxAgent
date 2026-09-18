import pytest

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import readout_within_semantic_v4 as workflow
from semantic_buckets import source_local_semantic_v3 as source_local


def test_branch_columns_are_local_consumed_and_scientific(monkeypatch) -> None:
    monkeypatch.setattr(
        core,
        "REFINEMENT_COLUMNS",
        {"source": ("endpoint", "unit", "condition_group")},
    )
    lookup = {
        "a": {"values": {"endpoint": "same", "unit": "x", "condition_group": "1"}},
        "b": {"values": {"endpoint": "same", "unit": "y", "condition_group": "2"}},
    }
    branch = {
        "source_id": "source",
        "consumed_columns": [],
        "atom_ids": ["a", "b"],
    }

    assert workflow._varying_columns(branch, lookup) == ["unit"]
    branch["consumed_columns"] = ["unit"]
    assert workflow._varying_columns(branch, lookup) == []


def test_binding_readout_split_rejects_full_collapse() -> None:
    with pytest.raises(ValueError, match="at least two groups"):
        core._validate_model_response(
            "readout_merge_values",
            {
                "merge_sets": [
                    {
                        "member_ids": ["a", "b"],
                        "label": "one",
                        "rationale": "equivalent",
                    }
                ],
                "rationale": "equivalent",
            },
            {"valid_ids": ["a", "b"], "require_multiple_groups": True},
        )


def test_stratified_samples_are_deterministic_and_cover_rare_values(monkeypatch) -> None:
    monkeypatch.setattr(core, "PROMPT_DIMENSION_COLUMNS", {"source": ("endpoint",)})
    lookup = {
        atom: {
            "source_id": "source",
            "values": {"endpoint": value},
        }
        for atom, value in (("a", "rare"), ("b", "common"), ("c", "common"))
    }

    first = workflow._sample_atom_ids("bucket", ["a", "b", "c"], lookup)
    second = workflow._sample_atom_ids("bucket", ["c", "a", "b"], lookup)

    assert first == second
    assert first[0] == "a"


def test_depth_one_rebuild_uses_every_frozen_semantic_parent(monkeypatch, tmp_path) -> None:
    artifact_root = tmp_path / "artifact"
    monkeypatch.setattr(
        source_local,
        "configure",
        lambda task: {
            "task": "bioavailability_ma",
            "artifact_root": artifact_root,
            "readout_root": artifact_root / "readout_buckets_v3",
        },
    )

    spec = workflow._paths("oral", depth_one=True)
    mapping, output = workflow._repair_inputs(spec, pilot=False)

    assert spec["readout_version"] == workflow.DEPTH_ONE_VERSION
    assert mapping == artifact_root / "semantic_bucket_map.parquet"
    assert output == artifact_root / "readout_buckets_v5_depth1"
