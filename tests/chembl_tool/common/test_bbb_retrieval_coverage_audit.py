import pytest

from tools.chembl_tool.paper_experiments.audit_bbb_retrieval_coverage import (
    GROUP_NAMES,
    _parse_args,
    _validate_index_provenance,
    summarize_query_rows,
)


def _query(label: int, covered: set[str], *, parity: bool = True) -> dict:
    return {
        "label": label,
        "all_groups_covered": len(covered) == len(GROUP_NAMES),
        "direct_index_full_index_parity": parity,
        "groups": {
            group_id: {
                "n_neighbors": 3 if group_id in covered else 0,
                "top_similarity": 0.5 if group_id in covered else None,
            }
            for group_id in GROUP_NAMES
        },
    }


def test_summarize_query_rows_reports_family_and_label_coverage() -> None:
    group_ids = list(GROUP_NAMES)
    result = summarize_query_rows(
        [
            _query(0, set(group_ids)),
            _query(1, {group_ids[0], group_ids[2]}, parity=False),
        ],
        top_k=3,
    )

    assert result["overall"] == {
        "n_queries": 2,
        "label_counts": {"0": 1, "1": 1},
        "n_all_groups_covered": 1,
        "all_groups_coverage": 0.5,
        "n_direct_parity_failures": 1,
    }
    direct = result["families"]["direct_brain_exposure"]
    passive = result["families"]["passive_permeability"]
    assert direct["coverage"] == 1.0
    assert direct["coverage_by_label"]["1"]["coverage"] == 1.0
    assert passive["coverage"] == 0.5
    assert passive["coverage_by_label"]["1"]["coverage"] == 0.0


def test_subset_specific_paths_are_derived_in_main_not_frozen_by_parser() -> None:
    args = _parse_args(["--evaluation-subset", "test"])
    assert args.input_jsonl == ""
    assert args.output_dir == ""


def test_index_provenance_requires_current_heldout_hash(tmp_path) -> None:
    heldout = tmp_path / "experimental_meaningful_cns_access_v2" / "heldout.jsonl"
    heldout.parent.mkdir()
    heldout.write_text('{"drug":"C"}\n', encoding="utf-8")
    from tools.chembl_tool.common.json_utils import sha256_file

    source = {
        "type": "starling_heldout_parent_filtered_index",
        "zero_parent_overlap": True,
        "heldout_labels_sha256": sha256_file(heldout),
        "heldout_labels_jsonl": str(heldout),
    }
    _validate_index_provenance(
        {"source": source},
        {"source": dict(source)},
        heldout_path=heldout,
        dataset_lineage="experimental_meaningful_cns_access_v2",
    )
    bad = {"source": {**source, "heldout_labels_sha256": "bad"}}
    with pytest.raises(ValueError, match="heldout label hash mismatch"):
        _validate_index_provenance(
            bad,
            {"source": source},
            heldout_path=heldout,
            dataset_lineage="experimental_meaningful_cns_access_v2",
        )
