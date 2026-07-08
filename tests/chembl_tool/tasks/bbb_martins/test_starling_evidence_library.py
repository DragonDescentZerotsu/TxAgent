from tools.chembl_tool.tasks.bbb_martins.build_starling_evidence_library import (
    GROUP_ID,
    build_starling_bbb_evidence_rows,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    _base_retrieval_groups,
    _merge_tier1_replacement_retrieval,
)


def test_qualitative_mode_omits_quantitative_fields_from_llm_text():
    rows, stats = build_starling_bbb_evidence_rows(
        [
            {
                "smiles": "CCO",
                "pmid": "1",
                "support_text": "Permeable across BBB.",
                "bbb_permeability_label": "permeable",
                "bbb_transport_label": None,
                "quant_metric": "logBB",
                "quant_value": "0.3",
                "quant_units": None,
            },
            {
                "smiles": "CCN",
                "pmid": "2",
                "support_text": "Only numeric.",
                "bbb_permeability_label": None,
                "bbb_transport_label": None,
                "quant_metric": "brain/plasma ratio",
                "quant_value": "0.5",
                "quant_units": "ratio",
            },
        ],
        include_numerical=False,
    )

    assert stats["n_source_rows_kept"] == 1
    assert len(rows) == 1
    assert rows[0]["group_id"] == GROUP_ID
    assert rows[0]["standard_value"] == ""
    assert "0.3" not in rows[0]["assay_description"]
    assert "Quantitative metric/value fields intentionally omitted" in rows[0]["activity_comment"]


def test_all_mode_keeps_quant_only_rows_and_exposes_quantitative_summary():
    rows, stats = build_starling_bbb_evidence_rows(
        [
            {
                "smiles": "CCN",
                "pmid": "2",
                "support_text": "Only numeric.",
                "bbb_permeability_label": None,
                "bbb_transport_label": None,
                "quant_metric": "brain/plasma ratio",
                "quant_value": "0.5",
                "quant_units": "ratio",
            },
        ],
        include_numerical=True,
    )

    assert stats["n_source_rows_kept"] == 1
    assert len(rows) == 1
    assert "brain/plasma ratio" in rows[0]["standard_type"]
    assert rows[0]["standard_value"] == "brain/plasma ratio 0.5 ratio"
    assert "quantitative_value: 0.5" in rows[0]["assay_description"]


def test_tier1_replacement_merge_drops_base_tier1_and_prepends_replacement():
    base_index = {
        "group_to_molecule_indices": {
            "Tier 1.direct_brain_plasma": [0],
            "Tier 2.passive_papp": [1],
        }
    }
    assert _base_retrieval_groups(base_index, tier1_replacement_enabled=True) == ["Tier 2.passive_papp"]
    assert (
        _base_retrieval_groups(
            base_index,
            requested_groups=["Tier 1.starling_direct_bbb_evidence"],
            tier1_replacement_enabled=True,
        )
        == []
    )

    merged = _merge_tier1_replacement_retrieval(
        {
            "status": "ok",
            "query": {},
            "evidence_source": {"type": "chembl"},
            "coverage": {"min_similarity": 0.3, "top_k_per_group": 3, "n_groups": 1},
            "groups": [{"group_id": "Tier 2.passive_papp", "neighbors": [1]}],
        },
        {
            "status": "ok",
            "query": {},
            "evidence_source": {"type": "starling"},
            "coverage": {"n_groups": 1},
            "groups": [{"group_id": GROUP_ID, "neighbors": [2, 3]}],
        },
    )

    assert [group["group_id"] for group in merged["groups"]] == [GROUP_ID, "Tier 2.passive_papp"]
    assert merged["coverage"]["n_neighbors_total"] == 3
    assert merged["evidence_source"]["type"] == "tier1_replaced_source_retrievals"
