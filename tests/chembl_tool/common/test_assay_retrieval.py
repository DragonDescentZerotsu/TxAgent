from pathlib import Path

import pandas as pd

from tools.chembl_tool.common.assay_retrieval import (
    FLAT_GROUP_ID,
    MECHANISM_AWARE_RECORD_CARD_SELECTION_VERSION,
    _filter_heldout_direct_records,
    build_family_molecule_prefix_view,
    build_assay_evidence_rows,
    geometric_assay_prefixes,
    retrieve_assay_prefix,
    retrieve_assay_prefixes,
    retrieve_family_molecule_prefixes,
)
from tools.chembl_tool.common.evidence_contract import (
    ASSAY_COMPACT_PROMPT_PROFILE,
    ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
    ASSAY_RAW_CARD_PROMPT_PROFILE,
    evidence_for_group_llm,
)
from tools.chembl_tool.common.json_utils import write_jsonl_atomic
from tools.chembl_tool.common.starling.assay_catalog import assay_id
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
)


def _row(molecule_id, smiles, assay_id, endpoint):
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "assay_chembl_id": assay_id,
        "assay_tier": "Assay",
        "endpoint_group": "assay_ranked_evidence",
        "group_id": f"Assay.{assay_id}",
        "standard_type": endpoint,
        "standard_value": "1",
        "standard_units": "unit",
        "evidence_source": "Starling",
        "assay_description": f"evidence for {endpoint}",
        "source_record_count": 1,
    }


def _index():
    rows = [
        _row("shared", "CCCO", "a1", "endpoint-1"),
        _row("only-a1", "CCCCO", "a1", "endpoint-1"),
        _row("shared", "CCCO", "a2", "endpoint-2"),
        _row("only-a2", "CCCN", "a2", "endpoint-2"),
    ]
    index = build_neighbor_index(rows, index_version="test", workers=1)
    index["assay_ranking"] = [
        {"assay_id": "a1", "assay_context": "assay one", "group_id": "Assay.a1"},
        {"assay_id": "a2", "assay_context": "assay two", "group_id": "Assay.a2"},
    ]
    index["source"] = {
        "type": "starling_assay_ranked_flat",
        "relevance_scores_visible_to_llm": False,
    }
    return index


def test_assay_prefix_is_flat_and_merges_same_molecule_across_assays():
    result = retrieve_assay_prefix(
        "CC",
        _index(),
        assay_prefix=2,
        top_k_per_assay=3,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    assert result["status"] == "ok"
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["group_id"] == FLAT_GROUP_ID
    assert group["evidence_prompt_profile"] == ASSAY_COMPACT_PROMPT_PROFILE
    assert group["n_selected_assays"] == 2
    assert group["n_assay_neighbor_slots"] == 4
    assert group["n_unique_neighbor_molecules"] == 3
    shared = next(
        row for row in group["neighbors"] if row["molecule_chembl_id"] == "shared"
    )
    assert len(shared["evidence_rows"]) == 2
    assert {row["standard_type"] for row in shared["evidence_rows"]} == {
        "endpoint-1",
        "endpoint-2",
    }


def test_assay_prefix_does_not_expose_relevance_scores_to_evidence_prompt_surface():
    result = retrieve_assay_prefix(
        "CC",
        _index(),
        assay_prefix=1,
        top_k_per_assay=3,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    serialized_group = str(result["groups"])
    assert "relevance_score" not in serialized_group
    assert "relevance_rank" not in serialized_group
    assert result["experiment"]["relevance_scores_visible_to_llm"] is False


def test_assay_prefix_uses_index_declared_raw_card_profile():
    index = _index()
    index["source"]["evidence_prompt_profile"] = ASSAY_RAW_CARD_PROMPT_PROFILE

    result = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=1,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )

    assert (
        result["groups"][0]["evidence_prompt_profile"] == ASSAY_RAW_CARD_PROMPT_PROFILE
    )


def test_prefix_selection_is_strictly_nested():
    index = _index()
    top_one = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=1,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    top_two = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=2,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    assert top_one["experiment"]["selected_assay_ids"] == ["a1"]
    assert top_two["experiment"]["selected_assay_ids"] == ["a1", "a2"]
    top_one_evidence = {
        row["standard_type"]
        for neighbor in top_one["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    }
    top_two_evidence = {
        row["standard_type"]
        for neighbor in top_two["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    }
    assert top_one_evidence < top_two_evidence


def test_multi_prefix_retrieval_matches_independent_prefix_results():
    index = _index()
    combined = retrieve_assay_prefixes(
        "CC",
        index,
        prefixes=[1, 2],
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    for prefix in (1, 2):
        independent = retrieve_assay_prefix(
            "CC",
            index,
            assay_prefix=prefix,
            min_similarity=0.0,
            neighbor_identity_policy="operational",
        )
        assert combined[prefix] == independent


def test_assay_start_excludes_earlier_assays_from_every_prefix():
    result = retrieve_assay_prefix(
        "CC",
        _index(),
        assay_start=1,
        assay_prefix=1,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    assert result["experiment"]["assay_start"] == 1
    assert result["experiment"]["selected_assay_ids"] == ["a2"]
    assert {
        row["standard_type"]
        for neighbor in result["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    } == {"endpoint-2"}


def test_record_card_cap_is_global_per_molecule_and_spans_assays():
    index = _index()
    index["source"]["evidence_prompt_profile"] = ASSAY_RAW_CARD_PROMPT_PROFILE
    for molecule_groups in index["evidence_by_molecule_group"].values():
        for group_rows in molecule_groups.values():
            for evidence_row in group_rows:
                endpoint = evidence_row["standard_type"]
                evidence_row["source_record_examples"] = [
                    {
                        "endpoint_type": endpoint,
                        "reported_value": str(card_index),
                        "support_text": f"{endpoint} support {card_index}",
                    }
                    for card_index in range(3)
                ]

    result = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=2,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
        max_record_cards_per_molecule=4,
    )
    group = result["groups"][0]
    for neighbor in group["neighbors"]:
        rendered = [
            evidence_for_group_llm(row, group) for row in neighbor["evidence_rows"]
        ]
        assert sum(len(row["records"]) for row in rendered) <= 4
        assert neighbor["n_prompt_record_cards"] <= 4
    shared = next(
        row for row in group["neighbors"] if row["molecule_chembl_id"] == "shared"
    )
    assert {row["standard_type"] for row in shared["evidence_rows"]} == {
        "endpoint-1",
        "endpoint-2",
    }
    assert shared["n_prompt_record_cards"] == 4


def test_global_molecule_cap_keeps_most_similar_unique_neighbors():
    result = retrieve_assay_prefix(
        "CC",
        _index(),
        assay_prefix=2,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
        max_neighbor_molecules=2,
    )
    neighbors = result["groups"][0]["neighbors"]
    assert len(neighbors) == 2
    assert [row["similarity"] for row in neighbors] == sorted(
        [row["similarity"] for row in neighbors], reverse=True
    )
    assert result["coverage"]["n_candidate_neighbor_molecules_before_cap"] == 3
    assert result["experiment"]["max_neighbor_molecules"] == 2


def test_mechanism_aware_record_cap_keeps_direct_and_indirect_families():
    index = _index()
    index["source"]["evidence_prompt_profile"] = ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE
    shared_rows = index["evidence_by_molecule_group"]["shared"]
    families = {
        "Assay.a1": ("direct_brain_exposure", 1),
        "Assay.a2": ("efflux_transport", 3),
    }
    for group_id, rows in shared_rows.items():
        family, level = families[group_id]
        rows[0]["source_record_examples"] = [
            {
                "endpoint_type": rows[0]["standard_type"],
                "reported_value": "unspecified" if level == 1 else "2.4",
                "reported_units": "" if level == 1 else "efflux ratio",
                "support_text": f"{family} experimental result",
                "evidence_family": family,
                "evidence_family_level": level,
            },
            {
                "endpoint_type": rows[0]["standard_type"],
                "reported_value": "1",
                "support_text": f"second {family} result",
                "evidence_family": family,
                "evidence_family_level": level,
            },
        ]

    result = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=2,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
        max_record_cards_per_molecule=2,
        record_card_selection=MECHANISM_AWARE_RECORD_CARD_SELECTION_VERSION,
    )
    group = result["groups"][0]
    shared = next(
        row for row in group["neighbors"] if row["molecule_chembl_id"] == "shared"
    )
    assert shared["prompt_record_families"] == [
        "direct_brain_exposure",
        "efflux_transport",
    ]
    rendered = [evidence_for_group_llm(row, group) for row in shared["evidence_rows"]]
    assert {card["evidence_family"] for row in rendered for card in row["records"]} == {
        "direct_brain_exposure",
        "efflux_transport",
    }


def test_mechanism_tagged_cards_are_not_visible_before_their_family_level():
    index = _index()
    index["source"]["evidence_prompt_profile"] = ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE
    index["assay_ranking"][0]["first_level"] = 1
    index["assay_ranking"][1]["first_level"] = 3
    row = index["evidence_by_molecule_group"]["shared"]["Assay.a1"][0]
    row["source_record_examples"] = [
        {
            "endpoint_type": "brain exposure",
            "support_text": "direct result",
            "evidence_family": "direct_brain_exposure",
            "evidence_family_level": 1,
        },
        {
            "endpoint_type": "efflux ratio",
            "support_text": "efflux result",
            "evidence_family": "efflux_transport",
            "evidence_family_level": 3,
        },
    ]

    level_one = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=1,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
        max_record_cards_per_molecule=4,
        record_card_selection=MECHANISM_AWARE_RECORD_CARD_SELECTION_VERSION,
    )
    shared = next(
        neighbor
        for neighbor in level_one["groups"][0]["neighbors"]
        if neighbor["molecule_chembl_id"] == "shared"
    )
    assert shared["prompt_record_families"] == ["direct_brain_exposure"]
    assert level_one["experiment"]["max_visible_evidence_family_level"] == 1


def test_family_molecule_prefix_view_removes_per_assay_neighbor_cap():
    rows = [
        _row(f"molecule-{index}", "C" * (index + 2) + "O", "shared-assay", "endpoint")
        for index in range(4)
    ]
    for row in rows:
        row["source_record_examples"] = [
            {
                "endpoint_type": "direct endpoint",
                "reported_value": "1",
                "support_text": f"support for {row['molecule_chembl_id']}",
                "evidence_family": "direct",
                "evidence_family_level": 1,
            }
        ]
    index = build_neighbor_index(rows, index_version="test", workers=1)
    index["source"] = {
        "type": "starling_assay_ranked_flat",
        "evidence_prompt_profile": ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
    }
    view = build_family_molecule_prefix_view(index, levels=[1])

    result = retrieve_family_molecule_prefixes(
        "CCN",
        view,
        levels=[1],
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )[1]

    assert len(result["groups"][0]["neighbors"]) == 4
    assert result["coverage"]["per_assay_neighbor_cap"] is None
    assert result["experiment"]["candidate_generation"] == (
        "global_molecule_similarity_within_cumulative_record_families"
    )


def test_family_molecule_prefix_view_delays_each_record_to_its_own_family():
    index = _index()
    index["source"]["evidence_prompt_profile"] = ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE
    for molecule_groups in index["evidence_by_molecule_group"].values():
        for rows in molecule_groups.values():
            for evidence_row in rows:
                evidence_row["source_record_examples"] = [
                    {
                        "endpoint_type": evidence_row["standard_type"],
                        "support_text": evidence_row["assay_description"],
                        "evidence_family": "direct",
                        "evidence_family_level": 1,
                    }
                ]
    row = index["evidence_by_molecule_group"]["shared"]["Assay.a1"][0]
    row["source_record_examples"] = [
        {
            "endpoint_type": "brain exposure",
            "support_text": "direct support",
            "evidence_family": "direct",
            "evidence_family_level": 1,
        },
        {
            "endpoint_type": "efflux ratio",
            "support_text": "efflux support",
            "evidence_family": "efflux",
            "evidence_family_level": 2,
        },
    ]
    view = build_family_molecule_prefix_view(index, levels=[1, 2])
    results = retrieve_family_molecule_prefixes(
        "CC",
        view,
        levels=[1, 2],
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )

    def supports(level):
        return {
            example["support_text"]
            for neighbor in results[level]["groups"][0]["neighbors"]
            if neighbor["molecule_chembl_id"] == "shared"
            for evidence_row in neighbor["evidence_rows"]
            for example in evidence_row["source_record_examples"]
        }

    assert "direct support" in supports(1)
    assert "efflux support" not in supports(1)
    assert {"direct support", "efflux support"} <= supports(2)


def test_mechanism_tagged_molecule_cap_prefers_direct_on_similarity_tie():
    index = _index()
    index["source"]["evidence_prompt_profile"] = ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE
    index["assay_ranking"][0]["first_level"] = 1
    index["assay_ranking"][1]["first_level"] = 2
    for molecule_rows in index["evidence_by_molecule_group"].values():
        for group_id, rows in molecule_rows.items():
            level = 1 if group_id == "Assay.a1" else 2
            for row in rows:
                row["assay_retrieval"] = {
                    "first_level": level,
                    "first_endpoint_group": "direct" if level == 1 else "mechanism",
                }

    result = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=2,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
        max_neighbor_molecules=1,
    )

    assert result["groups"][0]["neighbors"][0]["molecule_chembl_id"] == "shared"


def test_geometric_assay_prefixes_always_ends_at_full_catalog():
    assert geometric_assay_prefixes(22820) == (5, 20, 80, 320, 1280, 5120, 20480, 22820)
    assert geometric_assay_prefixes(1840) == (5, 20, 80, 320, 1280, 1840)
    assert geometric_assay_prefixes(3) == (3,)


def test_direct_only_filter_keeps_heldout_nondirect_records(tmp_path: Path):
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"drug":"CCO"}\n', encoding="utf-8")
    records = pd.DataFrame(
        [
            {
                "canonical_smiles": "CCO",
                "source_id": "hf_bioavailability",
                "canonical_bioavailability_evidence_scope": "direct",
                "record": "remove",
            },
            {
                "canonical_smiles": "CCO",
                "source_id": "hf_bioavailability",
                "canonical_bioavailability_evidence_scope": "residual",
                "record": "keep-same-source-nondirect",
            },
            {
                "canonical_smiles": "CCO",
                "source_id": "mechanism_source",
                "canonical_bioavailability_evidence_scope": "direct",
                "record": "keep-other-source",
            },
            {
                "canonical_smiles": "CCN",
                "source_id": "hf_bioavailability",
                "canonical_bioavailability_evidence_scope": "direct",
                "record": "keep-nonheldout",
            },
        ]
    )

    filtered, stats = _filter_heldout_direct_records(
        records,
        heldout_molecules_path=heldout,
        heldout_smiles_field="drug",
        filter_source_id="hf_bioavailability",
        filter_scope_field="canonical_bioavailability_evidence_scope",
        filter_scope_value="direct",
    )

    assert set(filtered["record"]) == {
        "keep-same-source-nondirect",
        "keep-other-source",
        "keep-nonheldout",
    }
    assert stats["reference_pool"] == "direct_only_heldout_filtered"
    assert stats["n_direct_heldout_records_excluded"] == 1
    assert stats["n_heldout_nondirect_records_retained"] == 2
    assert stats["n_direct_heldout_records_after_filter"] == 0


def test_direct_only_filter_can_use_purified_group_across_sources(tmp_path: Path):
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"drug":"CCO"}\n', encoding="utf-8")
    records = pd.DataFrame(
        [
            {
                "canonical_smiles": "CCO",
                "source_id": "fg",
                "group_id": "Observed.direct_oral_bioavailability",
                "record": "remove-moved-f",
            },
            {
                "canonical_smiles": "CCO",
                "source_id": "fg",
                "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
                "record": "keep-mechanism",
            },
            {
                "canonical_smiles": "CCN",
                "source_id": "hf_bioavailability",
                "group_id": "Observed.direct_oral_bioavailability",
                "record": "keep-nonheldout",
            },
        ]
    )

    filtered, stats = _filter_heldout_direct_records(
        records,
        heldout_molecules_path=heldout,
        heldout_smiles_field="drug",
        filter_source_id="",
        filter_scope_field="group_id",
        filter_scope_value="Observed.direct_oral_bioavailability",
    )

    assert set(filtered["record"]) == {"keep-mechanism", "keep-nonheldout"}
    assert stats["n_direct_heldout_records_excluded"] == 1
    assert stats["n_heldout_nondirect_records_retained"] == 1


def test_family_catalog_filters_disallowed_source_groups_before_aggregation(
    tmp_path: Path,
):
    records_path = tmp_path / "records.parquet"
    ranked_path = tmp_path / "family_assays.jsonl"
    context = "shared physical assay"
    stable_assay_id = assay_id("skin_reaction", context)
    base = {
        "molecule_id": "M1",
        "canonical_smiles": "CCO",
        "canonical_endpoint_name": "endpoint",
        "canonical_measurement_text": "positive",
        "canonical_unit_text": "",
        "canonical_assay_context": context,
        "canonical_species_context": "human",
        "qualifying_conditions": "",
        "confidence": 0.9,
        "source_name": "Starling",
        "molecule_name": "example",
        "retrieval_eligible": True,
    }
    pd.DataFrame(
        [
            {
                **base,
                "canonical_record_id": "allowed",
                "group_id": "Direct.skin_reaction",
                "support_text": "allowed support",
            },
            {
                **base,
                "canonical_record_id": "disallowed",
                "group_id": "Rejected.irritation",
                "support_text": "disallowed support",
                "confidence": 1.0,
            },
        ]
    ).to_parquet(records_path, index=False)
    write_jsonl_atomic(
        ranked_path,
        [
            {
                "assay_id": stable_assay_id,
                "assay_context": context,
                "selection_rank": 1,
                "first_level": 1,
                "source_groups": ["Direct.skin_reaction"],
            }
        ],
    )

    rows, ranking, stats = build_assay_evidence_rows(
        task="skin_reaction",
        records_path=records_path,
        membership_path=None,
        ranked_assays_path=ranked_path,
    )

    assert len(rows) == 1
    assert rows[0]["source_record_count"] == 1
    assert rows[0]["source_record_examples"][0]["support_text"] == "allowed support"
    assert ranking[0]["assay_id"] == stable_assay_id
    assert stats["family_catalog"] is True
    assert stats["allowed_source_groups"] == ["Direct.skin_reaction"]
