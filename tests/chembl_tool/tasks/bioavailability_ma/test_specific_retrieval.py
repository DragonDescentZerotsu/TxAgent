from tools.chembl_tool.tasks.bioavailability_ma.specific_retrieval import (
    FA_GROUP_ID,
    FG_GROUP_ID,
    FH_GROUP_ID,
    OBSERVED_F_GROUP_ID,
    regroup_retrieval_to_specific,
)


def test_regroups_old_tiers_into_observed_f_and_three_factors():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_F", [_row("Bioavailability", "human oral bioavailability")])],
            ),
            _group(
                "Tier 2.absorption_fraction_or_hia",
                [_neighbor("CHEMBL_FA", [_row("Fraction absorbed", "human intestinal absorption")])],
            ),
            _group(
                "Tier 3.cell_bidirectional_efflux_ratio",
                [_neighbor("CHEMBL_FG", [_row("Efflux ratio", "Caco-2 P-gp efflux")])],
            ),
            _group(
                "Tier 5.intrinsic_or_hepatic_clearance",
                [_neighbor("CHEMBL_FH", [_row("CLint", "human liver microsome intrinsic clearance")])],
            ),
        ]
    )

    specific = regroup_retrieval_to_specific(retrieval)
    groups = {group["group_id"]: group for group in specific["groups"]}

    assert set(groups) == {OBSERVED_F_GROUP_ID, FA_GROUP_ID, FG_GROUP_ID, FH_GROUP_ID}
    assert groups[OBSERVED_F_GROUP_ID]["neighbors"][0]["evidence_rows"][0]["specific_evidence_role"] == (
        "direct_absolute_oral_f_candidate"
    )
    assert groups[FA_GROUP_ID]["bioavailability_factor"] == "Fa"
    assert groups[FG_GROUP_ID]["bioavailability_factor"] == "Fg"
    assert groups[FH_GROUP_ID]["bioavailability_factor"] == "Fh"
    assert specific["coverage"]["n_groups_with_neighbors"] == 4


def test_oral_auc_is_observed_exposure_context_not_fa_factor():
    retrieval = _retrieval(
        [
            _group(
                "Tier 2.oral_auc_exposure",
                [_neighbor("CHEMBL_AUC", [_row("AUC", "plasma exposure after oral dose")])],
            )
        ]
    )

    specific = regroup_retrieval_to_specific(retrieval)
    groups = {group["group_id"]: group for group in specific["groups"]}
    observed_rows = groups[OBSERVED_F_GROUP_ID]["neighbors"][0]["evidence_rows"]

    assert observed_rows[0]["specific_evidence_role"] == "oral_exposure_proxy_not_direct_f"
    assert groups[FA_GROUP_ID]["neighbors"] == []
    assert groups[FG_GROUP_ID]["neighbors"] == []
    assert groups[FH_GROUP_ID]["neighbors"] == []


def test_first_pass_can_feed_both_fg_and_fh_when_context_mentions_intestine_and_liver():
    retrieval = _retrieval(
        [
            _group(
                "Tier 5.first_pass_or_extraction",
                [
                    _neighbor(
                        "CHEMBL_FP",
                        [_row("First pass", "intestinal CYP3A and hepatic first-pass extraction")],
                    )
                ],
            )
        ]
    )

    specific = regroup_retrieval_to_specific(retrieval)
    groups = {group["group_id"]: group for group in specific["groups"]}

    assert groups[FG_GROUP_ID]["neighbors"][0]["evidence_rows"][0]["specific_evidence_role"] == (
        "fg_gut_wall_efflux_or_intestinal_metabolism"
    )
    assert groups[FH_GROUP_ID]["neighbors"][0]["evidence_rows"][0]["specific_evidence_role"] == (
        "fh_hepatic_clearance_or_metabolism"
    )


def test_extra_starling_retrieval_is_merged_into_observed_f_group():
    chembl = _retrieval([])
    starling = _retrieval(
        [
            _group(
                "Starling.direct_oral_bioavailability",
                [_neighbor("STARLING_F", [_row("Bioavailability", "reported oral bioavailability was 80%")])],
            )
        ]
    )

    specific = regroup_retrieval_to_specific(chembl, extra_retrievals=[starling])
    groups = {group["group_id"]: group for group in specific["groups"]}

    assert groups[OBSERVED_F_GROUP_ID]["neighbors"][0]["molecule_chembl_id"] == "STARLING_F"
    assert specific["coverage"]["n_groups"] == 4


def _retrieval(groups):
    return {
        "status": "ok",
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "evidence_source": {"type": "test"},
        "coverage": {"n_groups": len(groups), "top_k_per_group": 3, "min_similarity": 0.3},
        "groups": groups,
    }


def _group(group_id, neighbors):
    tier, endpoint_group = group_id.split(".", 1)
    return {
        "group_id": group_id,
        "tier": tier,
        "endpoint_group": endpoint_group,
        "neighbors": neighbors,
    }


def _neighbor(molecule_id, rows):
    return {
        "rank": 1,
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": "CCCO",
        "standard_inchi_key": molecule_id,
        "similarity": 0.82,
        "similarity_bucket": "close_analog",
        "evidence_rows": rows,
    }


def _row(standard_type, assay_description):
    return {
        "assay_chembl_id": "ASSAY1",
        "standard_type": standard_type,
        "standard_relation": "=",
        "standard_value": "50",
        "standard_units": "%",
        "assay_description": assay_description,
        "evidence_source": "test_source",
    }
