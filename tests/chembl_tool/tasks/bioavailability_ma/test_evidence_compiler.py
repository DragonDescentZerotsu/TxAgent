from tools.chembl_tool.tasks.bioavailability_ma.evidence_compiler import compile_final_evidence


def test_direct_f_with_moderate_transferability_allows_direct_vote():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor(
                        "CHEMBL1",
                        [
                            _row(
                                standard_type="Bioavailability",
                                standard_value="55",
                                standard_units="%",
                            )
                        ],
                    )
                ],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
        )
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)

    assert compiled["direct_label_votes"][0]["group_id"] == "Tier 1.direct_absolute_bioavailability"
    card = compiled["group_cards"][0]
    assert card["endpoint_role"] == "direct_oral_f"
    assert card["direct_label_vote_allowed"] is True
    assert card["threshold_evidence"][0]["threshold_relation"] == "above_or_equal_high_threshold"


def test_low_transferability_direct_f_is_blocked_from_direct_vote():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL1", [_row(standard_type="Bioavailability", standard_value="60")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="low",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
        )
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)

    assert compiled["direct_label_votes"] == []
    blocked = compiled["transferability_blocked_direct_or_numeric_evidence"][0]
    assert blocked["reason"] == "transferability gate blocks direct label vote"


def test_clearance_numeric_values_are_not_treated_as_f_percent_thresholds():
    retrieval = _retrieval(
        [
            _group(
                "Tier 5.intrinsic_or_hepatic_clearance",
                [
                    _neighbor(
                        "CHEMBL_CL",
                        [
                            _row(
                                standard_type="CL",
                                standard_value="4.2",
                                standard_units="mL/min/kg",
                            ),
                            _row(
                                standard_type="Recovery",
                                standard_value="97",
                                standard_units="%",
                            ),
                        ],
                    )
                ],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 5.intrinsic_or_hepatic_clearance",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="first_pass_or_clearance_risk",
        )
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)

    card = compiled["group_cards"][0]
    assert card["threshold_evidence"] == []
    assert compiled["direct_label_votes"] == []


def test_llm_decision_view_blocks_numeric_direction_values_but_keeps_context():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL1", [_row(standard_type="Bioavailability", standard_value="60")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="low",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
        )
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)
    blocked_for_llm = compiled["llm_decision_view"]["blocked_numeric_evidence_policy"]["blocked_group_summaries"][0]

    assert "threshold_evidence" not in blocked_for_llm
    assert "compiler_threshold_direction" not in blocked_for_llm
    assert "reasoning_summary" in blocked_for_llm
    assert "caveats" in blocked_for_llm
    assert blocked_for_llm["n_numeric_values"] == 1


def test_blocked_numeric_summary_retains_group_context_without_structured_vote_fields():
    retrieval = _retrieval(
        [
            _group(
                "Tier 2.absorption_fraction_or_hia",
                [_neighbor("CHEMBL1", [_row(standard_type="Fa", standard_value="0.06", standard_units="fraction")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 2.absorption_fraction_or_hia",
            transferability="low",
            confidence="low",
            useful=True,
            direction="argues_against_high_bioavailability",
            reasoning_summary="The single neighbor has Fa=0.06 (6% oral absorption in humans).",
            caveats=["Do not overuse the 6% value."],
        )
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)
    blocked_for_llm = compiled["llm_decision_view"]["blocked_numeric_evidence_policy"]["blocked_group_summaries"][0]
    serialized = str(blocked_for_llm)

    assert "threshold_evidence" not in blocked_for_llm
    assert "compiler_threshold_direction" not in blocked_for_llm
    assert "0.06" in serialized
    assert "6%" in serialized
    assert "oral absorption" in serialized
    assert "reasoning_summary" in blocked_for_llm
    assert "caveats" in blocked_for_llm
    assert blocked_for_llm["n_numeric_values"] == 1
    assert blocked_for_llm["block_reason"] == "endpoint is not direct absolute oral bioavailability"


def test_auc_and_papp_are_proxy_evidence_not_direct_votes():
    retrieval = _retrieval(
        [
            _group(
                "Tier 2.oral_auc_exposure",
                [_neighbor("CHEMBL_AUC", [_row(standard_type="AUC", standard_value="1000", standard_units="h*nM")])],
            ),
            _group(
                "Tier 3.cell_permeability_papp",
                [_neighbor("CHEMBL_PAPP", [_row(standard_type="Papp", standard_value="12", standard_units="10^-6 cm/s")])],
            ),
        ]
    )
    group_outputs = [
        _group_output("Tier 2.oral_auc_exposure", transferability="high", useful=True, direction="absorption_support"),
        _group_output("Tier 3.cell_permeability_papp", transferability="high", useful=True, direction="permeability_support"),
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)
    roles = {card["group_id"]: card["endpoint_role"] for card in compiled["group_cards"]}

    assert compiled["direct_label_votes"] == []
    assert roles["Tier 2.oral_auc_exposure"] == "oral_exposure"
    assert roles["Tier 3.cell_permeability_papp"] == "in_vitro_permeability"
    assert {card["bioavailability_factor"] for card in compiled["proxy_evidence_cards"]} == {
        "contextual_F_or_exposure",
        "Fa_absorption_permeability_efflux",
    }


def test_threshold_boundary_and_borderline_zone_are_explicit():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor(
                        "CHEMBL_BORDER",
                        [
                            _row(standard_type="Bioavailability", standard_value="20"),
                            _row(standard_type="Bioavailability", standard_value="19.9"),
                        ],
                    )
                ],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="moderate",
            useful=True,
            direction="neutral_or_unclear",
        )
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)
    evidence = compiled["group_cards"][0]["threshold_evidence"]

    assert evidence[0]["threshold_relation"] == "above_or_equal_high_threshold"
    assert evidence[1]["threshold_relation"] == "below_high_threshold"
    assert all(item["borderline_to_20_percent"] for item in evidence)


def test_f_fraction_values_are_converted_to_percent():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.context_dependent",
                [_neighbor("CHEMBL_FRACTION", [_row(standard_type="F_fraction", standard_value="0.48", standard_units="")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.context_dependent",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
        )
    ]

    compiled = compile_final_evidence(retrieval, _single_output(), group_outputs)
    card = compiled["group_cards"][0]

    assert card["threshold_evidence"][0]["value_percent"] == 48.0
    assert card["threshold_summary"]["compiler_threshold_direction"] == "supports_high_bioavailability"


def _retrieval(groups):
    return {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "coverage": {"n_groups_with_neighbors": len(groups)},
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
        "similarity": 0.82,
        "similarity_bucket": "close_analog",
        "evidence_rows": rows,
    }


def _row(standard_type, standard_value, standard_units="%"):
    return {
        "molecule_chembl_id": "CHEMBL1",
        "assay_chembl_id": "ASSAY1",
        "standard_type": standard_type,
        "standard_relation": "=",
        "standard_value": standard_value,
        "standard_units": standard_units,
        "assay_description": f"{standard_type} assay",
        "evidence_source": "test_source",
    }


def _single_output():
    return {
        "status": "ok",
        "llm": {
            "content": {
                "oral_bioavailability_prior": "mixed_or_unclear",
                "confidence": "low",
            }
        },
    }


def _group_output(
    group_id,
    *,
    transferability,
    useful=True,
    confidence="moderate",
    direction="neutral_or_unclear",
    reasoning_summary="test reasoning",
    caveats=None,
):
    tier, endpoint_group = group_id.split(".", 1)
    return {
        "group_id": group_id,
        "tier": tier,
        "endpoint_group": endpoint_group,
        "status": "ok",
        "llm": {
            "content": {
                "useful_for_bioavailability_reasoning": useful,
                "transferability": transferability,
                "confidence": confidence,
                "evidence_direction": direction,
                "reasoning_summary": reasoning_summary,
                "key_evidence": [],
                "caveats": caveats or [],
            }
        },
    }
