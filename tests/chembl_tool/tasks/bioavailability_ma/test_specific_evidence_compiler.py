import json

from tools.chembl_tool.tasks.bioavailability_ma.specific_evidence_compiler import (
    build_specific_decision_policy,
    compile_specific_final_evidence,
)


def test_specific_compiler_folds_absorption_and_metabolism_into_factors():
    retrieval = _retrieval(
        [
            _group(
                "Tier 2.absorption_fraction_or_hia",
                [_neighbor("CHEMBL_FA", [_row("Fa", "0.95", "fraction")])],
            ),
            _group(
                "Tier 5.metabolic_stability",
                [_neighbor("CHEMBL_MET", [_row("intrinsic clearance", "120", "uL/min/mg")])],
            ),
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 2.absorption_fraction_or_hia",
            transferability="high",
            direction="supports_high_bioavailability",
            reasoning_summary="Human fraction absorbed is high.",
        ),
        _group_output(
            "Tier 5.metabolic_stability",
            transferability="high",
            direction="argues_against_high_bioavailability",
            reasoning_summary="High hepatic clearance and CYP first-pass metabolism lower parent exposure.",
        ),
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)

    assert {card["group_id"] for card in compiled["factor_evidence"]["Fa"]} == {"Tier 2.absorption_fraction_or_hia"}
    assert {card["group_id"] for card in compiled["factor_evidence"]["Fh"]} == {"Tier 5.metabolic_stability"}
    assert compiled["llm_decision_view"]["factor_model"]["equation"] == "F = Fa * Fg * Fh"
    assert "first_pass_or_clearance_mechanism" in compiled["context_flags"]


def test_transporter_and_gut_metabolism_can_map_to_fg():
    retrieval = _retrieval(
        [
            _group(
                "Tier 3.efflux_transporter",
                [_neighbor("CHEMBL_EFFLUX", [_row("P-gp efflux", "2.0", "ratio")])],
            ),
            _group(
                "Tier 5.clearance_first_pass",
                [_neighbor("CHEMBL_GUT", [_row("first pass", "60", "%")])],
            ),
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 3.efflux_transporter",
            transferability="moderate",
            direction="argues_against_high_bioavailability",
            reasoning_summary="P-gp intestinal efflux may lower gut availability.",
        ),
        _group_output(
            "Tier 5.clearance_first_pass",
            transferability="moderate",
            direction="argues_against_high_bioavailability",
            reasoning_summary="Gut wall CYP3A presystemic first-pass metabolism is strong.",
        ),
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)

    fg_groups = {card["group_id"] for card in compiled["factor_evidence"]["Fg"]}
    assert fg_groups == {"Tier 3.efflux_transporter", "Tier 5.clearance_first_pass"}


def test_mechanism_transfer_alerts_are_visible_as_soft_transfer_guards():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_SOFT_HIGH", [_row("Bioavailability", "60", "%")])],
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
            reasoning_summary="Moderate-transfer analog direct-F context supports high.",
            key_evidence=[{"molecule_chembl_id": "CHEMBL_SOFT_HIGH", "transferability": "moderate"}],
        )
    ]
    single_output = _single_output(
        oral_bioavailability_prior="low",
        absorption_prior="unfavorable",
        metabolism_or_clearance_prior="unfavorable",
        reasoning_summary=(
            "The molecule is a quaternary ammonium cation with permanent positive charge and a carboxylic "
            "ester that may act as a prodrug or active metabolite liability."
        ),
        property_drivers=[
            "permanent quaternary ammonium cation (low permeability)",
            "carboxylic ester hydrolysis may form active metabolite",
        ],
    )

    compiled = compile_specific_final_evidence(retrieval, single_output, group_outputs)
    alerts = compiled["mechanism_transfer_alerts"]
    llm_alerts = compiled["llm_decision_view"]["mechanism_transfer_alerts"]

    assert "permanent_quaternary_charge" in alerts["present_alerts"]
    assert "ester_prodrug_or_active_moiety" in alerts["present_alerts"]
    assert "permanent_quaternary_charge" in llm_alerts["transfer_guards"]
    assert any(
        "eligible clean direct-F anchor" in rule
        for rule in compiled["llm_decision_view"]["decision_rules"]
    )
    assert "source_transfer_arbitration" not in compiled["llm_decision_view"]


def test_direct_f_with_active_metabolite_language_requires_parent_analyte_review():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_PRODRUG", [_row("Bioavailability", "70", "%")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
            reasoning_summary="The 70% value measures active metabolite exposure after oral prodrug dosing, not unchanged parent F.",
            caveats=["Measured analyte differs from parent drug."],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    direct = compiled["direct_f_analog_evidence"]["parent_analyte_review_required"][0]

    assert direct["parent_applicability"] == "requires_parent_analyte_review"
    assert "prodrug_or_active_metabolite" in direct["context_flags"]
    assert "prodrug_or_active_metabolite" in compiled["context_flags"]
    assert compiled["direct_f_analog_evidence"]["clean_parent_like_analog_f"] == []
    gate = compiled["evidence_gate_summary"]
    assert gate["clean_direct_label_vote_available"] is False
    assert gate["review_required_label_vote_policy"] == "forbidden_as_direct_label_vote"
    assert "must not be the main reason" in gate["final_reasoning_contract"]
    source_summary = compiled["source_level_direct_f_summary"]
    assert source_summary["n_review_required_context_only_sources"] == 1
    assert source_summary["n_eligible_clean_direct_vote_sources"] == 0
    assert source_summary["review_context_source_direction_counts"] == {"supports_high_bioavailability": 1}

    llm_view = compiled["llm_decision_view"]
    llm_direct = llm_view["direct_f_analog_evidence"]["parent_analyte_review_required"][0]
    assert direct["threshold_evidence"][0]["value_percent"] == 70.0
    assert direct["compiler_threshold_direction"] == "supports_high_bioavailability"
    assert llm_direct["threshold_evidence"] == []
    assert llm_direct["threshold_summary"]["redacted"] is True
    assert llm_direct["compiler_threshold_direction"] == "redacted_review_required_context_only"
    assert (
        llm_view["evidence_state"]["review_required_direct_f_threshold_direction_counts"]
        == "redacted_from_llm_view"
    )
    llm_source_summary = llm_view["source_level_direct_f_summary"]
    assert llm_source_summary["review_context_source_direction_counts"] == "redacted_from_llm_view"
    llm_source = llm_source_summary["source_summaries"][0]
    assert llm_source["source_direction"] == "redacted_review_required_context_only"
    assert llm_source["min_value_percent"] == "redacted"
    assert llm_view["contextual_evidence"] == []
    assert "70%" not in json.dumps(llm_view, sort_keys=True)


def test_species_and_threshold_flags_are_exposed_for_final_decision():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_DOG", [_row("Bioavailability", "19.2", "%")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="argues_against_high_bioavailability",
            reasoning_summary="Dog F is 19.2% and may reflect species-specific first-pass metabolism.",
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)

    assert "species_translation" in compiled["context_flags"]
    assert "threshold_sensitive_15_25_percent" in compiled["context_flags"]
    direct = compiled["direct_f_analog_evidence"]["clean_parent_like_analog_f"][0]
    assert direct["threshold_summary"]["n_borderline_15_to_25_percent"] == 1


def test_clean_parent_f_language_does_not_force_parent_analyte_review():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_PARENT", [_row("Bioavailability", "64", "%")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
            reasoning_summary="This is clean direct parent F evidence for a source analog; no direct parent F measurement exists for the query.",
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)

    assert len(compiled["direct_f_analog_evidence"]["clean_parent_like_analog_f"]) == 1
    assert compiled["direct_f_analog_evidence"]["parent_analyte_review_required"] == []
    assert compiled["source_level_direct_f_summary"]["n_eligible_clean_direct_vote_sources"] == 1


def test_source_level_direct_f_uses_per_source_transferability_gate():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                    [
                        _neighbor(
                            "CHEMBL_CLOSE",
                            [
                                _row(
                                    "Bioavailability",
                                    "38",
                                    "%",
                                    source_molecule="CHEMBL_CLOSE",
                                    standard_inchi_key="INCHI_CLOSE",
                                )
                            ],
                        ),
                        _neighbor(
                            "CHEMBL_DISTANT",
                            [
                                _row(
                                    "Bioavailability",
                                    "82",
                                    "%",
                                    source_molecule="CHEMBL_DISTANT",
                                    standard_inchi_key="INCHI_DISTANT",
                                )
                            ],
                        ),
                    ],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="high",
            useful=True,
            direction="supports_high_bioavailability",
            key_evidence=[
                {"molecule_chembl_id": "CHEMBL_CLOSE", "transferability": "high"},
                {"molecule_chembl_id": "CHEMBL_DISTANT", "transferability": "low"},
            ],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source_summary = compiled["source_level_direct_f_summary"]
    source_by_id = {
        item["source_molecule_ids"][0]: item for item in source_summary["source_summaries"]
    }

    assert source_summary["n_eligible_clean_direct_vote_sources"] == 1
    assert source_summary["n_clean_but_not_direct_vote_sources"] == 1
    assert source_by_id["CHEMBL_CLOSE"]["label_vote_bucket"] == "eligible_clean_direct_vote"
    assert source_by_id["CHEMBL_DISTANT"]["label_vote_bucket"] == "clean_but_not_direct_vote"
    llm_source = {
        item["source_molecule_ids"][0]: item
        for item in compiled["llm_decision_view"]["source_level_direct_f_summary"]["source_summaries"]
    }
    assert llm_source["CHEMBL_DISTANT"]["source_direction"] == "supports_high_bioavailability"
    assert llm_source["CHEMBL_DISTANT"]["min_value_percent"] == 82.0
    assert llm_source["CHEMBL_DISTANT"]["soft_context_policy"] == "context_only_not_direct_label_vote"


def test_moderate_direct_f_source_is_context_not_direct_vote():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor(
                        "CHEMBL_MODERATE",
                        [
                            _row(
                                "Bioavailability",
                                "64",
                                "%",
                                source_molecule="CHEMBL_MODERATE",
                                standard_inchi_key="INCHI_MODERATE",
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
            key_evidence=[
                {"molecule_chembl_id": "CHEMBL_MODERATE", "transferability": "moderate"},
            ],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source_summary = compiled["source_level_direct_f_summary"]

    assert source_summary["n_eligible_clean_direct_vote_sources"] == 0
    assert source_summary["n_clean_but_not_direct_vote_sources"] == 1
    source = source_summary["source_summaries"][0]
    assert source["source_direction"] == "supports_high_bioavailability"
    assert source["label_vote_bucket"] == "clean_but_not_direct_vote"
    soft_context = compiled["llm_decision_view"]["evidence_gate_summary"]["soft_direct_f_context"]
    assert soft_context["n_soft_clean_direct_f_sources"] == 1
    assert soft_context["soft_clean_source_direction_counts"] == {"supports_high_bioavailability": 1}
    assert soft_context["source_summaries"][0]["soft_context_policy"] == "context_only_not_direct_label_vote"


def test_source_level_direct_f_uses_range_crossing_as_threshold_straddling():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor(
                        "STARLING_RANGE",
                        [
                            {
                                **_row(
                                    "Bioavailability",
                                    "19.2",
                                    "%",
                                    source_molecule="STARLING_RANGE",
                                    standard_inchi_key="INCHI_RANGE",
                                ),
                                "source_value_min_percent": 13.4,
                                "source_value_median_percent": 19.2,
                                "source_value_max_percent": 25.0,
                            }
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
            confidence="high",
            useful=True,
            direction="neutral_or_unclear",
            key_evidence=[
                {"molecule_chembl_id": "STARLING_RANGE", "transferability": "high"},
            ],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source_summary = compiled["source_level_direct_f_summary"]
    source = source_summary["source_summaries"][0]

    assert source_summary["source_level_consensus"] == "eligible_clean_sources_threshold_straddling"
    assert source["source_direction"] == "mixed_threshold_straddling"
    assert source["n_range_crosses_20_percent"] == 1
    assert source["min_value_percent"] == 13.4
    assert source["max_value_percent"] == 25.0


def test_factor_effect_overrides_caveat_risk_words_for_signal():
    retrieval = _retrieval(
        [
            _group(
                "Fa.absorption_solubility_permeability",
                [_neighbor("CHEMBL_FA", [_row("Fa", "0.95", "fraction")])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Fa.absorption_solubility_permeability",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="absorption_support",
            factor_effect="supports_higher_F",
            reasoning_summary="Absorption is favorable for this analog.",
            caveats=["Fa alone does not remove Fg or Fh risk."],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)

    card = compiled["factor_evidence"]["Fa"][0]
    assert card["factor_effect"] == "supports_higher_f"
    assert card["signal"] == "supports_higher_F"


def test_factor_product_guard_blocks_high_from_mixed_direct_without_fg_fh_bounds():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor(
                        "CHEMBL_LOW",
                        [
                            _row(
                                "Bioavailability",
                                "16",
                                "%",
                                source_molecule="CHEMBL_LOW",
                                standard_inchi_key="INCHI_LOW",
                            )
                        ],
                    ),
                    _neighbor(
                        "CHEMBL_HIGH",
                        [
                            _row(
                                "Bioavailability",
                                "50",
                                "%",
                                source_molecule="CHEMBL_HIGH",
                                standard_inchi_key="INCHI_HIGH",
                            )
                        ],
                    ),
                ],
            ),
            _group("Fa.absorption_solubility_permeability", [_neighbor("CHEMBL_FA", [_row("Fa", "1", "fraction")])]),
            _group("Fh.hepatic_clearance_metabolic_stability", [_neighbor("CHEMBL_FH", [_row("CL", "80", "mL/min/kg")])]),
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="high",
            useful=True,
            key_evidence=[
                {"molecule_chembl_id": "CHEMBL_LOW", "transferability": "high"},
                {"molecule_chembl_id": "CHEMBL_HIGH", "transferability": "high"},
            ],
        ),
        _group_output(
            "Fa.absorption_solubility_permeability",
            transferability="high",
            confidence="high",
            direction="absorption_support",
            factor_effect="supports_higher_F",
        ),
        _group_output(
            "Fh.hepatic_clearance_metabolic_stability",
            transferability="moderate",
            confidence="moderate",
            direction="first_pass_or_clearance_risk",
            factor_effect="risk_for_lower_F",
        ),
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    constraints = compiled["evidence_gate_summary"]["factor_product_guard"]["class_constraints"]

    assert "high" in constraints
    assert "blocked_mixed_direct_without_explicit_fg_fh_bounds" in constraints["high"]


def test_factor_product_guard_allows_high_for_mixed_direct_without_limiting_factors():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor("CHEMBL_LOW", [_row("Bioavailability", "19.2", "%", source_molecule="CHEMBL_LOW")]),
                    _neighbor("CHEMBL_HIGH", [_row("Bioavailability", "38", "%", source_molecule="CHEMBL_HIGH")]),
                ],
            ),
            _group("Fa.absorption_solubility_permeability", [_neighbor("CHEMBL_FA", [_row("Fa", "1", "fraction")])]),
            _group("Fh.hepatic_clearance_metabolic_stability", [_neighbor("CHEMBL_FH", [_row("CL", "7", "mL/min/kg")])]),
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="high",
            useful=True,
            key_evidence=[
                {"molecule_chembl_id": "CHEMBL_LOW", "transferability": "high"},
                {"molecule_chembl_id": "CHEMBL_HIGH", "transferability": "high"},
            ],
        ),
        _group_output(
            "Fa.absorption_solubility_permeability",
            transferability="high",
            confidence="high",
            direction="absorption_support",
            factor_effect="supports_higher_F",
        ),
        _group_output(
            "Fh.hepatic_clearance_metabolic_stability",
            transferability="moderate",
            confidence="low",
            direction="neutral_or_unclear",
            factor_effect="mixed_or_context",
        ),
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    constraints = compiled["evidence_gate_summary"]["factor_product_guard"]["class_constraints"]

    assert "high" not in constraints


def test_factor_product_guard_blocks_high_without_direct_when_low_prior_and_no_fh_support():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_CONTEXT_LOW", [_row("Bioavailability", "14", "%")])],
            ),
            _group("Fa.absorption_solubility_permeability", [_neighbor("CHEMBL_FA", [_row("Fa", "1", "fraction")])]),
            _group("Fg.gut_wall_efflux_intestinal_metabolism", [_neighbor("CHEMBL_FG", [_row("P-gp efflux", "0.8", "ratio")])]),
            _group("Fh.hepatic_clearance_metabolic_stability", [_neighbor("CHEMBL_FH", [_row("CL", "80", "mL/min/kg")])]),
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="argues_against_high_bioavailability",
            key_evidence=[{"molecule_chembl_id": "CHEMBL_CONTEXT_LOW", "transferability": "moderate"}],
        ),
        _group_output(
            "Fa.absorption_solubility_permeability",
            transferability="moderate",
            confidence="moderate",
            direction="absorption_support",
            factor_effect="supports_higher_F",
        ),
        _group_output(
            "Fg.gut_wall_efflux_intestinal_metabolism",
            transferability="moderate",
            confidence="moderate",
            direction="supports_high_bioavailability",
            factor_effect="supports_higher_F",
        ),
        _group_output(
            "Fh.hepatic_clearance_metabolic_stability",
            transferability="low",
            confidence="low",
            direction="neutral_or_unclear",
            factor_effect="mixed_or_context",
        ),
    ]
    single_output = _single_output(
        oral_bioavailability_prior="low",
        solubility_or_dissolution_prior="unfavorable",
        metabolism_or_clearance_prior="unfavorable",
    )

    compiled = compile_specific_final_evidence(retrieval, single_output, group_outputs)
    constraints = compiled["evidence_gate_summary"]["factor_product_guard"]["class_constraints"]

    assert "high" in constraints
    assert "blocked_no_direct_low_property_prior_without_fh_support" in constraints["high"]


def test_soft_high_direct_f_context_counters_low_prior_high_block():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor("CHEMBL_H1", [_row("Bioavailability", "44", "%", source_molecule="CHEMBL_H1", standard_inchi_key="INCHI_H1")]),
                    _neighbor("CHEMBL_H2", [_row("Bioavailability", "58", "%", source_molecule="CHEMBL_H2", standard_inchi_key="INCHI_H2")]),
                    _neighbor("CHEMBL_H3", [_row("Bioavailability", "71", "%", source_molecule="CHEMBL_H3", standard_inchi_key="INCHI_H3")]),
                ],
            ),
            _group("Fa.absorption_solubility_permeability", [_neighbor("CHEMBL_FA", [_row("Fa", "1", "fraction")])]),
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
            key_evidence=[
                {"molecule_chembl_id": "CHEMBL_H1", "transferability": "moderate"},
                {"molecule_chembl_id": "CHEMBL_H2", "transferability": "moderate"},
                {"molecule_chembl_id": "CHEMBL_H3", "transferability": "moderate"},
            ],
        ),
        _group_output(
            "Fa.absorption_solubility_permeability",
            transferability="moderate",
            confidence="moderate",
            direction="absorption_support",
            factor_effect="supports_higher_F",
        ),
    ]
    single_output = _single_output(
        oral_bioavailability_prior="low",
        solubility_or_dissolution_prior="unfavorable",
        metabolism_or_clearance_prior="unfavorable",
    )

    compiled = compile_specific_final_evidence(retrieval, single_output, group_outputs)
    guard = compiled["llm_decision_view"]["factor_product_guard"]
    policy = compiled["llm_decision_view"]["evidence_gate_summary"]["deterministic_decision_policy"]

    assert "high" not in guard["class_constraints"]
    assert guard["soft_high_context_counterweight"] is True
    assert policy["recommended_class"] == "high"
    assert policy["recommendation_strength"] == "moderate"
    assert policy["decision_state"] == "fallback_uncertain"
    assert policy["forced_class"] is None
    assert policy["fallback_candidate_class"] == "high"
    assert "soft_clean_direct_f_context_leans_high_without_strong_limiting_evidence" in policy["reason_codes"]


def test_policy_can_recommend_low_when_soft_context_and_limiting_prior_align():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor("CHEMBL_L1", [_row("Bioavailability", "9", "%", source_molecule="CHEMBL_L1", standard_inchi_key="INCHI_L1")]),
                    _neighbor("CHEMBL_L2", [_row("Bioavailability", "18", "%", source_molecule="CHEMBL_L2", standard_inchi_key="INCHI_L2")]),
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
            direction="argues_against_high_bioavailability",
            key_evidence=[
                {"molecule_chembl_id": "CHEMBL_L1", "transferability": "moderate"},
                {"molecule_chembl_id": "CHEMBL_L2", "transferability": "moderate"},
            ],
        )
    ]
    single_output = _single_output(
        oral_bioavailability_prior="low",
        absorption_prior="unfavorable",
        solubility_or_dissolution_prior="unfavorable",
    )

    compiled = compile_specific_final_evidence(retrieval, single_output, group_outputs)
    policy = build_specific_decision_policy(compiled)

    assert policy["recommended_class"] == "low"
    assert policy["recommendation_strength"] == "moderate"
    assert policy["decision_state"] == "fallback_uncertain"
    assert policy["forced_class"] is None
    assert policy["fallback_candidate_class"] == "low"
    assert "soft_clean_direct_f_context_leans_low_with_limiting_evidence" in policy["reason_codes"]


def test_threshold_straddling_soft_context_is_uncertainty_not_low_recommendation():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor("CHEMBL_STRADDLE", [
                        {
                            **_row("Bioavailability", "19", "%", source_molecule="CHEMBL_STRADDLE", standard_inchi_key="INCHI_STRADDLE"),
                            "source_value_min_percent": 8.0,
                            "source_value_max_percent": 25.0,
                        }
                    ]),
                    _neighbor("CHEMBL_LOW", [_row("Bioavailability", "12", "%", source_molecule="CHEMBL_LOW", standard_inchi_key="INCHI_LOW")]),
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
            direction="argues_against_high_bioavailability",
            key_evidence=[
                {"molecule_chembl_id": "CHEMBL_STRADDLE", "transferability": "moderate"},
                {"molecule_chembl_id": "CHEMBL_LOW", "transferability": "moderate"},
            ],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(absorption_prior="unfavorable"), group_outputs)
    soft = compiled["evidence_gate_summary"]["soft_direct_f_context"]
    policy = build_specific_decision_policy(compiled)

    assert soft["net_soft_context_direction"] == "soft_context_threshold_sensitive_mixed"
    assert policy["recommended_class"] is None
    assert "soft_clean_direct_f_threshold_sensitive_mixed" in policy["uncertainty_flags"]


def test_reactive_metabolites_do_not_trigger_active_metabolite_scope_review():
    row = {
        **_row("Bioavailability", "64", "%", source_molecule="CHEMBL_REACTIVE"),
        "source_record_examples": [
            {
                "bioavailability_report_type": "absolute",
                "oral_exposure_mode": "oral",
                "species_or_population": "Human",
                "support_text": "The parent drug has 64% oral bioavailability; reactive metabolites were monitored separately.",
            }
        ],
    }
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_REACTIVE", [row])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="high",
            useful=True,
            direction="supports_high_bioavailability",
            key_evidence=[{"molecule_chembl_id": "CHEMBL_REACTIVE", "transferability": "high"}],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source = compiled["source_level_direct_f_summary"]["source_summaries"][0]

    assert source["label_vote_bucket"] == "eligible_clean_direct_vote"
    assert source["parent_applicabilities"] == ["clean_parent_f"]
    assert source["parent_scope_reasons"] == []


def test_source_scope_context_summary_deduplicates_merge_and_raw_keys():
    row = {
        **_row(
            "Bioavailability",
            "72",
            "%",
            source_molecule="CHEMBL_ALIAS",
            standard_inchi_key="INCHI_ALIAS",
        ),
        "source_record_examples": [
            {
                "bioavailability_report_type": "absolute",
                "support_text": "Active metabolite exposure after prodrug dosing was 72%.",
            }
        ],
    }
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_ALIAS", [row])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="high",
            useful=True,
            direction="supports_high_bioavailability",
            key_evidence=[{"molecule_chembl_id": "CHEMBL_ALIAS", "transferability": "high"}],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    scope_summary = compiled["source_scope_context_summary"]

    assert scope_summary["summary_scope"] == "direct_f_groups_merge_key_deduplicated"
    assert scope_summary["n_group_source_scope_contexts"] == 1
    assert scope_summary["scope_review_reason_counts"] == {
        "active_metabolite_prodrug_or_total_radioactivity_scope": 1
    }


def test_llm_view_hides_diagnostic_class_constraints():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_CONTEXT_LOW", [_row("Bioavailability", "14", "%")])],
            ),
            _group("Fa.absorption_solubility_permeability", [_neighbor("CHEMBL_FA", [_row("Fa", "1", "fraction")])]),
            _group("Fg.gut_wall_efflux_intestinal_metabolism", [_neighbor("CHEMBL_FG", [_row("P-gp efflux", "0.8", "ratio")])]),
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="moderate",
            confidence="moderate",
            useful=True,
            direction="argues_against_high_bioavailability",
            key_evidence=[{"molecule_chembl_id": "CHEMBL_CONTEXT_LOW", "transferability": "moderate"}],
        ),
        _group_output(
            "Fa.absorption_solubility_permeability",
            transferability="moderate",
            confidence="moderate",
            direction="absorption_support",
            factor_effect="supports_higher_F",
        ),
        _group_output(
            "Fg.gut_wall_efflux_intestinal_metabolism",
            transferability="moderate",
            confidence="moderate",
            direction="supports_high_bioavailability",
            factor_effect="supports_higher_F",
        ),
    ]
    single_output = _single_output(
        oral_bioavailability_prior="low",
        solubility_or_dissolution_prior="unfavorable",
        metabolism_or_clearance_prior="unfavorable",
    )

    compiled = compile_specific_final_evidence(retrieval, single_output, group_outputs)
    full_guard = compiled["evidence_gate_summary"]["factor_product_guard"]
    llm_gate = compiled["llm_decision_view"]["evidence_gate_summary"]

    assert "high" in full_guard["class_constraints"]
    assert llm_gate["factor_product_guard"]["class_constraints"] == "hidden_from_llm_diagnostic_only"
    assert llm_gate["deterministic_decision_policy"]["diagnostic_blocked_classes"] == "hidden_from_llm_diagnostic_only"


def test_factor_product_guard_blocks_low_from_moderate_factor_only_risks():
    retrieval = _retrieval(
        [
            _group("Fa.absorption_solubility_permeability", [_neighbor("CHEMBL_FA", [_row("solubility", "3", "ug/mL")])]),
            _group("Fg.gut_wall_efflux_intestinal_metabolism", [_neighbor("CHEMBL_FG", [_row("CYP3A4", "1", "qual")])]),
            _group("Fh.hepatic_clearance_metabolic_stability", [_neighbor("CHEMBL_FH", [_row("CLint", "300", "uL/min")])]),
        ]
    )
    group_outputs = [
        _group_output(
            "Fa.absorption_solubility_permeability",
            transferability="moderate",
            confidence="moderate",
            direction="solubility_risk",
            factor_effect="risk_for_lower_F",
        ),
        _group_output(
            "Fg.gut_wall_efflux_intestinal_metabolism",
            transferability="moderate",
            confidence="moderate",
            direction="first_pass_or_clearance_risk",
            factor_effect="risk_for_lower_F",
        ),
        _group_output(
            "Fh.hepatic_clearance_metabolic_stability",
            transferability="moderate",
            confidence="moderate",
            direction="first_pass_or_clearance_risk",
            factor_effect="risk_for_lower_F",
        ),
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    constraints = compiled["evidence_gate_summary"]["factor_product_guard"]["class_constraints"]

    assert "low" in constraints
    assert "blocked_factor_only_moderate_analog_risks" in constraints["low"]


def test_source_level_direct_f_decorrelates_threshold_straddling_same_source():
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [
                    _neighbor(
                        "CHEMBL_MIXED",
                        [
                            _row("Bioavailability", "10", "%", source_molecule="CHEMBL_MIXED"),
                            _row("Bioavailability", "80", "%", source_molecule="CHEMBL_MIXED"),
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
            direction="supports_high_bioavailability",
            reasoning_summary="Same source has conflicting reported oral F values.",
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source_summary = compiled["source_level_direct_f_summary"]

    assert source_summary["n_source_molecules_with_direct_f_values"] == 1
    assert source_summary["source_level_consensus"] == "eligible_clean_sources_threshold_straddling"
    source = source_summary["source_summaries"][0]
    assert source["source_direction"] == "mixed_threshold_straddling"
    assert source["n_values"] == 2


def test_source_row_active_metabolite_scope_blocks_source_level_direct_vote():
    row = {
        **_row("Bioavailability", "72", "%", source_molecule="CHEMBL_PRODRUG"),
        "source_record_examples": [
            {
                "bioavailability_report_type": "absolute",
                "oral_exposure_mode": "oral",
                "species_or_population": "Human",
                "support_text": (
                    "Oral prodrug dosing produced 72% bioavailability based on active metabolite exposure; "
                    "the measured analyte differs from unchanged parent."
                ),
            }
        ],
    }
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_PRODRUG", [row])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="high",
            useful=True,
            direction="supports_high_bioavailability",
            reasoning_summary="Group summary says this is direct F evidence.",
            key_evidence=[{"molecule_chembl_id": "CHEMBL_PRODRUG", "transferability": "high"}],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source_summary = compiled["source_level_direct_f_summary"]
    source = source_summary["source_summaries"][0]

    assert source_summary["n_eligible_clean_direct_vote_sources"] == 0
    assert source_summary["n_review_required_context_only_sources"] == 1
    assert source["label_vote_bucket"] == "review_required_context_only"
    assert source["parent_applicabilities"] == ["requires_parent_analyte_review"]
    assert source["parent_scope_reasons"] == ["active_metabolite_prodrug_or_total_radioactivity_scope"]
    assert source["source_transfer_scope_classes"] == ["active_metabolite_or_prodrug_scope"]
    transfer_summary = compiled["source_transfer_summary"]
    assert transfer_summary["status"] == "audit_only_not_used_for_v14_1_label_policy"
    assert transfer_summary["scope_class_counts"] == {"active_metabolite_or_prodrug_scope": 1}
    assert transfer_summary["context_only_source_classes"]["active_metabolite_or_prodrug_scope"][0][
        "source_direction"
    ] == "supports_high_bioavailability"
    assert "source_transfer_arbitration" not in compiled["llm_decision_view"]
    llm_source = compiled["llm_decision_view"]["source_level_direct_f_summary"]["source_summaries"][0]
    assert llm_source["source_direction"] == "redacted_review_required_context_only"
    assert llm_source["min_value_percent"] == "redacted"


def test_source_row_formulation_specific_absolute_f_is_context_only():
    row = {
        **_row("Bioavailability", "45", "%", source_molecule="STARLING_FORMULATION"),
        "evidence_source": "starling-labs/Oral_Bioavailability",
        "source_record_examples": [
            {
                "bioavailability_report_type": "absolute",
                "oral_exposure_mode": "oral",
                "species_or_population": "dogs",
                "qualifying_conditions": "self-emulsified drug delivery system",
                "comparator": "vs oral solution",
                "support_text": (
                    "Absolute oral bioavailability was 45% for the self-emulsified formulation in dogs."
                ),
            }
        ],
    }
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("STARLING_FORMULATION", [row])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="high",
            useful=True,
            direction="supports_high_bioavailability",
            reasoning_summary="Group summary says this is clean direct parent F.",
            key_evidence=[{"molecule_chembl_id": "STARLING_FORMULATION", "transferability": "high"}],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source_summary = compiled["source_level_direct_f_summary"]
    source = source_summary["source_summaries"][0]

    assert source_summary["n_eligible_clean_direct_vote_sources"] == 0
    assert source_summary["n_review_required_context_only_sources"] == 1
    assert source["parent_applicabilities"] == ["requires_context_scope_review"]
    assert source["parent_scope_reasons"] == ["formulation_or_salt_specific_scope"]
    assert source["source_transfer_scope_classes"] == ["special_formulation_or_route_context"]
    assert compiled["source_transfer_summary"]["scope_class_counts"] == {
        "special_formulation_or_route_context": 1
    }
    assert "source_transfer_arbitration" not in compiled["llm_decision_view"]
    assert source["label_vote_bucket"] == "review_required_context_only"


def test_source_transfer_summary_marks_ordinary_salt_without_changing_v14_gate():
    row = {
        **_row("Bioavailability", "60", "%", source_molecule="CHEMBL_SALT"),
        "source_record_examples": [
            {
                "bioavailability_report_type": "absolute",
                "oral_exposure_mode": "oral",
                "species_or_population": "Human",
                "support_text": "The amlodipine besylate salt form has 60% absolute oral bioavailability.",
            }
        ],
    }
    retrieval = _retrieval(
        [
            _group(
                "Tier 1.direct_absolute_bioavailability",
                [_neighbor("CHEMBL_SALT", [row])],
            )
        ]
    )
    group_outputs = [
        _group_output(
            "Tier 1.direct_absolute_bioavailability",
            transferability="high",
            confidence="moderate",
            useful=True,
            direction="supports_high_bioavailability",
            key_evidence=[{"molecule_chembl_id": "CHEMBL_SALT", "transferability": "high"}],
        )
    ]

    compiled = compile_specific_final_evidence(retrieval, _single_output(), group_outputs)
    source = compiled["source_level_direct_f_summary"]["source_summaries"][0]
    transfer_summary = compiled["source_transfer_summary"]

    assert source["label_vote_bucket"] == "review_required_context_only"
    assert source["parent_scope_reasons"] == ["formulation_or_salt_specific_scope"]
    assert source["source_transfer_scope_classes"] == ["same_active_moiety_salt_or_freebase_context"]
    assert transfer_summary["scope_class_counts"] == {"same_active_moiety_salt_or_freebase_context": 1}
    assert transfer_summary["anchor_candidate_sources"][0]["source_transfer_scope_classes"] == [
        "same_active_moiety_salt_or_freebase_context"
    ]
    assert transfer_summary["anchor_candidate_sources"][0]["label_vote_bucket"] == "review_required_context_only"
    assert "source_transfer_arbitration" not in compiled["llm_decision_view"]


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


def _row(
    standard_type,
    standard_value,
    standard_units="%",
    *,
    source_molecule="CHEMBL1",
    standard_inchi_key=None,
):
    return {
        "molecule_chembl_id": source_molecule,
        "assay_chembl_id": "ASSAY1",
        "standard_type": standard_type,
        "standard_relation": "=",
        "standard_value": standard_value,
        "standard_units": standard_units,
        "assay_description": f"{standard_type} assay",
        "evidence_source": "test_source",
        **({"standard_inchi_key": standard_inchi_key} if standard_inchi_key else {}),
    }


def _single_output(
    *,
    oral_bioavailability_prior="mixed_or_unclear",
    absorption_prior=None,
    solubility_or_dissolution_prior=None,
    metabolism_or_clearance_prior=None,
    reasoning_summary="",
    property_drivers=None,
    caveats=None,
):
    return {
        "status": "ok",
        "llm": {
            "content": {
                "oral_bioavailability_prior": oral_bioavailability_prior,
                "absorption_prior": absorption_prior,
                "solubility_or_dissolution_prior": solubility_or_dissolution_prior,
                "metabolism_or_clearance_prior": metabolism_or_clearance_prior,
                "confidence": "low",
                "reasoning_summary": reasoning_summary,
                "property_drivers": property_drivers or [],
                "caveats": caveats or [],
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
    factor_effect=None,
    reasoning_summary="test reasoning",
    caveats=None,
    key_evidence=None,
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
                **({"factor_effect": factor_effect} if factor_effect is not None else {}),
                "reasoning_summary": reasoning_summary,
                "key_evidence": key_evidence or [],
                "caveats": caveats or [],
            }
        },
    }
