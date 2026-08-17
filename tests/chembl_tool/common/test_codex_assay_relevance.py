from tools.chembl_tool.paper_experiments.codex_assay_relevance import score_assay


def _row(task, context, endpoints):
    return {
        "task": task,
        "assay_context": context,
        "assay_id": f"{task}:{context}",
        "endpoints": [{"value": endpoint, "record_count": 1} for endpoint in endpoints],
        "assay_descriptions": [],
    }


def test_bbb_experimental_brain_exposure_outranks_prediction():
    direct = score_assay(_row("bbb_martins", "in vivo brain uptake", ["brain_to_plasma_ratio"]))
    predicted = score_assay(_row("bbb_martins", "in silico prediction", ["logbb"]))
    assert direct["relevance_score"] >= 94
    assert predicted["relevance_score"] <= 20


def test_bioavailability_endpoint_fallback_is_direct_not_computational():
    row = score_assay(
        _row(
            "bioavailability_ma",
            "endpoint_fallback::oral_bioavailability",
            ["oral_bioavailability"],
        )
    )
    assert row["relevance_score"] == 99


def test_skin_validated_sensitization_outranks_irritation_and_diffusion():
    llna = score_assay(_row("skin_reaction", "llna", ["stimulation_index"]))
    irritation = score_assay(_row("skin_reaction", "irritation assay", ["skin_irritation"]))
    diffusion = score_assay(_row("skin_reaction", "in vitro diffusion cell", ["flux"]))
    assert llna["relevance_score"] == 96
    assert irritation["relevance_score"] == 15
    assert diffusion["relevance_score"] == 22


def test_bioavailability_generic_context_does_not_inherit_high_endpoint_score():
    generic = score_assay(_row("bioavailability_ma", "animal study", ["intestinal_absorption"]))
    specific = score_assay(
        _row("bioavailability_ma", "human intestinal absorption", ["intestinal_absorption"])
    )
    assert generic["relevance_score"] == 64
    assert specific["relevance_score"] == 85


def test_in_vivo_is_not_mistaken_for_intravenous_abbreviation():
    oral = score_assay(
        _row("bioavailability_ma", "in vivo oral study", ["auc", "oral_exposure"])
    )
    oral_iv = score_assay(
        _row("bioavailability_ma", "oral vs IV pharmacokinetics", ["auc"])
    )
    assert oral["relevance_score"] == 90
    assert oral_iv["relevance_score"] == 96


def test_photo_llna_is_not_ranked_as_ordinary_skin_sensitization():
    photo = score_assay(_row("skin_reaction", "photo-llna", ["skin_sensitization"]))
    ordinary = score_assay(_row("skin_reaction", "llna", ["skin_sensitization"]))
    assert photo["relevance_score"] == 18
    assert ordinary["relevance_score"] == 96

    assert score_assay(_row("skin_reaction", "UV-LLNA", ["sensitization"]))[
        "relevance_score"
    ] == 18


def test_skin_prediction_and_generic_context_do_not_inherit_direct_assay_score():
    prediction = score_assay(_row("skin_reaction", "ADMETlab", ["sensitization"]))
    generic = score_assay(_row("skin_reaction", "assessment", ["sensitization"]))
    explicit = score_assay(
        _row("skin_reaction", "animal sensitization study", ["sensitization"])
    )
    assert prediction["relevance_score"] == 20
    assert generic["relevance_score"] == 72
    assert explicit["relevance_score"] == 88


def test_clintox_direct_outcome_outranks_nonclinical_and_cytotoxicity():
    direct_row = _row(
        "clintox",
        "clinical_trial_failure::hepatotoxicity",
        ["development terminated due to toxicity"],
    )
    direct_row["source_id"] = "clinical_trial_failure"
    animal_row = _row(
        "clintox", "nonclinical_in_vivo_toxicity::mortality_or_survival", ["mortality"]
    )
    animal_row["source_id"] = "nonclinical_in_vivo_toxicity"
    cell_row = _row("clintox", "general_cytotoxicity::mtt", ["ic50"])
    cell_row["source_id"] = "general_cytotoxicity"
    direct = score_assay(direct_row)
    animal = score_assay(animal_row)
    cell = score_assay(cell_row)
    assert direct["relevance_score"] == 100
    assert animal["relevance_score"] == 74
    assert cell["relevance_score"] == 42


def test_clintox_toxicity_absent_is_relevant_but_not_positive_failure():
    row = _row(
        "clintox",
        "clinical_trial_failure::toxicity_absent",
        ["study terminated for lack of efficacy, not toxicity; well tolerated"],
    )
    row["source_id"] = "clinical_trial_failure"
    scored = score_assay(row)
    assert scored["relevance_score"] == 94
    assert "不是毒性失败阳性结局" in scored["rationale"]
