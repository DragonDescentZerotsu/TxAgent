"""Retrieval recall and direct-outcome containment, separate from gold rules."""

import pytest

from tools.chembl_tool.tasks.ames.source_contract import (
    DAMAGE,
    DIRECT,
    MECHANISM,
    MUTATION,
    NEAR,
    classify,
)


def record(**changes):
    return {
        "SMILES": "CCO",
        "support_text": "The compound was tested in this assay.",
        "needs_more_context": False,
        **changes,
    }


@pytest.mark.parametrize("flag", [True, False, None])
@pytest.mark.parametrize(
    "source,fields,group",
    [
        (
            "ames_v1",
            {"assay_family": "micronucleus_assay", "result_call": "equivocal"},
            MUTATION,
        ),
        (
            "ames_v2",
            {
                "endpoint_class": "dna_adduct",
                "result_status": "damage_or_response_decreased",
                "molecule_role": "protective_or_modifying_agent",
            },
            DAMAGE,
        ),
        (
            "ames_v3",
            {
                "mechanism_category": "oxidative_stress_or_antioxidant_defense",
                "result_direction": "mixed_or_condition_dependent",
                "molecule_role": "modifier_or_protectant",
            },
            MECHANISM,
        ),
    ],
)
def test_uncertainty_prediction_and_modifiers_do_not_disqualify_indirect_records(
    flag, source, fields, group
):
    row = record(
        **fields,
        needs_more_context=flag,
        support_text="The predicted endpoint was compared to a protective effect under co-treatment.",
    )
    decision = classify(source, row)
    assert decision.group == group and decision.label is None


@pytest.mark.parametrize("source", ["ames_base", "ames_v1", "ames_v2", "ames_v3"])
@pytest.mark.parametrize(
    "field",
    [
        "support_text",
        "extra_details",
        "qualifying_conditions",
        "assay_method_and_endpoint",
        "endpoint_subtype",
    ],
)
@pytest.mark.parametrize(
    "direct",
    [
        "Predicted Ames-positive",
        "E. coli WP2uvrA response",
        "TA98NR negative",
        "S. typhimurium result",
        "bacterial_reverse_mutation",
        "Ames_test prediction",
    ],
)
def test_direct_content_anywhere_in_a_mixed_card_cannot_enter_later_families(
    source, field, direct
):
    row = record(
        assay_family="micronucleus_assay",
        endpoint_class="dna_adduct",
        mechanism_category="reactive_metabolite_formation",
        **{field: direct},
    )
    decision = classify(source, row)
    assert decision.group == NEAR and decision.label is None


def test_direct_predictions_have_no_vote_even_with_all_gold_condition_fields():
    row = record(
        assay_family="bacterial_reverse_mutation",
        chemical_entity_type="defined_chemical",
        experimental_context="in_vitro",
        evidence_basis="current_study_experiment",
        test_system="TA98",
        metabolic_activation="absent",
        mutagenicity_result="positive",
        support_text="The QSAR model predicted an Ames positive result.",
    )
    assert classify("ames_base", row).group == NEAR
    assert classify("ames_base", row).label is None
    row["support_text"] = "The compound was positive in this experiment."
    assert classify("ames_base", row).group == DIRECT


def test_bacterial_damage_reporter_alone_is_not_a_reverse_mutation_outcome():
    row = record(
        assay_family="bacterial_ddr_reporter",
        endpoint_class="dna_damage_response_signal",
        biological_system="Escherichia coli",
        support_text="A predicted SOS response was reported.",
    )
    assert classify("ames_v2", row).group == DAMAGE
    row["extra_details"] = "The same study also reported an Ames-negative outcome."
    assert classify("ames_v2", row).group == NEAR


def test_source_native_reverse_mutation_enum_is_contained_without_strain_prose():
    row = record(assay_family="non_salmonella_reverse_mutation", result_call="negative")
    assert classify("ames_v1", row).group == NEAR


def test_bacterial_result_after_mammalian_result_is_also_near_direct():
    row = record(
        assay_family="mammalian_cell_gene_mutation",
        support_text=(
            "p-aminophenol caused mutations at the TK locus in mouse lymphoma tests, "
            "although it did not induce point mutations in bacterial test systems."
        ),
    )
    assert classify("ames_base", row).group == NEAR


def test_irreversible_binding_is_damage_not_a_reverse_mutation_call():
    from tools.chembl_tool.tasks.ames.source_contract import has_direct_context

    row = record(
        assay_family="targeted_dna_adduct_measurement",
        endpoint_class="dna_adduct",
        biological_system="Escherichia coli K-12; intact bacterial cells",
        assay_version="irreversible covalent binding to bacterial macromolecules",
        support_text="Photoactivated CPZ binds irreversibly to bacterial DNA.",
    )
    assert classify("ames_v2", row).group == DAMAGE
    assert not has_direct_context("bacterial cells | irreversible covalent binding")


def test_ecoli_nonstandard_reversion_is_near_but_phage_and_yeast_mutations_are_indirect():
    row = record(assay_family="other_mutation_assay")
    row["support_text"] = "E. coli WU3610: reversion of amber and ochre mutations."
    assert classify("ames_base", row).group == NEAR
    row["support_text"] = (
        "Phenotypic reversions in Neurospora and induction of mutations at the ad-3 locus."
    )
    assert classify("ames_base", row).group == MUTATION


def test_ames_in_a_non_latin_passage_cannot_hide_in_other_gene_mutation():
    row = record(
        assay_family="tk_mouse_lymphoma",
        support_text="Ames試験で陰性。マウス細胞の結果も報告。",
    )
    assert classify("ames_v1", row).group == NEAR


def test_tk_mouse_lymphoma_with_mixed_endpoint_enum_is_still_genetic_damage():
    row = record(
        assay_family="tk_mouse_lymphoma", endpoint_class="mixed_fixed_damage_endpoint"
    )
    assert classify("ames_v1", row).group == MUTATION


def test_reverse_transcriptase_or_reversal_of_damage_is_not_a_mutation_outcome():
    row = record(
        assay_family="comet_assay",
        biological_system="bacteria",
        support_text="AMV reverse transcriptase mapped DNA lesions extracted from treated bacteria.",
    )
    assert classify("ames_v2", row).group == DAMAGE
    row = record(
        mechanism_category="oxidative_stress_or_antioxidant_defense",
        support_text="Fermented extract from lactic acid bacteria reversed lipid peroxidation in rat liver.",
    )
    assert classify("ames_v3", row).group == MECHANISM


def test_unknown_endpoint_is_not_rescued_by_a_prediction_flag():
    assert not classify(
        "ames_v3",
        record(mechanism_category="unrelated", support_text="Predicted taste."),
    ).group
    assert not classify(
        "ames_v3", record(SMILES="", mechanism_category="oxidative_stress")
    ).group


@pytest.mark.parametrize("source", ["ames_base", "ames_v1", "ames_v2", "ames_v3"])
def test_noncovalent_base_pair_binding_is_not_a_dna_lesion(source):
    row = record(
        assay_family="dna_binding_assay",
        assay_method_and_endpoint="Noncovalent intercalation at DNA base pairs; binding constant",
        support_text="The binding constant was measured without a lesion assay.",
    )
    assert classify(source, row).group == MECHANISM


@pytest.mark.parametrize("source", ["ames_base", "ames_v1", "ames_v2", "ames_v3"])
@pytest.mark.parametrize(
    "system,group", [("", NEAR), ("Homo sapiens; lymphocytes", MUTATION)]
)
def test_coarse_fixed_damage_enum_does_not_become_dna_damage(source, system, group):
    row = record(
        endpoint_class="mixed_fixed_damage_endpoint", biological_test_system=system
    )
    assert classify(source, row).group == group
    row["extra_details"] = "An Ames-negative result was also reported."
    assert classify(source, row).group == NEAR


@pytest.mark.parametrize("cache_error", [None, ValueError, FileNotFoundError, KeyError])
def test_refresh_reuses_source_only_after_full_hash_validation(
    monkeypatch, cache_error
):
    from tools.chembl_tool.tasks.ames import build_dataset as dataset

    frozen = {
        "benchmark": [],
        "source": [{"path": str(dataset.VOTES), "sha256": "fixed"}],
    }
    rebuilt = []
    published = []
    monkeypatch.setattr(
        dataset, "_read", lambda path: frozen if path == dataset.MANIFEST else {}
    )
    monkeypatch.setattr(dataset, "sha256_file", lambda path: "fixed")
    monkeypatch.setattr(dataset, "_verify", lambda files: None)

    def source_files():
        if cache_error is not None and not rebuilt:
            raise cache_error("source cannot be reused")
        return []

    monkeypatch.setattr(dataset, "_source_files", source_files)
    monkeypatch.setattr(
        dataset.source, "build", lambda **kwargs: rebuilt.append(kwargs)
    )
    monkeypatch.setattr(dataset, "_build_retrieval", lambda workers: {})
    monkeypatch.setattr(
        dataset,
        "write_json_atomic",
        lambda path, value: (
            published.append(value) if path == dataset.MANIFEST else None
        ),
    )
    result = dataset.build(phase="refresh-retrieval", workers=2)
    assert bool(rebuilt) == (cache_error is not None)
    assert result["benchmark"] == frozen["benchmark"]
    assert result["gold_preservation"]["source_votes_sha256"] == "fixed"
    assert published == [result]


def test_unchanged_refresh_preserves_manifest_and_validation_lineage(monkeypatch):
    from tools.chembl_tool.tasks.ames import build_dataset as dataset

    source_files = [{"path": str(dataset.VOTES), "sha256": "fixed"}]
    state = {"manifest": {"benchmark": [], "source": source_files}, "writes": 0}
    monkeypatch.setattr(dataset, "_read", lambda path: state["manifest"])
    monkeypatch.setattr(dataset, "sha256_file", lambda path: "fixed")
    monkeypatch.setattr(dataset, "_verify", lambda files: None)
    monkeypatch.setattr(dataset, "_source_files", lambda: source_files)
    monkeypatch.setattr(
        dataset, "_build_retrieval", lambda workers: {"unchanged": True}
    )

    def publish(path, result):
        state["manifest"] = result
        state["writes"] += 1

    monkeypatch.setattr(dataset, "write_json_atomic", publish)
    first = dataset.build(phase="refresh-retrieval", workers=2)
    second = dataset.build(phase="refresh-retrieval", workers=4)
    assert second is first
    assert state["writes"] == 1
