from tools.chembl_tool.common.starling.benchmark_dataset import has_reported_text
from tools.chembl_tool.tasks.clintox.experiment_config import get_source_config
from tools.chembl_tool.tasks.clintox.starling_human_toxicity_benchmark import (
    label_record,
)
from tools.chembl_tool.tasks.clintox.starling_raw import clean_text, route_record


def _human_row(**updates):
    row = {
        "source_id": "organ_specific_toxicity",
        "source_smiles": "CCO",
        "SMILES": "CCO",
        "support_text": "No liver injury was observed.",
        "organ_system": "hepatic",
        "toxicity_endpoint": "liver injury",
        "effect_status": "no_injury_observed",
        "evidence_context": "human_clinical",
        "biological_system": "patients",
        "exposure_regimen": "oral dosing",
        "quantitative_result": None,
        "qualifying_conditions": None,
        "needs_more_context": False,
        "pmid": "123",
        "extraction_id": "record-1",
    }
    row.update(updates)
    return row


def test_basic_null_cleaning_matches_shared_benchmark_semantics():
    for value in (None, "", "  ", "NaN", "none", "NULL", "n/a", "NA"):
        assert clean_text(value) is None
        assert not has_reported_text(value)
    assert clean_text(" unspecified ") == "unspecified"
    assert clean_text(" - ") == "-"
    assert has_reported_text("unspecified")
    assert has_reported_text("-")
    assert clean_text("a\n  b") == "a b"


def test_human_toxicity_adapter_accepts_only_unqualified_explicit_human_rows():
    decision = label_record(_human_row(), source_index=7)
    assert decision.record is not None
    assert decision.record.label == 0
    assert decision.record.smiles == "CCO"

    qualified = label_record(
        _human_row(qualifying_conditions="patients with a susceptible genotype"),
        source_index=8,
    )
    assert qualified.record is None
    assert qualified.reason == "interpretation_altering_qualifying_conditions"

    unresolved = label_record(_human_row(qualifying_conditions="unspecified"), source_index=9)
    assert unresolved.record is None
    assert unresolved.reason == "interpretation_altering_qualifying_conditions"


def test_human_toxicity_adapter_has_no_confidence_gate():
    decision = label_record(_human_row(confidence=0.01), source_index=0)
    assert decision.record is not None


def test_source_routes_keep_direct_human_records_separate_from_context():
    assert route_record(_human_row()) == (
        "clinical_human_safety",
        "Direct.human_organ_toxicity",
        "direct_outcome",
    )
    assert route_record(_human_row(qualifying_conditions="disease model")) == (
        "clinical_human_safety",
        "Context.human_organ_toxicity",
        "context_modifier",
    )
    assert route_record({"source_id": "general_cytotoxicity"}) == (
        "general_cytotoxicity",
        "Mechanism.general_cytotoxicity",
        "mechanistic_factor",
    )


def test_starling_raw_is_an_explicit_nondefault_retrieval_source():
    config = get_source_config("starling_raw")
    assert config.source_name == "starling_raw"
    assert [group.group_id for group in config.direct_groups] == [
        "Direct.human_organ_toxicity"
    ]
    assert len(config.mechanism_groups) == 7
