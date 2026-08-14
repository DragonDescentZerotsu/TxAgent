from tools.chembl_tool.tasks.clintox.clintox_base_benchmark import (
    ALLOWED_CATEGORIES,
    NEGATIVE_CATEGORY,
    label_record,
)


def _row(**updates):
    row = {
        "support_text": "No clinically significant hepatotoxicity was observed.",
        "toxicity_outcome": "no clinically significant hepatotoxicity",
        "toxicity_category": NEGATIVE_CATEGORY,
        "outcome_measure": None,
        "clinical_context": "human clinical trial",
        "dose_or_exposure": None,
        "fda_approval_status": None,
        "approved_indication": None,
        "needs_more_context": False,
        "pmid": "123",
        "extraction_id": "ext_1",
        "SMILES": "CCO",
    }
    row.update(updates)
    return row


def test_declared_categories_map_only_by_controlled_category():
    for category in sorted(ALLOWED_CATEGORIES):
        decision = label_record(_row(toxicity_category=category), source_index=7)
        assert decision.record is not None
        assert decision.record.label == int(category != NEGATIVE_CATEGORY)
        assert decision.record.raw_value == category


def test_adapter_rejects_off_schema_and_incomplete_rows():
    cases = (
        ({"toxicity_category": "ototoxicity"}, "missing_or_invalid_toxicity_category"),
        ({"toxicity_outcome": None}, "missing_toxicity_outcome"),
        ({"needs_more_context": True}, "needs_more_context"),
        ({"support_text": "null"}, "missing_support_text"),
        ({"pmid": ""}, "missing_pmid"),
        ({"SMILES": None}, "missing_source_smiles"),
    )
    for update, reason in cases:
        decision = label_record(_row(**update), source_index=0)
        assert decision.record is None
        assert decision.reason == reason


def test_fda_and_confidence_do_not_infer_or_override_label():
    negative = label_record(
        _row(fda_approval_status="FDA_black_box_warning", confidence=0.01),
        source_index=0,
    )
    positive = label_record(
        _row(
            toxicity_category="hepatotoxicity",
            fda_approval_status="FDA_not_approved",
            confidence=0.01,
        ),
        source_index=1,
    )
    assert negative.record is not None and negative.record.label == 0
    assert positive.record is not None and positive.record.label == 1


def test_source_row_number_makes_duplicate_extraction_ids_unique():
    first = label_record(_row(extraction_id="ext_1"), source_index=3).record
    second = label_record(_row(extraction_id="ext_1"), source_index=4).record
    assert first is not None and second is not None
    assert first.source_record_id != second.source_record_id
    assert first.source_record_id == "row:000000003:ext_1"


def test_missing_qualifying_conditions_is_not_a_synthetic_empty_gate():
    decision = label_record(_row(), source_index=0)
    assert decision.record is not None
