from tools.chembl_tool.tasks.bbb_martins.audit_passive_distance_candidates import (
    CANDIDATES,
    classify_activity_row,
)


def _row(**updates):
    row = {
        "target_chembl_id": "CHEMBL333",
        "confidence_score": 9,
        "relationship_type": "D",
        "description": "Inhibition of human recombinant MMP2 using fluorogenic substrate",
        "standard_type": "IC50",
        "canonical_smiles": "CCO",
        "data_validity_comment": "",
        "potential_duplicate": 0,
    }
    row.update(updates)
    return row


def test_direct_mmp2_functional_activity_is_candidate():
    family, reason = classify_activity_row(_row())
    assert family == "mmp2_activity"
    assert reason == "direct_functional_target_activity"


def test_rock2_functional_kinase_assay_is_candidate():
    family, _ = classify_activity_row(
        _row(
            target_chembl_id="CHEMBL2973",
            description="Inhibition of human ROCK2 kinase using peptide substrate and ATP",
        )
    )
    assert family == "rock2_activity"


def test_mylk_binding_only_assay_is_excluded():
    family, reason = classify_activity_row(
        _row(
            target_chembl_id="CHEMBL2428",
            description="Binding affinity to human MYLK by KINOMEscan",
            standard_type="Kd",
        )
    )
    assert family is None
    assert reason == "endpoint_not_functional_activity"


def test_binding_word_excludes_even_ki_row():
    family, reason = classify_activity_row(
        _row(description="Binding affinity toward MMP2", standard_type="Ki")
    )
    assert family is None
    assert reason == "binding_only_assay"


def test_base_quality_flags_are_excluded():
    family, reason = classify_activity_row(_row(data_validity_comment="Outside typical range"))
    assert family is None
    assert reason == "data_validity_flag"


def test_mmp14_is_not_publishable_as_h2_after_shortcut_audit():
    specs = {spec.family_id: spec for spec in CANDIDATES}
    assert specs["mmp14_activity"].declared_level == "H2"
    assert specs["mmp14_activity"].mechanism_status == "rejected_h2_shortcut_risk"


def test_h1_literature_status_is_separate_from_engineering_gate():
    supported_h1 = {
        spec.family_id
        for spec in CANDIDATES
        if spec.declared_level == "H1" and spec.mechanism_status == "supported"
    }
    # MYLK and RhoA still require the engineering coverage gate; the full audit
    # rejects them.  This unit assertion freezes only the literature status.
    assert supported_h1 == {"mmp2_activity", "rock2_activity", "mylk_activity", "rhoa_activity"}
