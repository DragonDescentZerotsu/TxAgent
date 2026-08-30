from tools.chembl_tool.common.starling.external_condition import propose_pattern_condition
from tools.chembl_tool.tasks.bbb_martins.prepare_condition_review import (
    RULES as BBB_RULES,
    _remove_non_conditions as _remove_bbb_non_conditions,
)
from tools.chembl_tool.common.starling.external_condition import ConditionAtom
from tools.chembl_tool.tasks.skin_reaction.context_conditioned_benchmark import (
    RULES as SKIN_RULES,
    _remove_query_material_conditions,
)


def _atoms(text: str, rules) -> set[str]:
    return {atom.key for atom in propose_pattern_condition(text, rules=rules).atoms}


def test_stage_iv_is_not_intravenous_route() -> None:
    assert "route=intravenous" not in _atoms("stage IV brain tumor", BBB_RULES)


def test_skin_ethanol_extract_is_not_a_vehicle() -> None:
    assert "vehicle=ethanol" not in _atoms(
        "cardol is a component of the ethanol extract; patch tested at 1%", SKIN_RULES
    )
    assert "vehicle=ethanol" in _atoms("estradiol 1% ethanol", SKIN_RULES)


def test_skin_vehicle_abbreviations_are_supported() -> None:
    assert "vehicle=petrolatum" in _atoms("lidocaine 5% pet.", SKIN_RULES)
    assert "vehicle=aqueous" in _atoms("glycerine 10% aq.", SKIN_RULES)


def test_skin_query_material_is_not_its_own_vehicle_condition() -> None:
    atoms = (ConditionAtom("vehicle", "ethanol"),)
    assert _remove_query_material_conditions(atoms, "CCO") == ()
    sls = (ConditionAtom("coexposure", "sls_barrier_enhancement"),)
    assert _remove_query_material_conditions(
        sls, "CCCCCCCCCCCCOS(=O)(=O)O"
    ) == ()


def test_skin_human_age_groups_are_external_conditions() -> None:
    assert "age_group=pediatric" in _atoms(
        "patch tested in children with dermatitis", SKIN_RULES
    )
    assert "age_group=elderly" in _atoms(
        "5-day patch test in 70-year-old subjects", SKIN_RULES
    )


def test_skin_heterogeneous_prior_sensitizers_are_not_collapsed() -> None:
    assert "population_state=previously_sensitized" not in _atoms(
        "nickel-allergic subjects underwent patch testing", SKIN_RULES
    )
    assert "population_state=healthy_volunteers" in _atoms(
        "human maximization test in healthy volunteers", SKIN_RULES
    )


def test_skin_geriatric_occupation_is_not_elderly_age() -> None:
    assert "age_group=elderly" not in _atoms(
        "occupational airborne dermatitis in a geriatric nurse", SKIN_RULES
    )


def test_skin_alternative_vehicles_are_flagged_as_incompatible() -> None:
    proposal = propose_pattern_condition(
        "tested separately at 10% in water and petrolatum",
        rules=SKIN_RULES,
        incompatible_families={"vehicle"},
    )
    assert proposal.signature is None
    assert proposal.reason == "incompatible_values:vehicle"


def test_bbb_absent_or_self_co_treatment_is_not_a_condition() -> None:
    probenecid = (ConditionAtom("co_treatment", "probenecid"),)
    assert _remove_bbb_non_conditions(probenecid, "OTHER", "without probenecid") == ()
    assert _remove_bbb_non_conditions(
        probenecid,
        "CCCN(CCC)S(=O)(=O)c1ccc(C(=O)O)cc1",
        "free fraction of probenecid",
    ) == ()
