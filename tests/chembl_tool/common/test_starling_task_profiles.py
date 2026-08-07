from pathlib import Path

from tools.chembl_tool.common.starling import starling_molecule_id
from tools.chembl_tool.tasks.bbb_martins.build_starling_full_evidence_library import bbb_profiles
from tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library import skin_reaction_profiles


def test_bbb_starling_profiles_cover_the_three_missing_mechanisms():
    profiles = bbb_profiles(Path("data/starling_data/bbb_martins"))

    assert [profile.group_id for profile in profiles] == [
        "Mechanism.passive_permeability",
        "Mechanism.efflux_transport",
        "Mechanism.influx_transport",
    ]
    assert all(profile.smiles_field == "SMILES" for profile in profiles)


def test_skin_starling_profiles_cover_direct_and_three_mechanisms():
    profiles = skin_reaction_profiles(Path("data/starling_data/skin_reaction"))

    assert [profile.group_id for profile in profiles] == [
        "Direct.skin_reaction",
        "Mechanism.sensitization_aop",
        "Mechanism.phototoxicity_irritation_local_damage",
        "Mechanism.skin_exposure",
    ]
    assert all(profile.smiles_field == "SMILES" for profile in profiles)
    assert [profile.evidence_role for profile in profiles] == [
        "direct_outcome",
        "mechanistic_factor",
        "mechanistic_factor",
        "context_modifier",
    ]


def test_starling_molecule_id_is_cross_profile_and_stable():
    assert starling_molecule_id("CCO") == "STARLING_89B394FD02E5E5E6"
    assert starling_molecule_id("CCO") == starling_molecule_id("CCO")
