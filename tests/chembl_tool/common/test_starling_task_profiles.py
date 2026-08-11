from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.chembl_tool.common.starling import starling_molecule_id
from tools.chembl_tool.tasks.bbb_martins.build_starling_full_evidence_library import bbb_profiles
from tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library import (
    DEFAULT_OUT_DIR,
    DIRECT_SCOPE_BROAD_V1,
    DIRECT_SCOPE_SENSITIZATION_V2,
    HISTORICAL_OUT_DIR,
    _resolve_out_dir,
    skin_reaction_profiles,
)


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
    assert profiles[0].record_filter is not None
    assert profiles[0].record_filter_name == "is_tdc_skin_sensitization_scope.v1"

    historical = skin_reaction_profiles(
        Path("data/starling_data/skin_reaction"),
        direct_scope=DIRECT_SCOPE_BROAD_V1,
    )
    assert historical[0].record_filter is None


def test_starling_molecule_id_is_cross_profile_and_stable():
    assert starling_molecule_id("CCO") == "STARLING_89B394FD02E5E5E6"
    assert starling_molecule_id("CCO") == starling_molecule_id("CCO")


def test_skin_scope_profiles_resolve_to_separate_default_output_roots():
    aligned = SimpleNamespace(
        direct_scope=DIRECT_SCOPE_SENSITIZATION_V2,
        max_rows_per_source=0,
        out_dir="",
    )
    historical = SimpleNamespace(
        direct_scope=DIRECT_SCOPE_BROAD_V1,
        max_rows_per_source=0,
        out_dir="",
    )

    assert _resolve_out_dir(aligned) == DEFAULT_OUT_DIR
    assert _resolve_out_dir(historical) == HISTORICAL_OUT_DIR


def test_partial_skin_index_build_requires_isolated_output_root():
    args = SimpleNamespace(
        direct_scope=DIRECT_SCOPE_SENSITIZATION_V2,
        max_rows_per_source=10,
        out_dir="",
    )

    with pytest.raises(SystemExit, match="explicit non-canonical"):
        _resolve_out_dir(args)
