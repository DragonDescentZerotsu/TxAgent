import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.processing.evidence_library.evidence_library import starling_molecule_id
from tools.chembl_tool.common.json_utils import sha256_file
from tools.chembl_tool.tasks.bbb_martins.build_starling_full_evidence_library import bbb_profiles
from tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library import (
    ALIGNED_V2_OUT_DIR,
    DEFAULT_OUT_DIR,
    SOURCE_PROFILE_BROAD_V1,
    SOURCE_PROFILE_CANONICAL_V3,
    SOURCE_PROFILE_SENSITIZATION_V2,
    HISTORICAL_OUT_DIR,
    _resolve_out_dir,
    _validate_canonical_source,
    skin_reaction_profiles,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.canonical_starling_source import (
    CANONICAL_VERSION,
)


def test_bbb_starling_profiles_cover_the_three_missing_mechanisms():
    profiles = bbb_profiles(Path("data/raw/starling/bbb_martins"))

    assert [profile.group_id for profile in profiles] == [
        "Mechanism.passive_permeability",
        "Mechanism.efflux_transport",
        "Mechanism.influx_transport",
    ]
    assert all(profile.smiles_field == "SMILES" for profile in profiles)


def test_skin_starling_profiles_use_canonical_direct_and_aop_only():
    profiles = skin_reaction_profiles(Path("data/raw/starling/skin_reaction"))

    assert [profile.group_id for profile in profiles] == [
        "Direct.skin_reaction",
        "Mechanism.sensitization_aop",
    ]
    assert all(profile.smiles_field == "SMILES" for profile in profiles)
    assert [profile.evidence_role for profile in profiles] == [
        "direct_outcome",
        "mechanistic_factor",
    ]
    assert profiles[0].record_filter is None
    assert profiles[0].path.endswith("canonical_sensitization_v3/direct_records.parquet")
    assert profiles[1].path.endswith("canonical_sensitization_v3/aop_records.parquet")

    historical = skin_reaction_profiles(
        Path("data/raw/starling/skin_reaction"),
        source_profile=SOURCE_PROFILE_BROAD_V1,
    )
    assert historical[0].record_filter is None
    aligned_v2 = skin_reaction_profiles(
        Path("data/raw/starling/skin_reaction"),
        source_profile=SOURCE_PROFILE_SENSITIZATION_V2,
    )
    assert aligned_v2[0].record_filter is not None
    assert aligned_v2[0].record_filter_name == "is_tdc_skin_sensitization_scope.v1"


def test_starling_molecule_id_is_cross_profile_and_stable():
    assert starling_molecule_id("CCO") == "STARLING_89B394FD02E5E5E6"
    assert starling_molecule_id("CCO") == starling_molecule_id("CCO")


def test_skin_scope_profiles_resolve_to_separate_default_output_roots():
    aligned = SimpleNamespace(
        source_profile=SOURCE_PROFILE_CANONICAL_V3,
        max_rows_per_source=0,
        out_dir="",
    )
    historical = SimpleNamespace(
        source_profile=SOURCE_PROFILE_BROAD_V1,
        max_rows_per_source=0,
        out_dir="",
    )

    assert _resolve_out_dir(aligned) == DEFAULT_OUT_DIR
    assert _resolve_out_dir(historical) == HISTORICAL_OUT_DIR
    assert _resolve_out_dir(
        SimpleNamespace(
            source_profile=SOURCE_PROFILE_SENSITIZATION_V2,
            max_rows_per_source=0,
            out_dir="",
        )
    ) == ALIGNED_V2_OUT_DIR


def test_partial_skin_index_build_requires_isolated_output_root():
    args = SimpleNamespace(
        source_profile=SOURCE_PROFILE_CANONICAL_V3,
        max_rows_per_source=10,
        out_dir="",
    )

    with pytest.raises(SystemExit, match="explicit non-canonical"):
        _resolve_out_dir(args)


def test_canonical_skin_source_preflight_checks_contract_and_hashes(tmp_path):
    canonical = tmp_path / "canonical_sensitization_v3"
    canonical.mkdir()
    direct = canonical / "direct_records.parquet"
    aop = canonical / "aop_records.parquet"
    direct.write_bytes(b"direct")
    aop.write_bytes(b"aop")
    (canonical / "manifest.json").write_text(
        json.dumps(
            {
                "contract_version": CANONICAL_VERSION,
                "stats": {
                    "partition_reconciles": True,
                    "direct_aop_source_record_overlap": 0,
                },
                "paths": {
                    "direct_records_sha256": sha256_file(direct),
                    "aop_records_sha256": sha256_file(aop),
                },
            }
        ),
        encoding="utf-8",
    )

    _validate_canonical_source(tmp_path)
    aop.write_bytes(b"changed")
    with pytest.raises(ValueError, match="failed hash validation"):
        _validate_canonical_source(tmp_path)
