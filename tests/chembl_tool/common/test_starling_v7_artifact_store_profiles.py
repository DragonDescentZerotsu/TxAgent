from data.processing.evidence_library.versions.v7.tasks.bbb_martins.build_normalized_starling_evidence_library import (
    ARTIFACT_STAGES as BBB_CANONICAL_STAGES,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.build_normalized_starling_evidence_library import (
    ARTIFACT_STAGES as BIO_CANONICAL_STAGES,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.build_normalized_starling_evidence_library import (
    ARTIFACT_STAGES as SKIN_CANONICAL_STAGES,
)
from tools.chembl_tool.tasks.bbb_martins.starling_v7_artifact_store import (
    PROFILE as BBB_PROFILE,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_v7_artifact_store import (
    PROFILE as BIO_PROFILE,
)
from tools.chembl_tool.tasks.skin_reaction.starling_v7_artifact_store import (
    PROFILE as SKIN_PROFILE,
)

EXPECTED_STAGES = (
    "00_source",
    "01_cleaned",
    "02_canonicalized",
    "03_pair_buckets",
)


def test_all_v7_artifact_profiles_cover_the_complete_stage_layout():
    profiles = (BBB_PROFILE, BIO_PROFILE, SKIN_PROFILE)
    assert {profile.task_id for profile in profiles} == {
        "bbb_martins",
        "bioavailability_ma",
        "skin_reaction",
    }
    for profile in profiles:
        assert profile.stages == EXPECTED_STAGES
        assert profile.local_root.name == "starling_normalized_v7"
        assert profile.tracked_root.name == "starling_normalized_v7"
        assert profile.store_version.endswith(".normalized_v7_store.v4")


def test_current_build_boundary_is_the_three_stage_core():
    assert BBB_CANONICAL_STAGES == EXPECTED_STAGES
    assert BIO_CANONICAL_STAGES == EXPECTED_STAGES
    assert SKIN_CANONICAL_STAGES == EXPECTED_STAGES
