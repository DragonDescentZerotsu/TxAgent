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
    "01_cleaned",
    "02_canonicalized",
    "03_records",
    "04_pair_buckets",
    "05_distance_calibration",
    "06_remove_heldout_overlap",
    "07_molecule_evidence",
    "08_neighbor_index",
    "09_audits",
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
        assert profile.store_version.endswith(".normalized_v7_store.v1")
