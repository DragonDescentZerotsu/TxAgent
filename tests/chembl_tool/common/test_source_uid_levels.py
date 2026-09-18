import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import unittest

from data.processing.evidence_library.build_source_uid_levels import map_uids


EXPECTED_ROWS = {
    "ames": 1_299_584,
    "dili": 1_386_455,
    "carcinogens": 3_151_140,
    "skin_reaction": 66_964,
}

V10_GOLD_V1_ROWS = {
    "bbb_martins": 496_094,
    "bioavailability_ma": 431_501,
}


def test_exact_lineage_disambiguates_repeated_source_record_ids():
    check = unittest.TestCase()
    membership = pd.DataFrame({
        "canonical_record_id": ["a", "b"], "source_id": ["source", "source"],
        "source_record_id": ["ext_1", "ext_1"], "level": [1, 3], "family_key": ["direct", "mechanism"],
    })
    canonical = membership[["canonical_record_id", "source_id", "source_record_id"]].assign(
        cleaned_record_id=["clean_a", "clean_b"], source_row_number=[1, 7],
    )
    ledger = pd.DataFrame({
        "source_id": ["source", "source"], "source_record_id": ["ext_1", "ext_1"],
        "legacy_cleaned_record_id": ["clean_a", "clean_b"],
        "source_row_uid": ["sr_" + "a" * 32, "sr_" + "b" * 32],
        "acquisition_source_row_number": [1, 7],
    })
    aliases = {"source": "source"}
    mapped = map_uids(membership, canonical, ledger, aliases)
    assert dict(zip(mapped.source_row_uid, mapped.level)) == {
        "sr_" + "a" * 32: 1, "sr_" + "b" * 32: 3,
    }
    with check.assertRaisesRegex(ValueError, "missing from the frozen"):
        map_uids(membership, canonical.iloc[:1], ledger, aliases)
    with check.assertRaisesRegex(ValueError, "exact acquisition UID"):
        map_uids(membership, canonical, ledger.iloc[:1], aliases)
    with check.assertRaisesRegex(ValueError, "exact acquisition UID"):
        map_uids(membership, canonical, ledger.assign(acquisition_source_row_number=[2, 8]), aliases)
    with check.assertRaisesRegex(ValueError, "conflicting family/level"):
        map_uids(membership, canonical, ledger.assign(source_row_uid="sr_" + "a" * 32), aliases)
    with check.assertRaises(pd.errors.MergeError):
        map_uids(membership, canonical, pd.concat([ledger, ledger.iloc[:1]]), aliases)


def test_published_sharded_mappings_match_their_manifest():
    root = Path(
        "data/legacy/artifacts/evidence_libraries/"
        "v10_before_v24_1_v25_compatibility_restore_20260916/source_uid_levels"
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["version"] == "source_uid_levels.v3"
    for task, expected_rows in EXPECTED_ROWS.items():
        spec = manifest["tasks"][task]
        assert spec["mapped_rows"] == spec["unique_uids"] == expected_rows
        observed_rows = 0
        for part in spec["parts"]:
            path = root / spec["path"] / part["path"]
            assert path.stat().st_size == part["size_bytes"]
            with path.open("rb") as handle:
                assert hashlib.file_digest(handle, "sha256").hexdigest() == part["sha256"]
            observed_rows += pq.read_metadata(path).num_rows
        assert observed_rows == expected_rows


def test_v10_level_maps_cover_stage3_and_exact_gold_v1_l1():
    public = {
        "bbb_martins": "BBB_Martins",
        "bioavailability_ma": "Bioavailability_Ma",
    }
    for task, expected_rows in V10_GOLD_V1_ROWS.items():
        root = Path(f"data/evidence_libraries/{task}/v10")
        manifest = json.loads((root / "level_mapping/manifest.json").read_text())
        mapping_path = root / "level_mapping" / manifest["output"]["path"]
        with mapping_path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == manifest["output"]["sha256"]
        mapping = pq.read_table(mapping_path, columns=["source_row_uid", "level"]).to_pandas()
        stage3 = pq.read_table(
            root / "03_pair_buckets/records.parquet", columns=["source_row_uid"]
        ).to_pandas()
        voters = pq.read_table(
            Path(f"data/gold_labels/{public[task]}/v1/scaffold/voter_membership.parquet"),
            columns=["source_row_uid"],
        ).to_pandas()
        assert len(mapping) == mapping.source_row_uid.nunique() == expected_rows
        assert set(mapping.source_row_uid) == set(stage3.source_row_uid)
        assert set(mapping.loc[mapping.level.eq(1), "source_row_uid"]) == set(voters.source_row_uid)


if __name__ == "__main__":
    test_exact_lineage_disambiguates_repeated_source_record_ids()
    print("Exact UID lineage and failure-mode checks passed")
