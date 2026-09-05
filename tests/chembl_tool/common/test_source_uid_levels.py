import pandas as pd
import unittest

from data.processing.evidence_library.build_source_uid_levels import map_uids


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
    mapped = map_uids(membership, canonical, ledger)
    assert dict(zip(mapped.source_row_uid, mapped.level)) == {
        "sr_" + "a" * 32: 1, "sr_" + "b" * 32: 3,
    }
    with check.assertRaisesRegex(ValueError, "missing from the frozen"):
        map_uids(membership, canonical.iloc[:1], ledger)
    with check.assertRaisesRegex(ValueError, "no exact acquisition"):
        map_uids(membership, canonical, ledger.iloc[:1])
    with check.assertRaisesRegex(ValueError, "row numbers disagree"):
        map_uids(membership, canonical, ledger.assign(acquisition_source_row_number=[2, 8]))
    with check.assertRaisesRegex(ValueError, "conflicting family/level"):
        map_uids(membership, canonical, ledger.assign(source_row_uid="sr_" + "a" * 32))
    with check.assertRaises(pd.errors.MergeError):
        map_uids(membership, canonical, pd.concat([ledger, ledger.iloc[:1]]))


if __name__ == "__main__":
    test_exact_lineage_disambiguates_repeated_source_record_ids()
    print("Exact UID lineage and failure-mode checks passed")
