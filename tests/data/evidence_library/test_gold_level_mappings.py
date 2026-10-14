import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from data.processing.gold_labels.level_mappings import level_mapping_release


ROOT = Path("data/gold_labels")
PUBLICATIONS = {
    "Ames": ("v1",),
    "BBB_Martins": ("v1", "v2"),
    "Bioavailability_Ma": ("v1", "v2"),
    "Carcinogens": ("v1",),
    "DILI": ("v1",),
    "Skin_Reaction": ("v1",),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_active_gold_releases_have_receipted_tianang_mappings():
    for task, versions in PUBLICATIONS.items():
        assert (ROOT / task / "CURRENT").read_text().strip() == "v1"
        for version in versions:
            root = ROOT / task / "level_mappings" / version
            manifest = json.loads((root / "manifest.json").read_text())
            assert manifest["status"] == "complete"
            assert manifest["gold_release"] == version
            assert json.loads((root / "alignment_report.json").read_text())["published"]["status"] == "pass"
            for name in ("original_level_mapping", "level_mapping"):
                receipt = manifest["outputs"][name]
                path = root / receipt["path"]
                if receipt["kind"] == "parquet_file":
                    assert digest(path) == receipt["sha256"]
                    assert pq.read_metadata(path).num_rows == receipt["rows"]
                else:
                    rows = 0
                    for part in receipt["parts"]:
                        part_path = path / part["path"]
                        assert digest(part_path) == part["sha256"]
                        rows += pq.read_metadata(part_path).num_rows
                    assert rows == receipt["rows"]
    assert not (ROOT / "legacy/clintox/level_mappings").exists()


def test_gold_level_mapping_resolver_uses_current_v1_and_explicit_v2():
    manifest, mapping, receipt = level_mapping_release("bbb_martins")
    assert manifest == (ROOT / "BBB_Martins/level_mappings/v1/manifest.json").resolve()
    assert mapping == manifest.parent / receipt["path"]

    manifest, mapping, receipt = level_mapping_release("ames")
    assert manifest == (ROOT / "Ames/level_mappings/v2/manifest.json").resolve()
    assert mapping == manifest.parent / receipt["path"]


def test_reviewed_auxiliary_successors_pin_only_l1_to_l2_decisions():
    for task, demotions in {
        "Ames": 202,
        "DILI": 7_913,
        "Carcinogens": 2_054,
    }.items():
        root = ROOT / task / "level_mappings/v2"
        manifest = json.loads((root / "manifest.json").read_text())
        assert (root.parent / "CURRENT").read_text().strip() == "v2"
        assert manifest["gold_release"] == "v1"
        assert manifest["level_mapping_version"] == "v2"
        assert manifest["review"] == {
            "basis": "user_approved_current_voter_exact_l1",
            "decision": "former physical voters move from L1 to L2",
            "promotions": 0,
            "demotions": demotions,
        }
        decisions = root / manifest["outputs"]["reviewed_level_decisions"]["path"]
        assert digest(decisions) == manifest["outputs"]["reviewed_level_decisions"]["sha256"]
        assert pq.read_metadata(decisions).num_rows == demotions
        for part in manifest["outputs"]["level_mapping"]["parts"]:
            path = root / "level_mapping" / part["path"]
            assert digest(path) == part["sha256"]

    manifest, mapping, receipt = level_mapping_release("Bioavailability_Ma", "v2")
    assert manifest == (ROOT / "Bioavailability_Ma/level_mappings/v2/manifest.json").resolve()
    assert mapping == manifest.parent / receipt["path"]


def test_preserved_v1_index_points_to_gold_owned_publications():
    index_path = ROOT / "level_mappings.v1.json"
    index = json.loads(index_path.read_text())
    assert index["status"] == "complete"
    assert set(index["tasks"]) == {
        "ames", "bbb_martins", "bioavailability_ma", "carcinogens", "dili",
        "skin_reaction",
    }
    for receipt in index["tasks"].values():
        manifest = index_path.parent / receipt["manifest"]
        mapping = index_path.parent / receipt["path"]
        assert digest(manifest) == receipt["manifest_sha256"]
        assert mapping.exists()
        assert "data/artifacts" not in str(mapping)

def test_gold_level_mappings_preserve_originals_and_cover_voters():
    for task in ("BBB_Martins", "Bioavailability_Ma"):
        mappings = {}
        for version in ("v1", "v2"):
            root = ROOT / task / "level_mappings" / version
            manifest = json.loads((root / "manifest.json").read_text())
            report = json.loads((root / "alignment_report.json").read_text())
            original_path = root / "original_level_mapping.parquet"
            mapping_path = root / "level_mapping.parquet"
            assert digest(original_path) == manifest["outputs"]["original_level_mapping"]["sha256"]
            assert digest(mapping_path) == manifest["outputs"]["level_mapping"]["sha256"]
            assert digest(original_path) == manifest["upstream"]["sha256"]

            membership = pq.read_table(
                ROOT / task / version / "scaffold/voter_membership.parquet"
            ).to_pandas()
            if "aggregate_status" in membership:
                membership = membership.loc[membership["aggregate_status"].eq("published")]
            voters = set(membership["source_row_uid"])
            mapping = pq.read_table(mapping_path).to_pandas().set_index("source_row_uid")
            mappings[version] = mapping
            assert voters <= set(mapping.index)
            assert set(mapping.loc[list(voters), "level"]) == {1}

            if task == "BBB_Martins":
                assert original_path.read_bytes() == mapping_path.read_bytes()
                continue

            original = pq.read_table(original_path).to_pandas().set_index("source_row_uid")
            review = json.loads(
                (ROOT / task / "level_mappings/v1/reviewed_l1_corrections.json").read_text()
            )
            reviewed = {
                row["source_row_uid"] for row in review["corrections"]
                if row["source_row_uid"] in voters
            }
            added = set(mapping.index) - set(original.index)
            changed = {
                uid for uid in original.index
                if not original.loc[uid].equals(mapping.loc[uid])
            }
            assert reviewed == added | changed
            assert len(added) == report["reviewed_corrections"]["add"]
            assert len(changed) == report["reviewed_corrections"]["promote"]
            unchanged = original.index.difference(list(reviewed))
            pd.testing.assert_frame_equal(original.loc[unchanged], mapping.loc[unchanged])

        l3_v1 = mappings["v1"].loc[mappings["v1"]["level"].ge(3)]
        l3_v2 = mappings["v2"].loc[mappings["v2"]["level"].ge(3)]
        pd.testing.assert_frame_equal(l3_v1, l3_v2)
