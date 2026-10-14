import json
import pandas as pd
import pytest
from optimization.semantic_buckets.publication import _reviewed_changes
from optimization.semantic_buckets import publication


def test_reviewed_merge_and_split_are_atom_level():
    mapping = pd.DataFrame(
        [
            {
                "level": "L2",
                "source_id": "s",
                "semantic_bucket_id": bucket,
                "atom_id": atom,
            }
            for bucket, atom in (
                ("small_a", "a"),
                ("small_b", "b"),
                ("large", "c"),
                ("large", "d"),
            )
        ]
    )
    sizes = pd.DataFrame(
        [
            {"semantic_bucket_id": bucket, "record_count": count}
            for bucket, count in (("small_a", 1), ("small_b", 99), ("large", 100))
        ]
    )
    proposals = pd.DataFrame(
        [{"member_bucket_ids_json": json.dumps(["small_a", "small_b"])}]
    )
    flags = pd.DataFrame([{"semantic_bucket_id": "large", "risk": "high"}])
    review = {
        "merges": [
            {
                "member_bucket_ids": ["small_a", "small_b"],
                "rationale": "same biological activity",
            }
        ],
        "splits": [
            {
                "old_bucket_id": "large",
                "atom_groups": [["c"], ["d"]],
                "rationale": "distinct activities",
            }
        ],
    }
    replacement, counts = _reviewed_changes(mapping, sizes, review, proposals, flags)
    assert replacement["a"] == replacement["b"]
    assert replacement["c"] != replacement["d"]
    assert counts == {"merged_families": 1, "split_families": 1, "changed_atoms": 4}
    review["merges"][0]["member_bucket_ids"] = ["small_a", "large"]
    with pytest.raises(ValueError, match="unreviewed merge"):
        _reviewed_changes(mapping, sizes, review, proposals, flags)


def test_publication_pins_payloads_and_rejects_foreign_weights(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    pd.DataFrame([dict(level="L2", atom_id="a", semantic_bucket_id="b")]).to_parquet(
        root / "map.parquet"
    )
    pd.DataFrame(
        [dict(level="L2", source_row_uid="u", semantic_bucket_id="b")]
    ).to_parquet(root / "records.parquet")
    pd.DataFrame([dict(level="L2", semantic_bucket_id="b", weight=0.6)]).to_parquet(
        root / "weights.parquet"
    )
    (root / "map.json").write_text("{}")
    descriptor = dict(
        schema_version=publication.SCHEMA_VERSION,
        task="dili",
        status="complete_reviewed",
        evidence_library_version="fixture",
        selected=dict(
            semantic_map="map.parquet",
            semantic_map_manifest="map.json",
            record_semantic_bucket_map="records.parquet",
            semantic_bucket_weights="weights.parquet",
        ),
    )
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(descriptor))
    manifest = publication.build_manifest("dili", root, path)
    assert publication.verify_manifest(root)["coverage"]["physical_records"] == 1
    with pytest.raises(FileExistsError, match="fresh staging"):
        publication.build_manifest("dili", root, path)
    (root / "map.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        publication.verify_manifest(root)
    descriptor["selected"]["semantic_map"] = "../descriptor.json"
    with pytest.raises(ValueError, match="escaping"):
        publication._selected_files(root, descriptor)
    descriptor["selected"]["semantic_map"] = "map.parquet"
    pd.DataFrame(
        [dict(level="L2", semantic_bucket_id="foreign", weight=0.6)]
    ).to_parquet(root / "weights.parquet")
    with pytest.raises(ValueError, match="foreign semantic"):
        publication._coverage(publication._selected_files(root, descriptor))
