import json

import pytest

from data.processing.evidence_library.versions.v10.tasks.skin_reaction import (
    build_starling_pair_bucket_transfer_policy as builder,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_auxiliary_metadata import (
    INCREMENTAL_MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_pair_buckets import (
    SKIN_REACTION_V7_PAIR_BUCKET_VERSION,
)


def test_incremental_auxiliary_version_reaches_v7_calibration(tmp_path, monkeypatch):
    pair_metadata = tmp_path / "pair.json"
    auxiliary_metadata = tmp_path / "auxiliary.json"
    pair_metadata.write_text(
        json.dumps({"contract_version": SKIN_REACTION_V7_PAIR_BUCKET_VERSION})
    )
    auxiliary_metadata.write_text(
        json.dumps({"mapping_version": INCREMENTAL_MAPPING_VERSION})
    )
    captured = {}

    def fake_calibration(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(builder, "_build_distance_calibration", fake_calibration)
    result = builder.build_pair_bucket_transfer_policy(
        records_path=tmp_path / "records.parquet",
        pair_bucket_records_path=tmp_path / "buckets.parquet",
        pair_bucket_metadata_path=pair_metadata,
        auxiliary_manifest_path=auxiliary_metadata,
        out_dir=tmp_path / "out",
    )

    assert result == {"ok": True}
    assert captured["spec"].auxiliary_mapping_version == INCREMENTAL_MAPPING_VERSION


def test_unknown_auxiliary_version_still_fails_closed(tmp_path):
    pair_metadata = tmp_path / "pair.json"
    auxiliary_metadata = tmp_path / "auxiliary.json"
    pair_metadata.write_text(
        json.dumps({"contract_version": SKIN_REACTION_V7_PAIR_BUCKET_VERSION})
    )
    auxiliary_metadata.write_text(json.dumps({"mapping_version": "foreign"}))

    with pytest.raises(ValueError, match="unsupported Skin auxiliary mapping version"):
        builder.build_pair_bucket_transfer_policy(
            records_path=tmp_path / "records.parquet",
            pair_bucket_records_path=tmp_path / "buckets.parquet",
            pair_bucket_metadata_path=pair_metadata,
            auxiliary_manifest_path=auxiliary_metadata,
            out_dir=tmp_path / "out",
        )
