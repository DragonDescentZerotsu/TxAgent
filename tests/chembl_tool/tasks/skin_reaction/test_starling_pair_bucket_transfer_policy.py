"""Skin's transfer-policy binding, and the one gate it adds over its siblings.

Skin encodes categorical outcomes onto latent scales with very few distinct
levels by construction -- the signed-direction encoder has exactly three, the
single-subject count encoder exactly two.  A bucket of such records can clear
the record-count and variance gates while carrying almost no spread, which
makes its within-bucket SD scale meaningless and every standardised distance
derived from it arbitrary.  ``MINIMUM_DISTINCT_MEASUREMENT_LEVELS`` exists to
reject those buckets, and is the skin-only rule these tests pin.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_transfer_policy import (
    POLICY_FILENAME,
    build_pair_bucket_transfer_policy,
)
from tools.chembl_tool.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from tools.chembl_tool.tasks.skin_reaction.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    OUTPUT_FIELDS,
)
from tools.chembl_tool.tasks.skin_reaction.starling_pair_bucket_transfer_policy import (
    MINIMUM_DISTINCT_MEASUREMENT_LEVELS,
    PAIR_BUCKET_TRANSFER_POLICY_VERSION,
    SOURCE_CANDIDATE_FIELDS,
    load_pair_bucket_transfer_policy,
    validate_pair_bucket_transfer_policy,
)
from tools.chembl_tool.tasks.skin_reaction.starling_pair_buckets import (
    ENDPOINT_FIELD_BY_SOURCE,
    SKIN_REACTION_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)
from tools.chembl_tool.tasks.skin_reaction.starling_source_column_contracts import (
    SOURCE_COLUMNS,
)

CANDIDATE_COLUMNS = sorted(
    {field for fields in SOURCE_CANDIDATE_FIELDS.values() for field in fields}
)


def _build(tmp_path, buckets_spec, *, out_name="policy"):
    """Materialize a policy from an explicit {bucket_key: (source, endpoint, values)}."""
    records: list[dict] = []
    buckets: list[dict] = []
    for key, (source_id, endpoint, values) in buckets_spec.items():
        for index, value in enumerate(values):
            record_id = f"{key}-{index:03d}"
            records.append(
                {
                    "normalized_record_id": record_id,
                    "finite_scalar_value": value,
                    **{column: None for column in CANDIDATE_COLUMNS},
                }
            )
            buckets.append(
                {
                    "normalized_record_id": record_id,
                    "source_id": source_id,
                    "canonical_endpoint": endpoint,
                    "canonical_unit": "%",
                    "pair_bucket_key": key,
                    "bucket_eligible": True,
                }
            )

    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "pair_bucket_records.parquet"
    pair_metadata_path = tmp_path / "pair_bucket_metadata.json"
    auxiliary_path = tmp_path / "auxiliary_mapping_manifest.json"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(buckets).to_parquet(buckets_path, index=False)
    pair_metadata_path.write_text(
        json.dumps(
            {
                "contract_version": SKIN_REACTION_PAIR_BUCKET_VERSION,
                "source_required_fields": {
                    source: list(fields) for source, fields in SOURCE_PAIR_FIELDS.items()
                },
                "bucket_endpoint_field_by_source": {
                    source: ENDPOINT_FIELD_BY_SOURCE.get(source, "canonical_endpoint")
                    for source in SOURCE_PAIR_FIELDS
                },
            }
        ),
        encoding="utf-8",
    )
    auxiliary_path.write_text(
        json.dumps(
            {
                "mapping_version": MAPPING_VERSION,
                "mapping_sha256": "fixture",
                "attachment_version": AUXILIARY_ATTACHMENT_VERSION,
                "output_fields": list(OUTPUT_FIELDS),
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / out_name
    payload = build_pair_bucket_transfer_policy(
        records_path=records_path,
        pair_bucket_records_path=buckets_path,
        pair_bucket_metadata_path=pair_metadata_path,
        auxiliary_manifest_path=auxiliary_path,
        out_dir=out_dir,
    )
    return payload, out_dir


# --- the skin-only distinct-level gate -------------------------------------


def test_a_bucket_with_too_few_distinct_levels_is_rejected(tmp_path):
    """Enough records and real variance, but only two distinct values.

    This is exactly the shape `single_subject_fraction.v1` produces, and the SD it
    yields does not describe a spread worth standardising against.
    """
    values = [0.0] * 15 + [1.0] * 15
    payload, _ = _build(tmp_path, {"two-level": ("direct_skin_reaction", "sens", values)})

    entry = payload["buckets"]["two-level"]
    assert entry["minimum_support_met"] is True
    assert entry["observed_sample_standard_deviation"] > 0
    assert entry["assay_transfer_eligible"] is False
    assert entry["eligibility_reason"] == "fewer_than_minimum_distinct_levels"
    assert entry["distance_policy"] is None
    # Rejected, not silently dropped: the bucket is still described.
    assert entry["distinct_measurement_levels"] == 2


def test_a_bucket_with_enough_distinct_levels_is_eligible(tmp_path):
    values = [float(index) for index in range(30)]
    payload, _ = _build(tmp_path, {"many-level": ("direct_skin_reaction", "sens", values)})

    entry = payload["buckets"]["many-level"]
    assert entry["assay_transfer_eligible"] is True
    assert entry["eligibility_reason"] == "eligible"
    assert entry["distinct_measurement_levels"] >= MINIMUM_DISTINCT_MEASUREMENT_LEVELS
    assert entry["distance_policy"]["sample_standard_deviation"] > 0


def test_the_gate_sits_exactly_at_the_declared_minimum(tmp_path):
    """One level below the threshold fails; the threshold itself passes."""
    below = [0.0] * 15 + [1.0] * 15
    at = [0.0] * 10 + [1.0] * 10 + [2.0] * 10
    payload, _ = _build(
        tmp_path,
        {
            "below": ("direct_skin_reaction", "sens", below),
            "at": ("direct_skin_reaction", "sens", at),
        },
    )

    assert MINIMUM_DISTINCT_MEASUREMENT_LEVELS == 3
    assert payload["buckets"]["below"]["assay_transfer_eligible"] is False
    assert payload["buckets"]["at"]["assay_transfer_eligible"] is True


def test_the_record_count_gate_still_precedes_the_level_gate(tmp_path):
    """A tiny bucket reports its sample-size failure, not the level failure."""
    payload, _ = _build(
        tmp_path,
        {"tiny": ("direct_skin_reaction", "sens", [float(i) for i in range(5)])},
    )
    entry = payload["buckets"]["tiny"]
    assert entry["assay_transfer_eligible"] is False
    assert entry["eligibility_reason"].startswith("fewer_than_")
    assert "distinct" not in entry["eligibility_reason"]


# --- the published contract ------------------------------------------------


def test_the_artifact_declares_the_gate_and_the_soft_target(tmp_path):
    payload, out_dir = _build(
        tmp_path,
        {"many": ("direct_skin_reaction", "sens", [float(i) for i in range(30)])},
    )

    assert payload["policy_version"] == PAIR_BUCKET_TRANSFER_POLICY_VERSION
    gate = payload["variance_gate_contract"]
    assert gate["minimum_distinct_measurement_levels"] == MINIMUM_DISTINCT_MEASUREMENT_LEVELS
    # Skin opts into the soft target; bioavailability does not, so this must be
    # asserted rather than assumed shared.
    soft = payload["distance_contract"]["soft_transfer_target"]
    assert soft["value_at_midpoint"] == 0.5
    assert soft["replaces_boolean_label"] is False
    assert (out_dir / POLICY_FILENAME).is_file()


def test_policy_round_trips_and_rejects_a_foreign_version(tmp_path):
    _, out_dir = _build(
        tmp_path,
        {"many": ("direct_skin_reaction", "sens", [float(i) for i in range(30)])},
    )
    path = out_dir / POLICY_FILENAME

    loaded = load_pair_bucket_transfer_policy(path)
    assert loaded["policy_version"] == PAIR_BUCKET_TRANSFER_POLICY_VERSION

    foreign = dict(loaded)
    foreign["policy_version"] = "bioavailability_ma_pair_bucket_transfer_policy.v1"
    with pytest.raises(ValueError, match="version mismatch"):
        validate_pair_bucket_transfer_policy(foreign)


def test_the_written_policy_is_deterministic_gzip(tmp_path):
    spec = {"many": ("direct_skin_reaction", "sens", [float(i) for i in range(30)])}
    _, first = _build(tmp_path, spec, out_name="first")
    _, second = _build(tmp_path, spec, out_name="second")
    assert (first / POLICY_FILENAME).read_bytes() == (second / POLICY_FILENAME).read_bytes()
    assert gzip.decompress((first / POLICY_FILENAME).read_bytes())


# --- the substituted bucket endpoint ---------------------------------------


def test_sensitization_buckets_key_on_the_reconciled_endpoint_concept(tmp_path):
    """One bucket spans many raw endpoints, which must not trip the one-endpoint check.

    `sensitization_aop` keys its buckets on `global_endpoint_context` rather
    than `canonical_endpoint`, so a bucket legitimately contains records whose
    raw endpoints differ.  The entry's endpoint has to come from the bucket key.
    """
    assert ENDPOINT_FIELD_BY_SOURCE["sensitization_aop"] == "global_endpoint_context"

    records: list[dict] = []
    buckets: list[dict] = []
    raw_endpoints = ["EC3", "EC3 value", "EC3 (%)"]
    for index in range(30):
        record_id = f"sens-{index:03d}"
        records.append(
            {
                "normalized_record_id": record_id,
                "finite_scalar_value": float(index),
                **{column: None for column in CANDIDATE_COLUMNS},
            }
        )
        buckets.append(
            {
                "normalized_record_id": record_id,
                "source_id": "sensitization_aop",
                # Deliberately inconsistent raw endpoints within one bucket.
                "canonical_endpoint": raw_endpoints[index % len(raw_endpoints)],
                "canonical_unit": "%",
                "pair_bucket_key": json.dumps(
                    ["sensitization_aop", "ec3", "%", "x", "y", "z"],
                    separators=(",", ":"),
                ),
                "bucket_eligible": True,
            }
        )

    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "pair_bucket_records.parquet"
    pair_metadata_path = tmp_path / "pair_bucket_metadata.json"
    auxiliary_path = tmp_path / "auxiliary_mapping_manifest.json"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(buckets).to_parquet(buckets_path, index=False)
    pair_metadata_path.write_text(
        json.dumps(
            {
                "contract_version": SKIN_REACTION_PAIR_BUCKET_VERSION,
                "source_required_fields": {
                    source: list(fields) for source, fields in SOURCE_PAIR_FIELDS.items()
                },
                "bucket_endpoint_field_by_source": {
                    source: ENDPOINT_FIELD_BY_SOURCE.get(source, "canonical_endpoint")
                    for source in SOURCE_PAIR_FIELDS
                },
            }
        ),
        encoding="utf-8",
    )
    auxiliary_path.write_text(
        json.dumps(
            {
                "mapping_version": MAPPING_VERSION,
                "mapping_sha256": "fixture",
                "attachment_version": AUXILIARY_ATTACHMENT_VERSION,
                "output_fields": list(OUTPUT_FIELDS),
            }
        ),
        encoding="utf-8",
    )

    payload = build_pair_bucket_transfer_policy(
        records_path=records_path,
        pair_bucket_records_path=buckets_path,
        pair_bucket_metadata_path=pair_metadata_path,
        auxiliary_manifest_path=auxiliary_path,
        out_dir=tmp_path / "policy",
    )

    entry = next(iter(payload["buckets"].values()))
    assert entry["source_id"] == "sensitization_aop"
    # Derived from the key's endpoint slot, not from the inconsistent column.
    assert entry["canonical_endpoint"] == "ec3"
    assert entry["assay_transfer_eligible"] is True


# --- declared contract sanity ----------------------------------------------


def test_candidate_fields_cover_every_source_and_are_real_columns():
    """A candidate column that is not persisted would silently never score."""
    assert set(SOURCE_CANDIDATE_FIELDS) == set(SOURCE_PAIR_FIELDS)
    for source_id, fields in SOURCE_CANDIDATE_FIELDS.items():
        assert fields, source_id
        declared = set(SOURCE_COLUMNS[source_id])
        missing = [field for field in fields if field not in declared]
        assert not missing, (source_id, missing)


def test_candidate_fields_never_double_as_bucket_identity():
    """A field cannot both define the stratum and be tested for heterogeneity."""
    for source_id, candidates in SOURCE_CANDIDATE_FIELDS.items():
        overlap = set(candidates) & set(SOURCE_PAIR_FIELDS[source_id])
        assert not overlap, (source_id, sorted(overlap))
