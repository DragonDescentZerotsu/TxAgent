import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.versions.v10 import (
    assay_transfer_record_pruning as pruning,
)
from data.processing.evidence_library.versions.v10.assay_transfer_record_pruning import (
    SCHEMA_VERSION,
    _load_manual_ineligibility,
    _candidate_buckets,
    held_out_quartile_tail_outliers,
    tail_outliers,
    validate_review_response,
)
from data.processing.evidence_library.versions.v10.pair_bucket_build import (
    PairBucketBuildSpec,
    _build_canonical_artifacts,
)


def test_final_stage3_requires_record_pruning_manifest(tmp_path: Path) -> None:
    canonical = tmp_path / "02_canonicalized"
    canonical.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"canonical_record_id": "record-1"}]),
        canonical / "records.parquet",
    )
    (canonical / "auxiliary_mapping_manifest.json").write_text("{}\n")
    spec = PairBucketBuildSpec(
        task_id="bbb_martins",
        policy=SimpleNamespace(stage1_canonical_deduplicator=None),
        pair_bucket_version="fixture",
        build_sidecar=lambda **kwargs: {},
        build_transfer_policy=lambda **kwargs: {},
    )

    with pytest.raises(FileNotFoundError, match="record-pruning review"):
        _build_canonical_artifacts(spec, normalized_root=tmp_path)


def test_generation_uses_and_records_high_reasoning_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "normalized"
    stage3 = root / "03_pair_buckets"
    stage3.mkdir(parents=True)
    (stage3 / "records.parquet").write_bytes(b"records")
    (stage3 / "pair_bucket_records.parquet").write_bytes(b"buckets")
    candidate = {
        "task_id": "bbb_martins",
        "bucket_id": "bucket-1",
        "pair_bucket_key": "pair-key",
        "canonical_bucket": {
            "canonical_endpoint_name": "endpoint",
            "canonical_unit_text": "unit",
        },
        "distribution": {"finite_record_count": 20},
        "target_rows": [
            {"canonical_record_id": "record-1", "finite_scalar_value": 1.0}
        ],
        "context_rows": [],
    }
    detector_summary = {
        "tail_candidate_rows": 1,
        "log10_gate_approved_buckets": 0,
        "log10_gate_approved_review_rows": 0,
    }
    captured: dict[str, object] = {}

    class FakeClient:
        def chat_json(self, messages):
            return {
                "content": {
                    "schema_version": SCHEMA_VERSION,
                    "rows": [
                        {
                            "row_id": "r0001",
                            "decision": "keep",
                            "reason_code": "fits_bucket",
                            "reason": "The value is compatible with the assay context.",
                        }
                    ],
                },
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "model": "served-model",
                "id": "response-1",
            }

    def fake_client(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(pruning, "_validate_stage3_inputs", lambda *args: "a" * 64)
    monkeypatch.setattr(
        pruning,
        "_configured_manual_ineligibility",
        lambda *args: ({}, {"path": None, "sha256": None, "records": 0}),
    )
    monkeypatch.setattr(
        pruning, "_candidate_buckets", lambda *args: ([candidate], detector_summary)
    )
    monkeypatch.setattr(pruning, "_attach_semantic_rows", lambda *args: None)
    monkeypatch.setattr(pruning, "split_review_candidate", lambda value: [value])
    monkeypatch.setattr(pruning, "render_review_prompt", lambda value: "prompt")
    monkeypatch.setattr(pruning, "_load_review_cache", lambda path: {})
    monkeypatch.setattr(
        pruning, "openai_compatible_client", lambda **kwargs: (object(), {})
    )
    monkeypatch.setattr(pruning, "OpenAICompatibleClient", fake_client)

    output_dir = tmp_path / "reviews"
    published_stage3 = tmp_path / "published-stage3"
    published_stage3.mkdir()
    (published_stage3 / "records.parquet").write_bytes(b"records")
    (published_stage3 / "pair_bucket_records.parquet").write_bytes(b"buckets")
    manifest = pruning.generate_assay_transfer_record_pruning(
        task_id="bbb_martins",
        normalized_root=root,
        output_dir=output_dir,
        budget_epoch="test",
        token_ledger_path=tmp_path / "ledger.json",
        published_stage3_input_root=published_stage3,
    )
    written_manifest = json.loads(
        (output_dir / pruning.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    cached_review = json.loads(
        (output_dir / pruning.CACHE_FILENAME).read_text(encoding="utf-8")
    )

    assert captured["reasoning_effort"] == "high"
    assert cached_review["reasoning_effort"] == "high"
    assert manifest["review"]["reasoning_effort"] == "high"
    assert written_manifest["review"]["reasoning_effort"] == "high"
    assert written_manifest["inputs"]["records"]["path"] == str(
        published_stage3 / "records.parquet"
    )


def test_generation_preserves_frozen_prior_prunes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    decisions_path = prior_dir / pruning.DECISIONS_FILENAME
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "canonical_record_id": "record-1",
                    "review_decision": "prune_record",
                    "review_reason_code": "wrong_unit_or_scale",
                    "review_reason": "Reviewed unit mismatch.",
                }
            ]
        ),
        decisions_path,
    )
    prior_manifest = prior_dir / pruning.MANIFEST_FILENAME
    prior_manifest.write_text(
        json.dumps(
            {
                "version": pruning.VERSION,
                "task_id": "bbb_martins",
                "contract": pruning.PRUNING_CONTRACT,
                "files": {
                    pruning.DECISIONS_FILENAME: pruning.file_sha256(decisions_path)
                },
            }
        )
    )
    root = tmp_path / "normalized"
    stage3 = root / "03_pair_buckets"
    stage3.mkdir(parents=True)
    (stage3 / "records.parquet").write_bytes(b"records")
    (stage3 / "pair_bucket_records.parquet").write_bytes(b"buckets")
    candidate = {
        "task_id": "bbb_martins",
        "bucket_id": "bucket-1",
        "pair_bucket_key": "pair-key",
        "canonical_bucket": {},
        "distribution": {"finite_record_count": 20},
        "target_rows": [
            {"canonical_record_id": "record-1", "finite_scalar_value": 1.0}
        ],
        "context_rows": [],
    }
    cached = {
        "bucket_id": "bucket-1",
        "reviewer": pruning.REVIEWER,
        "prompt_sha256": pruning._sha256("prompt"),
        "requested_model": pruning.DEFAULT_MODEL,
        "served_model": "served-model",
        "reasoning_effort": pruning.REASONING_EFFORT,
        "response": {
            "schema_version": SCHEMA_VERSION,
            "rows": [
                {
                    "canonical_record_id": "record-1",
                    "decision": "keep",
                    "reason_code": "valid_extreme",
                    "reason": "Plausible extreme.",
                }
            ],
        },
    }
    monkeypatch.setattr(pruning, "_validate_stage3_inputs", lambda *args: "a" * 64)
    monkeypatch.setattr(
        pruning,
        "_configured_manual_ineligibility",
        lambda *args: ({}, {"path": None, "sha256": None, "records": 0}),
    )
    monkeypatch.setattr(pruning, "_candidate_buckets", lambda *args: ([candidate], {
        "tail_candidate_rows": 1,
        "log10_gate_approved_buckets": 0,
        "log10_gate_approved_review_rows": 0,
    }))
    monkeypatch.setattr(pruning, "_attach_semantic_rows", lambda *args: None)
    monkeypatch.setattr(pruning, "split_review_candidate", lambda value: [value])
    monkeypatch.setattr(pruning, "render_review_prompt", lambda value: "prompt")
    monkeypatch.setattr(pruning, "_load_review_cache", lambda path: {"bucket-1": cached})

    manifest = pruning.generate_assay_transfer_record_pruning(
        task_id="bbb_martins",
        normalized_root=root,
        output_dir=tmp_path / "reviews",
        frozen_prior_manifest=prior_manifest,
    )
    rows = pq.read_table(tmp_path / "reviews" / pruning.DECISIONS_FILENAME).to_pylist()

    assert rows[0]["review_decision"] == "prune_record"
    assert manifest["summary"]["frozen_prior_pruned_rows"] == 1
    assert manifest["validations"]["all_frozen_prior_prunes_preserved"] is True


def test_frozen_prior_prune_survives_tail_candidate_drift(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "canonical_record_id": "record-1",
                    "pair_bucket_key": "pair-key",
                    "finite_scalar_value": 1.0,
                    "assay_transfer_eligible": True,
                }
            ]
        ),
        records_path,
    )
    decisions = pruning._carry_forward_frozen_prunes(
        ["record-1"],
        frozen_prunes={
            "record-1": {
                "decision": "prune_record",
                "reason_code": "wrong_quantity_or_endpoint",
                "reason": "Reviewed quantity mismatch.",
            }
        },
        records_path=records_path,
        task_id="skin_reaction",
    )

    assert decisions[0]["review_decision"] == "prune_record"
    assert decisions[0]["tail_geometry"] == "frozen_prior_carryforward"


def test_manual_ineligibility_is_identity_bound(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    mapping_path = tmp_path / "manual.json"
    pq.write_table(
        pa.Table.from_pylist(
            [{"canonical_record_id": "a" * 64, "source_row_uid": "source-row-1"}]
        ),
        records_path,
    )
    payload = {
        "version": "bbb_assay_transfer_manual_ineligibility.v1",
        "task_id": "bbb_martins",
        "records": [
            {
                "canonical_record_id": "a" * 64,
                "source_row_uid": "source-row-1",
                "reason_code": "valid_source_out_of_assay_transfer_domain",
                "rationale": "A source-valid result can still be outside the comparable transfer domain.",
            }
        ],
    }
    mapping_path.write_text(json.dumps(payload), encoding="utf-8")

    decisions, manifest = _load_manual_ineligibility(
        mapping_path, task_id="bbb_martins", records_path=records_path
    )

    assert decisions == {"a" * 64: "valid_source_out_of_assay_transfer_domain"}
    assert manifest["records"] == 1
    payload["records"][0]["source_row_uid"] = "wrong-source-row"
    mapping_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity drift"):
        _load_manual_ineligibility(
            mapping_path, task_id="bbb_martins", records_path=records_path
        )


def test_held_out_quartile_finds_a_masked_tied_tail() -> None:
    values = [1.0] * 15 + [1000.0] * 5

    assert tail_outliers(values, unit_text="log(response)") == []
    tails = held_out_quartile_tail_outliers(values, unit_text="log(response)")

    assert [tail["record_index"] for tail in tails] == list(range(15, 20))
    assert {tail["held_out_tail_side"] for tail in tails} == {"upper"}
    assert all(tail["held_out_tail_zero_reference_sd"] for tail in tails)


def test_bbb_candidates_union_existing_and_held_out_triggers(tmp_path: Path) -> None:
    values = [1.0] * 15 + [1000.0] * 5
    records = [
        {
            "canonical_record_id": f"r{index:02d}",
            "canonical_smiles": f"parent-{index:02d}",
            "measurement_kind": "continuous",
            "finite_scalar_value": value,
            "assay_transfer_pretransform_scalar_value": value,
            "assay_transfer_transform_id": "raw.v1",
        }
        for index, value in enumerate(values)
    ]
    buckets = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": '["source","endpoint","log(response)"]',
            "canonical_pair_fields_json": "{}",
            "canonical_endpoint_name": "endpoint",
            "canonical_unit_text": "log(response)",
            "assay_transfer_eligible": True,
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "buckets.parquet"
    pq.write_table(pa.Table.from_pylist(records), records_path)
    pq.write_table(pa.Table.from_pylist(buckets), buckets_path)

    candidates, summary = _candidate_buckets("bbb_martins", records_path, buckets_path)

    assert candidates[0]["target_record_ids"] == [
        f"r{index:02d}" for index in range(15, 20)
    ]
    assert summary["leave_one_out_candidate_rows"] == 0
    assert summary["held_out_quartile_candidate_rows"] == 5
    assert summary["tail_candidate_rows"] == 5
    assert candidates[0]["distribution"]["observed_median"] == 1.0
    assert candidates[0]["distribution"]["observed_maximum"] == 1000.0


def test_review_accepts_scientifically_implausible_pruning() -> None:
    response = {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "row_id": "r0001",
                "decision": "prune_record",
                "reason_code": "scientifically_implausible_or_out_of_domain",
                "reason": "Magnitude is incompatible with the assay's useful range.",
            }
        ],
    }

    validated = validate_review_response(
        response, record_ids={"r0001"}, id_field="row_id"
    )

    assert validated == response
