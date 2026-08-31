from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from tools.chembl_tool.common.starling.record_collapse import (
    _apply_semantic_transfer_decision,
    _aggregate_group,
    _aggregation_method,
    _collapse_group_key,
    _semantic_payload,
    build_collapsed_record_stage,
)
from tools.chembl_tool.common.starling.record_deduplication import (
    build_deduplicated_record_stage,
)
from tools.chembl_tool.common.starling.collapsed_informativeness import (
    CACHE_VERSION as INFORMATIVENESS_CACHE_VERSION,
    TEMPLATE_PATH as INFORMATIVENESS_TEMPLATE_PATH,
    VERSION as INFORMATIVENESS_VERSION,
    Batch as InformativenessBatch,
    build_views as build_informativeness_views,
    render_batch as render_informativeness_batch,
    seed_exact_requests,
    validate_response as validate_informativeness_response,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.pair_buckets import materialize_pair_buckets
from tools.chembl_tool.common.starling.semantic_record_aggregation import (
    SemanticAggregationBudgetExhausted,
    SemanticAggregationConfig,
    _run_one,
    aggregate_semantic_groups,
    match_semantic_cache_row,
    render_prompt,
    validate_semantic_response,
)


def test_reviewed_semantic_bucket_promotion_is_exact_key_only():
    collapsed = {
        "pair_bucket_key": '["fg","bidirectional_permeability","log10(ratio)"]',
        "assay_transfer_eligible": False,
        "assay_transfer_ineligibility_reason": (
            "reference_scope_endpoint_defined_ratio"
        ),
    }
    assert (
        _apply_semantic_transfer_decision(
            collapsed,
            policy_entries={
                collapsed["pair_bucket_key"]: {
                    "decision": "eligible",
                    "reason_code": "single_axis",
                }
            },
        )
        == "promoted"
    )
    assert collapsed["assay_transfer_eligible"] is True
    assert collapsed["assay_transfer_ineligibility_reason"] is None

    rejected = {
        "pair_bucket_key": '["fg","other","log10(ratio)"]',
        "assay_transfer_eligible": False,
        "assay_transfer_ineligibility_reason": "reference_scope_unknown",
    }
    assert (
        _apply_semantic_transfer_decision(
            rejected,
            policy_entries={
                rejected["pair_bucket_key"]: {
                    "decision": "ineligible",
                    "reason_code": "mixed_axis",
                }
            },
        )
        == "unchanged"
    )
    assert rejected["assay_transfer_eligible"] is False
    assert rejected["assay_transfer_ineligibility_reason"] == (
        "semantic_policy_mixed_axis"
    )
def test_record_collapse_uses_deterministic_aggregates_and_cross_source_dedup(
    tmp_path,
):
    records = [
        _record("r1", "source_a", "CCO", "continuous", value=10.0, support="same"),
        _record("r2", "source_b", "CCO", "continuous", value=20.0, support="other"),
        _record("r3", "source_c", "CCO", "continuous", value=10.0, support="same"),
        _record("r4", "source_a", "CCN", "ordinal", category="low"),
        _record("r5", "source_b", "CCN", "ordinal", category="high"),
        _record("r6", "direct", "CCC", "binary", category="positive"),
        _record("r7", "direct", "CCC", "binary", category="positive"),
    ]
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": (
                "continuous_bucket"
                if row["canonical_record_id"] in {"r1", "r2", "r3"}
                else "ordinal_bucket"
                if row["canonical_record_id"] in {"r4", "r5"}
                else None
            ),
            "canonical_pair_fields_json": "{}",
            "assay_transfer_eligible": row["source_id"] != "direct",
            "assay_transfer_ineligibility_reason": None,
            "bucket_eligible": row["source_id"] != "direct",
            "bucket_exclusion_reason": None,
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")

    dedup_manifest = build_deduplicated_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        out_dir=tmp_path / "deduplicated",
        direct_mapping_builder=_direct_mapping,
    )
    deduplicated_sidecar = pd.read_parquet(
        tmp_path / "deduplicated/pair_bucket_records.parquet"
    )
    direct_sidecar = deduplicated_sidecar[
        ~deduplicated_sidecar["assay_transfer_eligible"]
    ].iloc[0]
    assert not direct_sidecar["bucket_eligible"]
    assert direct_sidecar["bucket_exclusion_reason"] == (
        "direct_binary_vote_not_assay_transferable"
    )
    manifest = build_collapsed_record_stage(
        task_id="test_task",
        records_path=tmp_path / "deduplicated/records.parquet",
        pair_bucket_records_path=tmp_path / "deduplicated/pair_bucket_records.parquet",
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
        duplicate_lineage_path=tmp_path / "deduplicated/duplicates.parquet",
    )

    collapsed = pd.read_parquet(tmp_path / "collapsed/records.parquet")
    continuous = collapsed[collapsed["pair_bucket_key"] == "continuous_bucket"].iloc[0]
    ordinal = collapsed[collapsed["pair_bucket_key"] == "ordinal_bucket"].iloc[0]
    direct = collapsed[collapsed["retrieval_source_id"] == "direct_vote"].iloc[0]
    assert manifest["summary"] == {
        **manifest["summary"],
        "input_records": 5,
        "retrieval_eligible_input_records": 5,
        "deduplicated_input_records": 5,
        "collapsed_records": 3,
        "direct_mapping_records": 0,
        "semantic_groups": 0,
        "source_records_represented": 7,
    }
    assert continuous["aggregation_method"] == "continuous_median"
    assert continuous["finite_scalar_value"] == 15.0
    assert continuous["display_scalar_value"] == 15.0
    assert continuous["display_unit_text"] == "mg/L"
    assert continuous["source_record_count"] == 3
    assert pd.isna(continuous["direct_label_informativeness"])
    assert continuous["source_id"] == "multi_source"
    assert continuous["group_id"] == "Observed.indirect"
    assert set(json.loads(continuous["source_ids_json"])) == {
        "source_a",
        "source_b",
        "source_c",
    }
    assert "qualifying_conditions" not in collapsed.columns
    assert "semantic_aggregation_json" not in collapsed.columns
    assert "pair_bucket_unit_text" not in collapsed.columns
    assert ordinal["aggregation_method"] == "categorical_mode"
    assert ordinal["aggregation_status"] == "conflict"
    assert pd.isna(ordinal["canonical_category_id"])
    assert direct["finite_scalar_value"] == 1.0
    assert pd.isna(direct["display_unit_text"])
    assert direct["group_id"] == "Observed.direct"
    assert direct["assay_transfer_eligible"] == False  # noqa: E712
    assert dedup_manifest["summary"]["duplicate_scope_counts"] == {
        "cross_source": 1,
        "within_source": 1,
    }
    mapping = pd.read_parquet(tmp_path / "deduplicated/direct_record_mapping.parquet")
    assert set(mapping["direct_vote_status"]) == {"counted"}
    assert "canonical_claim_id" not in mapping.columns
    duplicate_mapping = mapping[mapping["dedup_status"] != "retained"]
    assert duplicate_mapping["dedup_retained_canonical_record_id"].notna().all()
    duplicates = pd.read_parquet(tmp_path / "deduplicated/duplicates.parquet")
    assert duplicates[
        ["retained_canonical_record_id", "discarded_canonical_record_id"]
    ].values.tolist() == [["r1", "r3"], ["r6", "r7"]]


def test_row_dedup_is_one_stage_and_never_merges_conflicting_direct_labels(tmp_path):
    records = [
        _record("r1", "source_a", "CCO", "continuous", value=10.0, support="same"),
        _record("r2", "source_a", "CCO", "continuous", value=10.0, support="same"),
        _record("r3", "direct", "CCC", "binary", category="positive"),
        _record("r4", "direct", "CCC", "binary", category="positive"),
    ]
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": "bucket" if row["source_id"] != "direct" else None,
            "canonical_pair_fields_json": "{}",
            "assay_transfer_eligible": row["source_id"] != "direct",
            "assay_transfer_ineligibility_reason": None,
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)

    def conflicting_mapping(rows):
        mapped = _direct_mapping(rows)
        for row in mapped:
            row["direct_vote_label"] = 0 if row["canonical_record_id"] == "r4" else 1
        return mapped

    manifest = build_deduplicated_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        out_dir=tmp_path / "dedup",
        direct_mapping_builder=conflicting_mapping,
    )

    assert manifest["summary"]["duplicate_scope_counts"] == {"within_source": 1}
    assert len(pd.read_parquet(tmp_path / "dedup/records.parquet")) == 3
    mapping = pd.read_parquet(tmp_path / "dedup/direct_record_mapping.parquet")
    assert len(mapping) == 2
    assert set(mapping["direct_vote_label"]) == {0, 1}
    assert set(mapping["dedup_status"]) == {"retained"}

    metadata_path = tmp_path / "pair_bucket_metadata.json"
    metadata_path.write_text("{}\n", encoding="utf-8")
    build_collapsed_record_stage(
        task_id="test_task",
        records_path=tmp_path / "dedup/records.parquet",
        pair_bucket_records_path=tmp_path / "dedup/pair_bucket_records.parquet",
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
        duplicate_lineage_path=tmp_path / "dedup/duplicates.parquet",
    )
    collapsed = pd.read_parquet(tmp_path / "collapsed/records.parquet")
    residual = collapsed[collapsed["retrieval_source_id"] == "direct_residual"].iloc[0]
    assert residual["group_id"] == "Observed.direct"
    assert residual["aggregation_status"] == "conflict"


def test_row_dedup_does_not_collapse_distinct_within_source_contexts(tmp_path):
    records = [
        {
            **_record("r1", "source_a", "CCO", "continuous", value=10.0),
            "deduplication_context_id": "context-a",
        },
        {
            **_record("r2", "source_a", "CCO", "continuous", value=10.0),
            "deduplication_context_id": "context-b",
        },
    ]
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": "same-bucket",
            "canonical_pair_fields_json": "{}",
            "assay_transfer_eligible": True,
            "assay_transfer_ineligibility_reason": None,
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)

    manifest = build_deduplicated_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        out_dir=tmp_path / "dedup",
    )

    assert manifest["summary"]["duplicates_removed"] == 0
    assert len(pd.read_parquet(tmp_path / "dedup/records.parquet")) == 2


def test_row_dedup_keeps_incompatible_categorical_scales(tmp_path):
    records = [
        {
            **_record("r1", "source_a", "CCO", "binary", category="positive"),
            "canonical_measurement_scale_id": "scale_a",
        },
        {
            **_record("r2", "source_a", "CCO", "binary", category="positive"),
            "canonical_measurement_scale_id": "scale_b",
        },
    ]
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": "same-bucket",
            "canonical_pair_fields_json": "{}",
            "assay_transfer_eligible": False,
            "assay_transfer_ineligibility_reason": "categorical",
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)

    manifest = build_deduplicated_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        out_dir=tmp_path / "dedup",
    )

    assert manifest["summary"]["duplicates_removed"] == 0
    assert len(pd.read_parquet(tmp_path / "dedup/records.parquet")) == 2


def test_semantic_response_is_summary_only_and_rejects_internal_names():
    response = {"summary": "The reported result was consistent."}
    assert validate_semantic_response(response) == response
    with pytest.raises(ValueError, match="fields differ"):
        validate_semantic_response(
            {**response, "direct_label_informativeness": "informative"}
        )
    with pytest.raises(ValueError, match="internal structure identifier"):
        validate_semantic_response({"summary": "Compound SMILES:123 was active."})


def test_semantic_prompt_keeps_only_scientific_source_fields():
    record = {
        **_record("r1", "source_a", "CCO", "semantic"),
        "retrieval_source_id": "indirect",
        "pair_bucket_key": "semantic-bucket",
        "canonical_unit_text": "free-text",
        "canonical_pair_fields_json": '{"endpoint":"reported outcome"}',
        "positive_count": 3,
        "oral_dose": "10 mg/kg",
        "formulation_or_solid_form": "lipid formulation",
        "support_text": "The tested compound (SMILES: CCO) produced a response.",
        "pmid": "12345",
        "assay_transfer_eligible": False,
    }
    payload = _semantic_payload(
        "test_task",
        "group-key",
        [record],
        semantic_source_columns={
            "source_a": (
                "positive_count",
                "oral_dose",
                "formulation_or_solid_form",
                "support_text",
            )
        },
    )
    prompt = render_prompt(payload)

    fields = {
        field["label"]: field["value"]
        for field in payload["records"][0]["fields"]
    }
    assert fields["Positive count"] == "3"
    assert fields["Endpoint"] == "test_endpoint"
    assert fields["Oral dose"] == "10 mg/kg"
    assert fields["Formulation or solid form"] == "lipid formulation"
    assert "Positive count: 3" in prompt
    assert "direct-label" not in prompt
    assert "assay_transfer_eligible" not in prompt
    assert "canonical_smiles" not in prompt
    assert "group-key" not in prompt
    assert "source_a" not in prompt
    assert "12345" not in prompt
    assert "SMILES" not in prompt
    assert "CCO" not in prompt
    with pytest.raises(RuntimeError, match="require LLM aggregation"):
        aggregate_semantic_groups(
            {"group-key": payload},
            config=None,
        )


def test_deterministic_informativeness_uses_complete_representative_and_two_views(
    tmp_path,
):
    records = [
        {
            **_record("r1", "source_a", "CCO", "continuous", value=10.0),
            "support_text": "Measured after oral dosing.",
            "oral_dose": "10 mg/kg",
            "confidence": 0.4,
        },
        {
            **_record("r2", "source_a", "CCO", "continuous", value=20.0),
            "support_text": "Measured.",
            "oral_dose": None,
            "confidence": 0.9,
        },
    ]
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": "continuous-bucket",
            "canonical_pair_fields_json": '{"species":"rat"}',
            "assay_transfer_eligible": True,
            "assay_transfer_ineligibility_reason": None,
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "sidecar.parquet"
    metadata_path = tmp_path / "metadata.json"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")
    build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "baseline",
        semantic_source_columns={"source_a": ("support_text", "oral_dose")},
        direct_label_definition="Whether the direct outcome is positive.",
    )
    collapsed = pd.read_parquet(tmp_path / "baseline/records.parquet").iloc[0].to_dict()
    joined = []
    for record, bucket in zip(records, sidecar, strict=True):
        joined.append({**record, **bucket, "retrieval_source_id": "indirect"})
    views = build_informativeness_views(
        collapsed,
        joined,
        {"source_a": ("support_text", "oral_dose")},
        direct_label_definition="Whether the direct outcome is positive.",
    )
    representative, collapsed_view = views
    assert representative["representative_record_id"] == "r1"
    assert all(field["label"] != "Molecule" for field in representative["fields"])
    assert {field["label"] for field in collapsed_view["fields"]} >= {
        "Endpoint",
        "Result",
        "Species",
    }
    assert all(
        field["label"] not in {"Observed range", "Interquartile range"}
        for field in collapsed_view["fields"]
    )
    batch = InformativenessBatch(
        "test_task", tuple(views), "Whether the direct outcome is positive."
    )
    prompt = render_informativeness_batch(batch)
    assert "Measured after oral dosing." in prompt
    assert "Observed range" not in prompt
    assert "CCO" not in prompt
    response = validate_informativeness_response(
        {"items": [
            {"id": "0", "informativeness": "informative"},
            {"id": "1", "informativeness": "uninformative"},
        ]},
        2,
    )

    artifact = tmp_path / "informativeness"
    artifact.mkdir()
    payloads = artifact / "payloads.jsonl"
    payloads.write_text(
        "".join(json.dumps(view) + "\n" for view in views), encoding="utf-8"
    )
    request = {
        "cache_version": INFORMATIVENESS_CACHE_VERSION,
        "task_id": "test_task",
        "model": "gpt-5.6-luna",
        "template_sha256": file_sha256(INFORMATIVENESS_TEMPLATE_PATH),
        "items": [
            {
                "view_id": view["view_id"],
                "group_id": view["group_id"],
                "view": view["view"],
                "payload_sha256": view["payload_sha256"],
                "representative_record_id": view["representative_record_id"],
                "informativeness": result["informativeness"],
            }
            for view, result in zip(views, response, strict=True)
        ],
    }
    requests = artifact / "requests.jsonl"
    requests.write_text(json.dumps(request) + "\n", encoding="utf-8")
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "version": INFORMATIVENESS_VERSION,
                "task_id": "test_task",
                "status": "complete",
                "target_groups": 1,
                "target_views": 2,
                "template_sha256": file_sha256(INFORMATIVENESS_TEMPLATE_PATH),
                "requests_sha256": file_sha256(requests),
                "payloads_sha256": file_sha256(payloads),
            }
        ),
        encoding="utf-8",
    )
    manifest = build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "attached",
        semantic_source_columns={"source_a": ("support_text", "oral_dose")},
        direct_label_definition="Whether the direct outcome is positive.",
        collapsed_informativeness_dir=artifact,
    )
    attached = pd.read_parquet(tmp_path / "attached/records.parquet").iloc[0]
    assert attached["direct_label_informativeness"] == "uninformative"
    assert attached["representative_direct_label_informativeness"] == "informative"
    assert attached["informativeness_representative_record_id"] == "r1"
    assert attached["direct_label_informativeness_model"] == "gpt-5.6-luna"
    assert attached["retrieval_eligible"] == collapsed["retrieval_eligible"]
    assert manifest["summary"]["collapsed_informativeness_flag_pairs"] == {
        "informative -> uninformative": 1
    }


def test_semantic_informativeness_uses_only_the_collapsed_view():
    collapsed = {
        "collapse_group_key": "group",
        "aggregation_method": "semantic_support_passthrough",
        "canonical_endpoint_name": "oral exposure",
        "display_measurement_text": "Exposure increased after oral dosing.",
        "display_unit_text": None,
        "canonical_pair_fields_json": '{"species":"rat"}',
        "source_record_count": 1,
        "aggregate_counts_json": "{}",
    }
    views = build_informativeness_views(
        collapsed,
        [],
        {},
        direct_label_definition="Whether oral bioavailability is high.",
    )

    assert len(views) == 1
    assert views[0]["view"] == "collapsed"
    assert views[0]["representative_record_id"] is None
    assert {field["label"] for field in views[0]["fields"]} >= {
        "Endpoint",
        "Result",
        "Species",
    }


def test_informativeness_pilot_seed_reuses_only_exact_payloads_and_model(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "payloads.jsonl").write_text(
        json.dumps({"view_id": "exact", "payload_sha256": "hash-a"}) + "\n"
        + json.dumps({"view_id": "changed", "payload_sha256": "hash-b"}) + "\n",
        encoding="utf-8",
    )
    (target / "requests.jsonl").touch()
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps(
            {
                "cache_version": INFORMATIVENESS_CACHE_VERSION,
                "task_id": "test_task",
                "batch_id": "pilot",
                "model": "gpt-5.4-mini",
                "template_sha256": file_sha256(INFORMATIVENESS_TEMPLATE_PATH),
                "items": [
                    {
                        "view_id": "exact",
                        "group_id": "group-a",
                        "view": "collapsed",
                        "payload_sha256": "hash-a",
                        "representative_record_id": None,
                        "informativeness": "informative",
                    },
                    {
                        "view_id": "changed",
                        "group_id": "group-b",
                        "view": "collapsed",
                        "payload_sha256": "stale-hash",
                        "representative_record_id": None,
                        "informativeness": "uninformative",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert seed_exact_requests(target, prior, model="gpt-5.4-mini") == 1
    assert seed_exact_requests(target, prior, model="gpt-5.4-mini") == 0
    seeded = json.loads((target / "requests.jsonl").read_text(encoding="utf-8"))
    assert [item["view_id"] for item in seeded["items"]] == ["exact"]
    assert seeded["seeded_from"] == str(prior)


def test_structurally_valid_cached_summary_is_reused(tmp_path):
    payload = _semantic_test_payload("group")
    prompt = render_prompt(payload)
    row = _semantic_cache_row("group", payload, prompt, "test-model")
    row["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    row["response"]["summary"] = "This evidence does not establish the direct outcome."
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = aggregate_semantic_groups(
        {"group": payload}, config=None, prior_paths=(cache_path,)
    )
    assert result[0]["response"]["summary"].startswith("This evidence")


def test_legacy_prompt_cache_is_not_reused():
    payload = _semantic_test_payload("group")
    prompt_sha256 = hashlib.sha256(render_prompt(payload).encode()).hexdigest()
    legacy = _semantic_cache_row("group", payload, "old prompt", "test-model")
    legacy["input"] = json.loads(json.dumps(payload))
    legacy["input"]["records"][0]["fields"].extend(
        [
            {"label": "Source index", "value": "42"},
            {"label": "Skin source", "value": "legacy"},
            {"label": "Molecule", "value": "SMILES: CCO"},
        ]
    )
    matched = match_semantic_cache_row(
        legacy,
        payload=payload,
        prompt_sha256=prompt_sha256,
        expected_model="test-model",
    )
    assert matched is None
    changed = json.loads(json.dumps(legacy))
    changed["input"]["records"][0]["fields"][0]["value"] = "different result"
    assert match_semantic_cache_row(
        changed,
        payload=payload,
        prompt_sha256=prompt_sha256,
        expected_model="test-model",
    ) is None


def test_single_relative_scalar_does_not_require_an_llm(tmp_path):
    record = {
        **_record("r1", "source_a", "CCO", "relative", value=2.0),
        "retrieval_source_id": "indirect",
        "canonical_unit_text": "relative-scalar",
    }
    assert _aggregation_method([record]) == "single_record_passthrough"
    assert _aggregation_method(
        [{**record, "finite_scalar_value": None, "measurement_kind": "semantic"}]
    ) == "semantic_support_passthrough"
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    pd.DataFrame([record]).to_parquet(records_path, index=False)
    pd.DataFrame(
        [
            {
                "canonical_record_id": "r1",
                "pair_bucket_key": "semantic-bucket",
                "canonical_pair_fields_json": "{}",
                "assay_transfer_eligible": False,
                "assay_transfer_ineligibility_reason": "relative_scalar",
            }
        ]
    ).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")

    manifest = build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
    )

    collapsed = pd.read_parquet(tmp_path / "collapsed/records.parquet").iloc[0]
    assert manifest["summary"]["semantic_groups"] == 0
    assert collapsed["aggregation_method"] == "single_record_passthrough"
    assert collapsed["canonical_measurement_text"] == "2"
    assert collapsed["display_measurement_text"] == "2× relative to comparator"


def test_direct_residual_group_key_separates_deterministic_measurement_axes():
    condition = {
        "condition_group": "no_reported_external_condition",
        "condition_key_status": "none_reported",
    }
    first = {
        **_record("r1", "source_a", "CCO", "continuous", value=0.5),
        **condition,
        "retrieval_source_id": "direct_residual",
        "canonical_endpoint_name": "brain_uptake",
        "canonical_unit_text": "%",
    }
    same_axis = {
        **_record("r2", "source_a", "CCO", "continuous", value=0.7),
        **condition,
        "retrieval_source_id": "direct_residual",
        "canonical_endpoint_name": "brain_uptake",
        "canonical_unit_text": "%",
    }
    mixed_axis = {
        **_record("r3", "source_a", "CCO", "continuous", value=-0.2),
        **condition,
        "retrieval_source_id": "direct_residual",
        "canonical_endpoint_name": "whole_brain_uptake",
        "canonical_unit_text": "log10(%ID/g)",
    }

    assert _collapse_group_key(first) == _collapse_group_key(same_axis)
    assert _collapse_group_key(first) != _collapse_group_key(mixed_axis)
    assert _aggregation_method([first, same_axis]) == "continuous_median"
    with pytest.raises(ValueError, match="spans measurement axes"):
        _aggregation_method([first, mixed_axis])
    bioavailability = {
        **first,
        "canonical_endpoint_name": "bioavailability",
        "direct_residual_endpoint_name": "oral_bioavailability",
    }
    oral_bioavailability = {
        **same_axis,
        "canonical_endpoint_name": "oral_bioavailability",
        "direct_residual_endpoint_name": "oral_bioavailability",
    }
    assert _collapse_group_key(bioavailability) == _collapse_group_key(
        oral_bioavailability
    )
    assert _aggregation_method(
        [bioavailability, oral_bioavailability]
    ) == "continuous_median"


def test_collapse_names_alias_group_from_pair_bucket_endpoint():
    key = json.dumps(["oral_exposure", "auc_0_infinity", "log10(h·ng/mL)"])
    group = [
        {
            **_record(f"r{number}", "source_a", "CCO", "continuous", value=value),
            "retrieval_source_id": "indirect",
            "group_id": "Observed.oral_exposure",
            "canonical_endpoint_name": endpoint,
            "canonical_endpoint_concept": "auc_0_infinity",
            "canonical_unit_text": "log10(h·ng/mL)",
            "pair_bucket_key": key,
            "canonical_pair_fields_json": "{}",
            "assay_transfer_eligible": True,
        }
        for number, (endpoint, value) in enumerate(
            (("auc0_inf", 1.0), ("aucinf", 2.0)), start=1
        )
    ]

    collapsed = _aggregate_group(
        "bioavailability_ma",
        "alias-group",
        group,
        method="continuous_median",
        semantic=None,
        dedup_lineage={},
        preserved_columns=("canonical_endpoint_concept",),
    )

    assert collapsed["canonical_endpoint_name"] == "auc_0_infinity"
    assert collapsed["canonical_endpoint_concept"] == "auc_0_infinity"
    assert collapsed["finite_scalar_value"] == 1.5


def test_direct_residual_absolute_axes_collapse_without_llm(tmp_path, monkeypatch):
    records = []
    for record_id, source_id, endpoint, unit, value in (
        ("r1", "source_a", "brain_uptake", "%", 10.0),
        ("r2", "source_b", "brain_uptake", "%", 20.0),
        ("r3", "source_a", "brain_uptake", "log10(%ID/g)", -1.0),
        ("r4", "source_a", "brain_concentration", "%", 30.0),
    ):
        records.append(
            {
                **_record(record_id, source_id, "CCO", "continuous", value=value),
                "retrieval_source_id": "direct_residual",
                "group_id": "Observed.direct",
                "direct_group_id": "Observed.direct",
                "canonical_endpoint_name": endpoint,
                "canonical_unit_text": unit,
                "condition_group": "no_reported_external_condition",
                "condition_atoms": [],
                "condition_scope": "none_reported",
                "condition_key_status": "none_reported",
            }
        )
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": None,
            "canonical_pair_fields_json": None,
            "assay_transfer_eligible": False,
            "assay_transfer_ineligibility_reason": "direct_outcome",
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.record_collapse.aggregate_semantic_groups",
        lambda *args, **kwargs: pytest.fail("absolute direct residual called the LLM"),
    )

    manifest = build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
    )

    collapsed = pd.read_parquet(tmp_path / "collapsed/records.parquet")
    percent_uptake = collapsed[
        (collapsed["canonical_endpoint_name"] == "brain_uptake")
        & (collapsed["canonical_unit_text"] == "%")
    ].iloc[0]
    assert len(collapsed) == 3
    assert manifest["summary"]["semantic_groups"] == 0
    assert set(collapsed["aggregation_method"]) == {"continuous_median"}
    assert percent_uptake["finite_scalar_value"] == 15.0
    assert percent_uptake["source_id"] == "multi_source"
    assert percent_uptake["is_absolute_and_continuous"] == True  # noqa: E712


def test_direct_pair_buckets_add_conditions_and_transfer_residuals(tmp_path):
    records = [
        _record(record_id, "direct", "CCO", "continuous", value=value)
        for record_id, value in (
            ("r1", 10.0),
            ("r2", 20.0),
            ("r3", 30.0),
            ("r4", 40.0),
            ("r5", 50.0),
        )
    ]
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": (
                None
                if row["canonical_record_id"] == "r5"
                else '["direct","test_endpoint","mg/L"]'
            ),
            "canonical_pair_fields_json": (
                None if row["canonical_record_id"] == "r5" else '{}'
            ),
            "assay_transfer_eligible": True,
            "assay_transfer_ineligibility_reason": None,
        }
        for row in records
    ]
    condition_by_id = {
        "r1": ("no_reported_external_condition", "none_reported"),
        "r2": ("no_reported_external_condition", "none_reported"),
        "r3": ("disease=cancer", "accepted_unselected"),
        "r4": ("disease=cancer", "proposed_rejected"),
        "r5": ("no_reported_external_condition", "none_reported"),
    }

    def residual_mapping(rows):
        return [
            {
                "canonical_record_id": row["canonical_record_id"],
                "source_id": row["source_id"],
                "source_record_id": row["source_record_id"],
                "direct_vote_status": "ignored",
                "direct_vote_reason": "test",
                "direct_vote_label": None,
                "direct_vote_unit_id": f"direct:{row['source_row_number']}",
                "condition_group": condition_by_id[row["canonical_record_id"]][0],
                "condition_atoms": [],
                "condition_scope": "external",
                "condition_key_status": condition_by_id[row["canonical_record_id"]][1],
                "retrieval_source_id": "direct_residual",
                "direct_group_id": "Observed.direct",
                "dedup_status": "pending",
            }
            for row in rows
        ]

    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")
    build_deduplicated_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        out_dir=tmp_path / "deduplicated",
        direct_mapping_builder=residual_mapping,
    )

    deduplicated = pd.read_parquet(tmp_path / "deduplicated/records.parquet")
    deduplicated_sidecar = pd.read_parquet(
        tmp_path / "deduplicated/pair_bucket_records.parquet"
    )
    assert deduplicated["pair_bucket_key"].tolist() == deduplicated_sidecar[
        "pair_bucket_key"
    ].tolist()
    assert deduplicated.loc[0, "pair_bucket_key"] == deduplicated.loc[1, "pair_bucket_key"]
    assert deduplicated.loc[2, "pair_bucket_key"] != deduplicated.loc[3, "pair_bucket_key"]
    assert json.loads(deduplicated.loc[2, "canonical_pair_fields_json"])[
        "canonical_direct_condition_group"
    ] == "disease=cancer"
    assert deduplicated.loc[3, "assay_transfer_eligible"] == False  # noqa: E712
    assert deduplicated.loc[3, "assay_transfer_ineligibility_reason"] == (
        "untrusted_direct_condition_key"
    )
    assert deduplicated.loc[4, "retrieval_eligible"] == False  # noqa: E712
    assert deduplicated.loc[4, "assay_transfer_ineligibility_reason"] == (
        "missing_pair_bucket"
    )

    build_collapsed_record_stage(
        task_id="test_task",
        records_path=tmp_path / "deduplicated/records.parquet",
        pair_bucket_records_path=tmp_path / "deduplicated/pair_bucket_records.parquet",
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
        duplicate_lineage_path=tmp_path / "deduplicated/duplicates.parquet",
    )
    collapsed = pd.read_parquet(tmp_path / "collapsed/records.parquet")
    shared = collapsed[collapsed["source_record_count"] == 2].iloc[0]
    assert shared["finite_scalar_value"] == 15.0
    assert shared["assay_transfer_eligible"] == True  # noqa: E712
    assert shared["retrieval_source_id"] == "direct_residual"
    singleton_eligibility = dict(
        collapsed[collapsed["source_record_count"] == 1][
            ["finite_scalar_value", "assay_transfer_eligible"]
        ].itertuples(index=False, name=None)
    )
    assert singleton_eligibility == {30.0: True, 40.0: False}


def test_direct_residual_categorical_key_uses_pair_bucket_and_transfers():
    condition = {
        "retrieval_source_id": "direct_residual",
        "condition_group": "no_reported_external_condition",
        "condition_key_status": "none_reported",
        "condition_atoms": [],
        "condition_scope": "none_reported",
        "group_id": "Observed.direct",
        "direct_group_id": "Observed.direct",
        "pair_bucket_key": "categorical-bucket",
        "canonical_pair_fields_json": "{}",
        "assay_transfer_eligible": True,
        "assay_transfer_ineligibility_reason": None,
    }
    first = {
        **_record("r1", "source_a", "CCO", "binary", category="positive"),
        **condition,
        "canonical_endpoint_name": "assay_a",
        "canonical_measurement_scale_id": "binary_scale",
    }
    same_scale = {
        **_record("r2", "source_b", "CCO", "binary", category="positive"),
        **condition,
        "canonical_endpoint_name": "assay_a",
        "canonical_measurement_scale_id": "binary_scale",
    }
    different_scale = {
        **_record("r3", "source_b", "CCO", "binary", category="positive"),
        **condition,
        "canonical_endpoint_name": "assay_a",
        "canonical_measurement_scale_id": "different_binary_scale",
        "pair_bucket_key": "different-categorical-bucket",
    }

    assert _collapse_group_key(first) == _collapse_group_key(same_scale)
    assert _collapse_group_key(first) != _collapse_group_key(different_scale)
    assert _aggregation_method([first, same_scale]) == "categorical_mode"
    collapsed = _aggregate_group(
        "test_task",
        _collapse_group_key(first),
        [first, same_scale],
        method="categorical_mode",
        semantic=None,
        dedup_lineage={},
        preserved_columns=(),
    )
    assert collapsed["canonical_category_id"] == "positive"
    assert collapsed["pair_bucket_key"] == "categorical-bucket"
    assert collapsed["assay_transfer_eligible"] is True


def test_multi_record_relative_direct_residual_stays_semantic():
    records = [
        {
            **_record(record_id, "source_a", "CCO", "continuous", value=value),
            "is_absolute_and_continuous": False,
            "retrieval_source_id": "direct_residual",
            "condition_group": "no_reported_external_condition",
            "condition_key_status": "none_reported",
            "canonical_unit_text": "%",
        }
        for record_id, value in (("r1", 20.0), ("r2", 30.0))
    ]

    assert _collapse_group_key(records[0]) == _collapse_group_key(records[1])
    assert _aggregation_method(records) == "semantic_llm"


def test_single_semantic_record_uses_support_text_without_llm(tmp_path, monkeypatch):
    record = _record(
        "r1",
        "source_a",
        "CCO",
        "semantic",
        support="J3V (SMILES:3818410) showed a short half-life.",
    )
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    pd.DataFrame([record]).to_parquet(records_path, index=False)
    pd.DataFrame([{
        "canonical_record_id": "r1",
        "pair_bucket_key": "semantic-bucket",
        "canonical_pair_fields_json": "{}",
        "assay_transfer_eligible": False,
        "assay_transfer_ineligibility_reason": "free_text",
    }]).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.record_collapse.aggregate_semantic_groups",
        lambda *args, **kwargs: pytest.fail("singleton semantic record called the LLM"),
    )

    manifest = build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
    )

    collapsed = pd.read_parquet(tmp_path / "collapsed/records.parquet").iloc[0]
    assert manifest["summary"]["semantic_groups"] == 0
    assert collapsed["aggregation_method"] == "semantic_support_passthrough"
    assert collapsed["canonical_measurement_text"] == record["support_text"]
    assert collapsed["display_measurement_text"] == "J3V showed a short half-life."
    assert collapsed["pair_bucket_key"] == "semantic-bucket"


def test_deferred_semantic_shell_is_hidden_and_can_be_attached_from_cache(
    tmp_path, monkeypatch
):
    records = [
        _record("r1", "source_a", "CCO", "semantic", support="Reaction observed."),
        _record("r2", "source_b", "CCO", "semantic", support="No reaction."),
    ]
    sidecar = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": "semantic-bucket",
            "canonical_pair_fields_json": '{"endpoint":"reported outcome"}',
            "assay_transfer_eligible": False,
            "assay_transfer_ineligibility_reason": "free_text",
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    cache_path = tmp_path / "semantic_cache.jsonl"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(sidecar).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")

    stored_records = pd.read_parquet(records_path).to_dict("records")
    stored_sidecars = pd.read_parquet(sidecar_path).to_dict("records")
    for record, bucket in zip(stored_records, stored_sidecars, strict=True):
        record.update(bucket)
        record["retrieval_source_id"] = "indirect"
    group_key = _collapse_group_key(stored_records[0])
    payload = _semantic_payload("test_task", group_key, stored_records)
    cache_row = _semantic_cache_row(
        group_key, payload, render_prompt(payload), "test-model"
    )
    cache_row["prompt_sha256"] = hashlib.sha256(
        render_prompt(payload).encode()
    ).hexdigest()
    cache_path.write_text(json.dumps(cache_row) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.record_collapse.aggregate_semantic_groups",
        lambda *args, **kwargs: pytest.fail("deferred build called the LLM"),
    )
    manifest = build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "deferred",
        semantic_model="test-model",
        prior_semantic_paths=(cache_path,),
        defer_semantic_aggregation=True,
    )

    deferred = pd.read_parquet(tmp_path / "deferred/records.parquet").iloc[0]
    semantic = json.loads(
        (tmp_path / "deferred/semantic_aggregation.jsonl").read_text()
    )
    assert manifest["summary"]["semantic_groups"] == 1
    assert manifest["summary"]["semantic_groups_completed"] == 1
    assert manifest["summary"]["semantic_groups_pending"] == 0
    assert deferred["aggregation_status"] == "valid"
    assert deferred["retrieval_eligible"] == True  # noqa: E712
    assert deferred["assay_transfer_eligible"] == False  # noqa: E712
    assert deferred["canonical_measurement_text"] == "A result was reported."
    assert deferred["canonical_unit_text"] == "free-text"
    assert pd.isna(deferred["display_unit_text"])
    assert pd.isna(deferred["finite_scalar_value"])
    assert pd.isna(deferred["direct_label_informativeness"])
    assert semantic["group_id"] == group_key
    assert semantic["input"]["canonical_source_record_ids"] == ["r1", "r2"]

    with pytest.raises(RuntimeError, match="require LLM aggregation"):
        build_collapsed_record_stage(
            task_id="test_task",
            records_path=records_path,
            pair_bucket_records_path=sidecar_path,
            pair_bucket_metadata_path=metadata_path,
            out_dir=tmp_path / "wrong-model",
            semantic_model="other-model",
            prior_semantic_paths=(cache_path,),
        )
    build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "attached",
        semantic_model="test-model",
        prior_semantic_paths=(cache_path,),
    )
    attached = pd.read_parquet(tmp_path / "attached/records.parquet").iloc[0]
    assert attached["canonical_record_id"] == deferred["canonical_record_id"]
    assert attached["retrieval_eligible"] == True  # noqa: E712
    assert attached["aggregation_status"] == "valid"
    assert attached["canonical_measurement_text"] == "A result was reported."
    assert pd.isna(attached["direct_label_informativeness"])
    assert attached["display_measurement_text"] == "A result was reported."


def test_active_pair_bucket_materializer_rejects_collapsed_multi_source_rows():
    pending = {
        "canonical_record_id": "collapsed-1",
        "collapsed_record_id": "collapsed-1",
        "source_id": "multi_source",
        "canonical_smiles": "CCO",
        "canonical_endpoint_name": "reported_outcome",
        "canonical_unit_text": "free-text",
        "canonical_pair_fields_json": '{"canonical_context":"oral"}',
        "pair_bucket_key": '["source_a","reported_outcome","free-text","oral"]',
        "assay_transfer_eligible": False,
        "assay_transfer_ineligibility_reason": "pending_semantic_aggregation",
        "retrieval_eligible": False,
        "measurement_kind": "semantic",
    }
    direct = {
        **pending,
        "canonical_record_id": "collapsed-2",
        "collapsed_record_id": "collapsed-2",
        "source_id": "source_a",
        "pair_bucket_key": None,
        "canonical_pair_fields_json": "{}",
        "retrieval_eligible": True,
        "assay_transfer_ineligibility_reason": (
            "direct_or_contextual_outcome_not_assay_transferable"
        ),
    }
    eligible = {
        **pending,
        "canonical_record_id": "collapsed-3",
        "collapsed_record_id": "collapsed-3",
        "pair_bucket_key": '["source_a","absolute","mg/L","oral"]',
        "canonical_unit_text": "mg/L",
        "retrieval_eligible": True,
        "assay_transfer_eligible": True,
        "assay_transfer_ineligibility_reason": None,
        "measurement_kind": "continuous",
    }

    with pytest.raises(ValueError, match="source_id='multi_source'"):
        materialize_pair_buckets(
            [pending, direct, eligible],
            source_required_fields={"source_a": ("canonical_context",)},
            canonical_record_contract=True,
        )


def test_semantic_responses_are_cached_as_each_request_finishes(tmp_path, monkeypatch):
    records = [
        {
            **_record(f"r{index}", "source_a", "CCO", "semantic"),
            "canonical_measurement_text": text,
        }
        for index, text in enumerate(("Reaction observed.", "No reaction."), 1)
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    cache_path = tmp_path / "semantic_cache.jsonl"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(
        [
            {
                "canonical_record_id": row["canonical_record_id"],
                "pair_bucket_key": "semantic-bucket",
                "canonical_pair_fields_json": "{}",
                "assay_transfer_eligible": False,
                "assay_transfer_ineligibility_reason": "free_text",
            }
            for row in records
        ]
    ).to_parquet(sidecar_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")

    def fake_run_one(group_id, payload, prompt, config):
        return {
            "group_id": group_id,
            "schema_version": "semantic_record_aggregation.v4",
            "prompt_sha256": "test",
            "requested_model": config.model,
            "served_model": config.model,
            "usage": {},
            "response": {"summary": "Reports conflict."},
            "input": dict(payload),
        }

    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.semantic_record_aggregation._run_one",
        fake_run_one,
    )
    build_collapsed_record_stage(
        task_id="test_task",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
        semantic_config=SemanticAggregationConfig(
            api_key="unused",
            model="test-model",
            cache_path=cache_path,
        ),
    )

    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 1


def test_successful_semantic_responses_survive_a_sibling_failure(
    tmp_path, monkeypatch
):
    payload = _semantic_payload(
        "test_task",
        "unused",
        [
            {
                **_record("r1", "source_a", "CCO", "semantic"),
                "retrieval_source_id": "indirect",
                "pair_bucket_key": "semantic-bucket",
                "canonical_pair_fields_json": "{}",
            }
        ],
    )
    cache_path = tmp_path / "semantic_cache.jsonl"

    def fake_run_one(group_id, payload, prompt, config):
        if group_id == "bad":
            raise TimeoutError("test timeout")
        return {
            "group_id": group_id,
            "schema_version": "semantic_record_aggregation.v4",
            "prompt_sha256": "test",
            "requested_model": config.model,
            "served_model": config.model,
            "usage": {},
            "response": {"summary": "A result was reported."},
            "input": dict(payload),
        }

    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.semantic_record_aggregation._run_one",
        fake_run_one,
    )
    with pytest.raises(RuntimeError, match="1 semantic aggregation request"):
        aggregate_semantic_groups(
            {"good": payload, "bad": payload},
            config=SemanticAggregationConfig(
                api_key="unused",
                model="test-model",
                workers=2,
                cache_path=cache_path,
            ),
        )

    cached = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert [row["group_id"] for row in cached] == ["good"]


def test_semantic_retry_includes_the_invalid_response(monkeypatch):
    calls = []
    valid = {"summary": "A result was reported."}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat_json(self, messages):
            calls.append(messages)
            content = {} if len(calls) == 1 else valid
            return {"content": content, "raw_content": json.dumps(content)}

    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.semantic_record_aggregation.OpenAICompatibleClient",
        FakeClient,
    )
    row = _run_one(
        "group",
        {"records": [{"canonical_record_id": "r1"}]},
        "prompt",
        SemanticAggregationConfig(api_key="unused", model="test-model"),
    )

    assert row["attempts"] == 2
    assert [message["role"] for message in calls[1]] == [
        "user",
        "assistant",
        "user",
    ]
    assert calls[1][1]["content"] == "{}"


def test_semantic_groups_run_smallest_prompt_first(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(group_id, payload, prompt, config):
        calls.append(group_id)
        return _semantic_cache_row(group_id, payload, prompt, config.model)

    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.semantic_record_aggregation._run_one",
        fake_run_one,
    )
    small = _semantic_test_payload("small")
    large = _semantic_test_payload("large", padding="x" * 1000)
    aggregate_semantic_groups(
        {
            "large": large,
            "small": small,
        },
        config=SemanticAggregationConfig(
            api_key="unused",
            model="test-model",
            workers=1,
            cache_path=tmp_path / "cache.jsonl",
        ),
    )

    assert calls == ["small", "large"]


def test_distillation_usage_is_charged_to_the_token_ledger(monkeypatch):
    valid = {"summary": "A result was reported."}

    class FakeLedger:
        completed = None

        def reserve(self, request_id, maximum):
            assert maximum > 4096
            return True

        def complete(self, request_id, usage):
            self.completed = usage

    ledger = FakeLedger()

    def fake_llm(prompt, **kwargs):
        assert kwargs["model"] == "gpt-5.4-mini"
        assert kwargs["reasoning_effort"] == "low"
        return {
            "content": json.dumps(valid),
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 45,
                "total_tokens": 168,
            },
            "reasoning": None,
        }

    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.semantic_record_aggregation._distillation_llm",
        lambda: fake_llm,
    )
    row = _run_one(
        "group",
        {"records": [{"canonical_record_id": "r1"}]},
        "prompt",
        SemanticAggregationConfig(
            model="gpt-5.4-mini",
            provider="distillation",
            reasoning_effort="low",
            token_ledger=ledger,
        ),
    )

    assert ledger.completed == {"input_tokens": 123, "output_tokens": 45}
    assert row["usage"] == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }
    assert row["endpoint"] == "therapeutic-tuning/distillation"


def test_budget_exhaustion_preserves_completed_rows(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    small = _semantic_test_payload("small")
    large = _semantic_test_payload("large", padding="x" * 1000)

    class FakeLedger:
        exhausted = False

        def mark_exhausted(self):
            self.exhausted = True

    ledger = FakeLedger()

    def fake_run_one(group_id, payload, prompt, config):
        if group_id == "large":
            raise SemanticAggregationBudgetExhausted(group_id)
        return _semantic_cache_row(group_id, payload, prompt, config.model)

    monkeypatch.setattr(
        "tools.chembl_tool.common.starling.semantic_record_aggregation._run_one",
        fake_run_one,
    )
    with pytest.raises(SemanticAggregationBudgetExhausted):
        aggregate_semantic_groups(
            {
                "small": small,
                "large": large,
            },
            config=SemanticAggregationConfig(
                model="gpt-5.4-mini",
                workers=1,
                cache_path=cache_path,
                token_ledger=ledger,
            ),
        )

    assert ledger.exhausted
    assert [json.loads(line)["group_id"] for line in cache_path.read_text().splitlines()] == [
        "small"
    ]


def _semantic_cache_row(group_id, payload, prompt, model):
    return {
        "group_id": group_id,
        "schema_version": "semantic_record_aggregation.v4",
        "prompt_sha256": "test",
        "requested_model": model,
        "served_model": model,
        "usage": {},
        "response": {"summary": "A result was reported."},
        "input": dict(payload),
    }


def _semantic_test_payload(group_id, *, padding=""):
    record = {
        **_record("r1", "source_a", "CCO", "semantic"),
        "retrieval_source_id": "indirect",
        "pair_bucket_key": "semantic-bucket",
        "canonical_pair_fields_json": "{}",
        "support_text": "evidence" + padding,
    }
    return _semantic_payload(
        "test_task",
        group_id,
        [record],
        semantic_source_columns={"source_a": ("support_text",)},
    )


def _record(
    record_id: str,
    source_id: str,
    smiles: str,
    kind: str,
    *,
    value: float | None = None,
    category: str | None = None,
    support: str = "evidence",
) -> dict:
    return {
        "canonical_record_id": record_id,
        "source_id": source_id,
        "source_record_id": record_id,
        "source_row_number": int(record_id[1:]),
        "canonical_smiles": smiles,
        "retrieval_eligible": True,
        "group_id": "Observed.direct" if source_id == "direct" else "Observed.indirect",
        "measurement_kind": kind,
        "finite_scalar_value": value,
        "is_absolute_and_continuous": kind == "continuous" and value is not None,
        "variation_value": None,
        "canonical_measurement_scale_id": "test_scale" if category else None,
        "canonical_category_id": category,
        "canonical_category_rank": {"low": 0, "high": 1, "positive": 1}.get(category),
        "canonical_measurement_text": category or str(value),
        "canonical_unit_text": (
            "free-text"
            if kind in {"semantic", "non_scalar"}
            else "relative-scalar"
            if kind == "relative"
            else "mg/L"
        ),
        "canonical_endpoint_name": "test_endpoint",
        "support_text": support,
        "pmid": None,
        "doi": None,
        "qualifying_conditions": "discarded non-key context",
    }


def _direct_mapping(records):
    return [
        {
            "canonical_record_id": row["canonical_record_id"],
            "source_id": row["source_id"],
            "source_record_id": row["source_record_id"],
            "direct_vote_status": "counted",
            "direct_vote_reason": "test",
            "direct_vote_label": 1,
            "direct_vote_unit_id": f"direct:{row['source_row_number']}",
            "condition_group": "no_reported_external_condition",
            "condition_atoms": [],
            "condition_scope": "none_reported",
            "condition_key_status": "none_reported",
            "retrieval_source_id": "direct_vote",
            "direct_group_id": "Observed.direct",
            "dedup_status": "pending",
        }
        for row in records
        if row["source_id"] == "direct"
    ]
