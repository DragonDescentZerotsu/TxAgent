from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing import (
    build_exact_unit_review as review_builder,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing import (
    run_exact_unit_review as review_runner,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.build_exact_unit_review import (
    build_unit_inventory,
    load_review_packet_manifest,
    validate_review_completion,
    write_review_packets,
    write_reviewed_decisions,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.plan_measurement_resolution import (
    ENDPOINT_RECEIPT_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_exact_unit_mapping import (
    compile_exact_unit_mapping,
)


def _input_rows() -> list[dict]:
    return [
        {
            "cleaned_record_id": "a",
            "source_row_uid": "uid-a",
            "source_id": "fixed_mutation",
            "canonical_endpoint_name": "micronucleus_assay",
            "measurement_resolution_route": "accept",
            "measurement_resolution_exact_measurement": "2",
            "measurement_resolution_exact_unit": "µM",
            "measurement_text": "2",
            "unit_text": "µM",
            "endpoint_name": "micronucleus_assay",
            "support_text": "source exact context",
        },
        {
            "cleaned_record_id": "b",
            "source_row_uid": "uid-b",
            "source_id": "fixed_mutation",
            "canonical_endpoint_name": "micronucleus_assay",
            "measurement_resolution_route": "extract",
            "measurement_resolution_exact_measurement": None,
            "measurement_resolution_exact_unit": None,
            "measurement_text": "about three micromolar",
            "unit_text": "µM",
            "endpoint_name": "micronucleus_assay",
            "support_text": "LLM context",
        },
        {
            "cleaned_record_id": "c",
            "source_row_uid": "uid-c",
            "source_id": "premutagenic_damage",
            "canonical_endpoint_name": "comet_assay",
            "measurement_resolution_route": "extract",
            "measurement_resolution_exact_measurement": None,
            "measurement_resolution_exact_unit": None,
            "measurement_text": "four cells per plate",
            "unit_text": "cells/plate",
            "endpoint_name": "comet_assay",
            "support_text": "second LLM context",
        },
    ]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cleaned = tmp_path / "cleaned.parquet"
    resolution = tmp_path / "resolution.parquet"
    resolved = [
        {
            "cleaned_record_id": key,
            "source_row_uid": f"uid-{key}",
            "source_id": source,
            "status": "ok",
            "measurements_json": json.dumps(
                [{"measurement": measurement, "unit": unit}]
            ),
        }
        for key, source, measurement, unit in (
            ("b", "fixed_mutation", "3", "µM"),
            ("c", "premutagenic_damage", "4", "cells/plate"),
        )
    ]
    pq.write_table(pa.Table.from_pylist(_input_rows()), cleaned)
    pq.write_table(pa.Table.from_pylist(resolved), resolution)
    return cleaned, resolution


def _review_decision(item: dict[str, object]) -> dict[str, object]:
    endpoint = str(item["canonical_endpoint"])
    unit = str(item["input_unit"])
    common = {
        "canonical_endpoint": endpoint,
        "input_unit": unit,
        "review_basis": "source-grounded fixture review",
    }
    if unit == "cells/plate":
        return {**common, "action": "exclude"}
    return {
        **common,
        "action": "map",
        "mapping_kind": "identity",
        "canonical_unit": unit,
    }


def _review_client() -> SimpleNamespace:
    def create(**request):
        user = json.loads(request["messages"][1]["content"])
        decisions = [_review_decision(item) for item in user["items"]]
        payload = {
            "model": review_runner.MODEL,
            "choices": [{"message": {"content": json.dumps({"decisions": decisions})}}],
        }
        return SimpleNamespace(model_dump=lambda mode="json": payload)

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(
        base_url=review_runner.BASE_URL,
        chat=SimpleNamespace(completions=completions),
    )


def _write_reviews(manifest_path: Path, path: Path) -> None:
    receipt = path.parent / "deepseek_host_endpoint_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "receipt_version": ENDPOINT_RECEIPT_VERSION,
                "checked_at_utc": "2026-09-08T00:00:00Z",
                "live_checks": {
                    "generation": {
                        "http_status": 200,
                        "returned_model": review_runner.MODEL,
                        "sha256": "a" * 64,
                    },
                    "model_info": {
                        "architectures": ["DeepseekV4ForCausalLM"],
                        "http_status": 200,
                        "model_path": review_runner.MODEL,
                        "model_type": "deepseek_v4",
                        "served_model_name": review_runner.MODEL,
                        "sha256": "b" * 64,
                    },
                    "models": {
                        "http_status": 200,
                        "ids": [review_runner.MODEL],
                        "sha256": "c" * 64,
                    },
                },
                "selected_endpoint": {
                    "host": "dgx027",
                    "provider": review_runner.PROVIDER,
                    "base_url": review_runner.BASE_URL,
                    "requested_model": review_runner.MODEL,
                    "returned_model": review_runner.MODEL,
                    "credential_env": review_runner.CREDENTIAL_ENV,
                    "max_concurrency": review_runner.MAX_CONCURRENCY,
                },
            }
        )
        + "\n"
    )
    review_runner.run_review(
        manifest_path,
        path.with_suffix(".cache.jsonl"),
        path,
        endpoint_receipt_path=receipt,
        client=_review_client(),
    )


def _allow_fixture_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_builder, "validate_compiled_mapping", lambda _: None)


def test_inventory_combines_exact_and_llm_units_with_review_evidence(
    tmp_path, monkeypatch
):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    inventory = build_unit_inventory(cleaned, resolution)
    pairs = {
        (row["canonical_endpoint"], row["input_unit"]): row
        for row in inventory["observed_pairs"]
    }
    shared = pairs[("micronucleus_assay", "µM")]
    assert shared["origins"] == ["llm", "source_exact"]
    assert shared["rows"] == 2
    assert shared["origin_counts"] == {"llm": 1, "source_exact": 1}
    assert {row["origin"] for row in shared["representative_contexts"]} == {
        "llm",
        "source_exact",
    }
    assert shared["parser_output"]["canonical"] == "µM"
    assert pairs[("comet_assay", "cells/plate")]["parser_output"]["unknown_tokens"] == [
        "plate"
    ]


def test_inventory_rejects_partial_measurement_resolution(tmp_path, monkeypatch):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    rows = pq.read_table(resolution).slice(0, 1)
    pq.write_table(rows, resolution)
    with pytest.raises(ValueError, match="extract rows missing measurement resolution"):
        build_unit_inventory(cleaned, resolution)


def test_inventory_rejects_intermediate_candidate_selection(tmp_path):
    cleaned, resolution = _write_inputs(tmp_path)
    resolution.with_suffix(".manifest.json").write_text(
        json.dumps({"generation_version": "ames_measurement_candidate_selection.v1"})
    )
    with pytest.raises(ValueError, match="compiled AMES mapping version mismatch"):
        build_unit_inventory(cleaned, resolution)


def test_packets_reject_upstream_digest_drift(tmp_path, monkeypatch):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    work = tmp_path / "review"
    write_review_packets(cleaned, resolution, work, packet_size=1)
    cleaned.write_bytes(b"drifted after packet freeze")
    with pytest.raises(ValueError, match="input digest drift for cleaned_records"):
        load_review_packet_manifest(work / "manifest.json")


def test_packets_reject_self_consistent_inventory_rewrite(tmp_path, monkeypatch):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    work = tmp_path / "review"
    manifest_path = work / "manifest.json"
    write_review_packets(cleaned, resolution, work, packet_size=1)
    manifest = json.loads(manifest_path.read_text())
    inventory_path = work / manifest["inventory"]["path"]
    inventory = json.loads(inventory_path.read_text())
    inventory["mapping_scale_default"] = "forged"
    inventory_path.write_text(json.dumps(inventory))
    manifest["inventory"]["sha256"] = file_sha256(inventory_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from frozen inputs"):
        load_review_packet_manifest(manifest_path)


@pytest.mark.parametrize("asset", ["inventory", "packet"])
def test_packets_reject_assets_outside_packet_directory(tmp_path, monkeypatch, asset):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    work = tmp_path / "review"
    manifest_path = work / "manifest.json"
    write_review_packets(cleaned, resolution, work, packet_size=1)
    manifest = json.loads(manifest_path.read_text())
    if asset == "inventory":
        source = work / manifest["inventory"]["path"]
        outside = tmp_path / "outside-inventory.json"
        outside.write_bytes(source.read_bytes())
        manifest["inventory"] = {"path": str(outside), "sha256": file_sha256(outside)}
    else:
        spec = manifest["packets"][0]
        source = work / spec["path"]
        outside = tmp_path / "outside-packet.json"
        outside.write_bytes(source.read_bytes())
        spec.update(path="../outside-packet.json", sha256=file_sha256(outside))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes the packet directory"):
        load_review_packet_manifest(manifest_path)


@pytest.mark.parametrize("tamper", ["packet_size", "review_scale", "duplicate_id"])
def test_packets_reject_false_grouping_claims(tmp_path, monkeypatch, tamper):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    work = tmp_path / "review"
    manifest_path = work / "manifest.json"
    write_review_packets(cleaned, resolution, work, packet_size=1)
    manifest = json.loads(manifest_path.read_text())
    if tamper == "packet_size":
        manifest["packet_size"] = 2
    else:
        index = 0 if tamper == "review_scale" else 1
        spec = manifest["packets"][index]
        packet_path = work / spec["path"]
        packet = json.loads(packet_path.read_text())
        if tamper == "review_scale":
            packet["review_scale_default"] = "1000"
        else:
            packet["packet_id"] = "packet_0001"
            spec["packet_id"] = "packet_0001"
        packet_path.write_text(json.dumps(packet))
        spec["sha256"] = file_sha256(packet_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_review_packet_manifest(manifest_path)


@pytest.mark.parametrize("failure_stage", ["write", "publish"])
def test_manifest_failure_leaves_clean_retryable_outputs(
    tmp_path, monkeypatch, failure_stage
):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    packets = tmp_path / "review"
    write_review_packets(cleaned, resolution, packets, packet_size=1)
    reviews = tmp_path / "reviews.jsonl"
    _write_reviews(packets / "manifest.json", reviews)
    output = tmp_path / "decisions.json"
    completion = output.with_suffix(".manifest.json")
    completion_temporary = completion.with_suffix(f"{completion.suffix}.tmp")
    original_write = review_builder._write_json
    original_replace = review_builder.os.replace

    def fail_completion_write(path, payload):
        if path == completion_temporary:
            raise OSError("injected completion-manifest write failure")
        original_write(path, payload)

    def fail_completion_publish(source, target):
        if Path(target) == completion:
            raise OSError("injected completion-manifest publish failure")
        original_replace(source, target)

    if failure_stage == "write":
        monkeypatch.setattr(review_builder, "_write_json", fail_completion_write)
    else:
        monkeypatch.setattr(review_builder.os, "replace", fail_completion_publish)
    with pytest.raises(OSError, match="injected completion-manifest"):
        write_reviewed_decisions(packets / "manifest.json", reviews, output)
    assert not output.exists()
    assert not completion.exists()
    assert not output.with_suffix(f"{output.suffix}.tmp").exists()
    assert not completion_temporary.exists()

    monkeypatch.setattr(review_builder, "_write_json", original_write)
    monkeypatch.setattr(review_builder.os, "replace", original_replace)
    result = write_reviewed_decisions(packets / "manifest.json", reviews, output)
    assert result["counts"]["unreviewed_items"] == 0
    assert output.is_file() and completion.is_file()


def test_packets_and_consolidation_prove_complete_fail_closed_review(
    tmp_path, monkeypatch
):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    work = tmp_path / "review"
    manifest = write_review_packets(cleaned, resolution, work, packet_size=1)
    loaded, inventory = load_review_packet_manifest(work / "manifest.json")
    assert manifest == loaded
    assert loaded["packet_count"] == 2
    assert max(row["item_count"] for row in loaded["packets"]) == 1
    assert loaded["item_count"] == len(inventory["observed_pairs"])

    incomplete = tmp_path / "incomplete.jsonl"
    _write_reviews(work / "manifest.json", incomplete)
    incomplete.write_text(incomplete.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="review_decisions hash mismatch"):
        write_reviewed_decisions(
            work / "manifest.json", incomplete, tmp_path / "incomplete-decisions.json"
        )

    reviews = tmp_path / "reviews.jsonl"
    output = tmp_path / "decisions.json"
    _write_reviews(work / "manifest.json", reviews)
    completion = write_reviewed_decisions(work / "manifest.json", reviews, output)
    assert completion["counts"]["unreviewed_items"] == 0
    assert validate_review_completion(output) == completion
    compiled = compile_exact_unit_mapping(output)
    assert compiled["ames_v10_contract"]["observed_pair_count"] == 2
    mapped = next(row for row in compiled["entries"] if row["input_unit"] == "µM")
    assert mapped["scale"] == "1"


def test_unsigned_fresh_reviews_cannot_be_consolidated(tmp_path, monkeypatch):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    work = tmp_path / "review"
    write_review_packets(cleaned, resolution, work, packet_size=1)
    reviews = tmp_path / "forged.jsonl"
    reviews.write_text(
        json.dumps(
            {
                "canonical_endpoint": "comet_assay",
                "input_unit": "cells/plate",
                "action": "exclude",
                "review_basis": "plausible but unsigned",
                "reviewer": review_runner.MODEL,
            }
        )
        + "\n"
    )
    with pytest.raises(FileNotFoundError):
        write_reviewed_decisions(
            work / "manifest.json", reviews, tmp_path / "forged-decisions.json"
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "truncated",
        "packet_hash",
        "run_receipt_hash",
        "validation",
        "decision_divergence",
    ],
)
def test_completion_manifest_rejects_false_provenance(tmp_path, monkeypatch, tamper):
    _allow_fixture_resolution(monkeypatch)
    cleaned, resolution = _write_inputs(tmp_path)
    packets = tmp_path / "review"
    write_review_packets(cleaned, resolution, packets, packet_size=1)
    reviews = tmp_path / "reviews.jsonl"
    _write_reviews(packets / "manifest.json", reviews)
    output = tmp_path / "decisions.json"
    completion = write_reviewed_decisions(packets / "manifest.json", reviews, output)
    if tamper == "truncated":
        completion = {
            "counts": completion["counts"],
            "reviewed_decision_asset": completion["reviewed_decision_asset"],
        }
    elif tamper == "packet_hash":
        completion["packet_manifest"]["sha256"] = "0" * 64
    elif tamper == "run_receipt_hash":
        completion["review_run_receipt"]["sha256"] = "0" * 64
    elif tamper == "validation":
        completion["validations"]["exact_pair_coverage"] = False
    else:
        decisions = json.loads(output.read_text())
        decisions["decisions"][0]["review_basis"] = "silently changed"
        output.write_text(json.dumps(decisions))
        completion["reviewed_decision_asset"]["sha256"] = file_sha256(output)
    output.with_suffix(".manifest.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError):
        validate_review_completion(output)
