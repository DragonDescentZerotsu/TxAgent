from __future__ import annotations

import ast
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing import (
    build_exact_unit_review as review_builder,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing import (
    run_exact_unit_review as runner,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.build_exact_unit_review import (
    INVENTORY_VERSION,
    PACKET_MANIFEST_VERSION,
    PACKET_VERSION,
    write_reviewed_decisions,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.plan_measurement_resolution import (
    ENDPOINT_RECEIPT_VERSION,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _endpoint_receipt(root: Path) -> Path:
    path = root / "deepseek_host_endpoint_receipt.json"
    payload = {
        "receipt_version": ENDPOINT_RECEIPT_VERSION,
        "checked_at_utc": "2026-09-08T00:00:00Z",
        "live_checks": {
            "generation": {
                "http_status": 200,
                "returned_model": runner.MODEL,
                "sha256": "a" * 64,
            },
            "model_info": {
                "architectures": ["DeepseekV4ForCausalLM"],
                "http_status": 200,
                "model_path": runner.MODEL,
                "model_type": "deepseek_v4",
                "served_model_name": runner.MODEL,
                "sha256": "b" * 64,
            },
            "models": {
                "http_status": 200,
                "ids": [runner.MODEL],
                "sha256": "c" * 64,
            },
        },
        "selected_endpoint": {
            "host": "dgx027",
            "provider": runner.PROVIDER,
            "base_url": runner.BASE_URL,
            "requested_model": runner.MODEL,
            "returned_model": runner.MODEL,
            "credential_env": runner.CREDENTIAL_ENV,
            "max_concurrency": runner.MAX_CONCURRENCY,
        },
    }
    if not path.exists():
        _write_json(path, payload)
    return path


def _run_review(manifest: Path, cache: Path, output: Path, *, client=None):
    return runner.run_review(
        manifest,
        cache,
        output,
        endpoint_receipt_path=_endpoint_receipt(output.parent),
        client=client,
    )


@pytest.fixture(autouse=True)
def _rebuild_synthetic_inventory(monkeypatch):
    def rebuild(cleaned_path, _resolution_path):
        inventory_path = Path(cleaned_path).parent / "packets/inventory.json"
        return json.loads(inventory_path.read_text())

    monkeypatch.setattr(review_builder, "build_unit_inventory", rebuild)


def _item(index: int, unit: str | None = None) -> dict[str, object]:
    selected_unit = unit or f"unit-{index:04d}"
    return {
        "canonical_endpoint": "micronucleus_assay",
        "input_unit": selected_unit,
        "origins": ["source_exact"],
        "rows": 1,
        "origin_counts": {"source_exact": 1},
        "representative_contexts": [
            {
                "origin": "source_exact",
                "source_unit": selected_unit,
                "resolved_unit": selected_unit,
                "support_text": f"The result was reported in {selected_unit}.",
            }
        ],
        "parser_output": {"canonical": selected_unit, "unknown_tokens": []},
    }


def _fixture_inventory(
    tmp_path: Path, items: list[dict[str, object]]
) -> tuple[dict[str, object], str]:
    cleaned, resolution = tmp_path / "cleaned.bin", tmp_path / "resolution.bin"
    cleaned.write_text("cleaned\n")
    resolution.write_text("resolution\n")
    inventory = {
        "version": INVENTORY_VERSION,
        "task": "ames",
        "inputs": {
            "cleaned_records": {"path": str(cleaned), "sha256": file_sha256(cleaned)},
            "measurement_resolution": {
                "path": str(resolution),
                "sha256": file_sha256(resolution),
            },
        },
        "observed_pairs": items,
    }
    inventory_path = tmp_path / "packets/inventory.json"
    _write_json(inventory_path, inventory)
    return inventory, file_sha256(inventory_path)


def _fixture_packet_specs(
    tmp_path: Path, items: list[dict[str, object]], size: int, inventory_hash: str
) -> list[dict[str, object]]:
    packet_specs = []
    for start in range(0, len(items), size):
        packet_id = f"packet_{len(packet_specs) + 1:04d}"
        selected = items[start : start + size]
        relative = Path("packet_files") / f"{packet_id}.json"
        packet_path = tmp_path / "packets" / relative
        _write_json(
            packet_path,
            {
                "version": PACKET_VERSION,
                "task": "ames",
                "packet_id": packet_id,
                "inventory_sha256": inventory_hash,
                "item_count": len(selected),
                "review_scale_default": "1",
                "items": selected,
            },
        )
        packet_specs.append(
            {
                "packet_id": packet_id,
                "path": str(relative),
                "sha256": file_sha256(packet_path),
                "item_count": len(selected),
                "first_key": [
                    selected[0]["canonical_endpoint"],
                    selected[0]["input_unit"],
                ],
                "last_key": [
                    selected[-1]["canonical_endpoint"],
                    selected[-1]["input_unit"],
                ],
            }
        )
    return packet_specs


def _manifest(tmp_path: Path, items: list[dict[str, object]], size: int) -> Path:
    inventory, inventory_hash = _fixture_inventory(tmp_path, items)
    packet_specs = _fixture_packet_specs(tmp_path, items, size, inventory_hash)
    manifest = {
        "version": PACKET_MANIFEST_VERSION,
        "task": "ames",
        "inventory": {"path": "inventory.json", "sha256": inventory_hash},
        "inputs": inventory["inputs"],
        "packet_size": size,
        "packet_count": len(packet_specs),
        "item_count": len(items),
        "unreviewed_item_count": len(items),
        "packets": packet_specs,
        "validations": {
            "complete_disjoint_inventory": True,
            "maximum_packet_items": 500,
        },
    }
    path = tmp_path / "packets/manifest.json"
    _write_json(path, manifest)
    return path


def _decision(item: dict[str, object]) -> dict[str, str]:
    endpoint, unit = str(item["canonical_endpoint"]), str(item["input_unit"])
    common = {
        "canonical_endpoint": endpoint,
        "input_unit": unit,
        "review_basis": "source context explicitly supplies this unit",
    }
    if unit == "not reported":
        return {**common, "action": "exclude"}
    if unit == "mutants per cell":
        return {
            **common,
            "action": "map",
            "mapping_kind": "spelling_alias",
            "canonical_unit": "mutants/cell",
            "equivalence_evidence": "source uses per cell for the slash spelling",
        }
    return {
        **common,
        "action": "map",
        "mapping_kind": "identity",
        "canonical_unit": unit,
    }


def _client(
    transform=None,
    *,
    returned_model: str = runner.MODEL,
    served_provider: str | None = None,
    base_url: str = runner.BASE_URL,
):
    calls: list[dict[str, object]] = []

    def create(**request):
        calls.append(request)
        user = json.loads(request["messages"][1]["content"])
        decisions = [_decision(item) for item in user["items"]]
        if transform is not None:
            decisions = transform(decisions)
        content = json.dumps({"decisions": decisions})
        payload = {
            "id": f"response-{len(calls)}",
            "model": returned_model,
            "choices": [{"message": {"content": content}}],
        }
        if served_provider is not None:
            payload["provider"] = served_provider
        return SimpleNamespace(model_dump=lambda mode="json": payload)

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(
        base_url=base_url, chat=SimpleNamespace(completions=completions)
    )
    return client, calls


def _canonical_json_hash(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def test_runner_pins_route_caches_and_writes_consolidatable_reviews(tmp_path):
    items = [
        _item(1, "mutants/cell"),
        _item(2, "mutants per cell"),
        _item(3, "not reported"),
    ]
    manifest = _manifest(tmp_path, items, 2)
    cache, reviews = tmp_path / "cache.jsonl", tmp_path / "reviews.jsonl"
    client, calls = _client()
    result = _run_review(manifest, cache, reviews, client=client)
    assert result["decision_counts"] == {
        "exclude": 1,
        "identity": 1,
        "spelling_alias": 1,
    }
    assert result["api_requests"] == 2 and result["cache_hits"] == 0
    seen = []
    for request in calls:
        assert request["model"] == runner.MODEL
        assert "extra_body" not in request
        batch = json.loads(request["messages"][1]["content"])["items"]
        assert len(batch) <= 500
        seen.extend((row["canonical_endpoint"], row["input_unit"]) for row in batch)
    assert len(seen) == len(set(seen)) == 3
    events = [json.loads(line) for line in cache.read_text().splitlines()]
    assert all(
        row["request_sha256"] == _canonical_json_hash(row["request"]) for row in events
    )
    assert all(
        row["response_sha256"] == _canonical_json_hash(row["response"])
        for row in events
    )
    assert {(row["returned_model"], row["served_provider"]) for row in events} == {
        (runner.MODEL, "")
    }
    assert {row["credential_env"] for row in events} == {""}
    assert {(row["attempt_status"], row["rejection_reason"]) for row in events} == {
        ("accepted", None)
    }
    receipt_path = runner.review_receipt_path(reviews)
    receipt = runner.validate_review_run_receipt(reviews)
    assert receipt["route"] == {
        "base_url": runner.BASE_URL,
        "provider": "local",
        "model": runner.MODEL,
        "returned_model": runner.MODEL,
        "credential_env": "",
        "max_concurrency": 512,
    }
    assert receipt["endpoint_receipt"]["sha256"] == file_sha256(
        _endpoint_receipt(tmp_path)
    )
    assert receipt["cache"]["sha256"] == file_sha256(cache)
    assert file_sha256(receipt_path) == result["receipt_sha256"]
    write_reviewed_decisions(manifest, reviews, tmp_path / "decisions.json")


def test_runner_replays_cache_without_mutation_or_api_call(tmp_path):
    manifest = _manifest(tmp_path, [_item(1), _item(2)], 1)
    cache = tmp_path / "cache.jsonl"
    client, _ = _client()
    first = tmp_path / "first.jsonl"
    _run_review(manifest, cache, first, client=client)
    frozen = cache.read_bytes()

    def fail(**_):
        raise AssertionError("cache replay called the API")

    replay = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    second = tmp_path / "second.jsonl"
    replay.base_url = runner.BASE_URL
    result = _run_review(manifest, cache, second, client=replay)
    assert result["api_requests"] == 0 and result["cache_hits"] == 2
    assert cache.read_bytes() == frozen
    assert second.read_bytes() == first.read_bytes()


def test_runner_receipt_rejects_cache_tampering(tmp_path):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    cache, reviews = tmp_path / "cache.jsonl", tmp_path / "reviews.jsonl"
    client, _ = _client()
    _run_review(manifest, cache, reviews, client=client)
    event = json.loads(cache.read_text())
    event["response"]["provider"] = "forged"
    event["response_sha256"] = _canonical_json_hash(event["response"])
    event["served_provider"] = "forged"
    cache.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        runner.validate_review_run_receipt(reviews)


def test_runner_receipt_publish_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    cache, reviews = tmp_path / "cache.jsonl", tmp_path / "reviews.jsonl"
    receipt = runner.review_receipt_path(reviews)
    original_replace = runner.os.replace

    def fail_receipt_publish(source, target):
        if Path(target) == receipt:
            raise OSError("injected run-receipt publish failure")
        original_replace(source, target)

    monkeypatch.setattr(runner.os, "replace", fail_receipt_publish)
    client, _ = _client()
    with pytest.raises(OSError, match="run-receipt publish failure"):
        _run_review(manifest, cache, reviews, client=client)
    assert cache.is_file()
    assert not reviews.exists() and not receipt.exists()
    assert not reviews.with_suffix(".jsonl.tmp").exists()
    assert not receipt.with_suffix(".json.tmp").exists()


def test_runner_rejects_frozen_input_drift_before_api_call(tmp_path):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    (tmp_path / "resolution.bin").write_text("drifted\n")
    client, calls = _client()
    cache, output = tmp_path / "cache.jsonl", tmp_path / "reviews.jsonl"
    with pytest.raises(
        ValueError, match="input digest drift for measurement_resolution"
    ):
        _run_review(manifest, cache, output, client=client)
    assert calls == []
    assert not cache.exists()
    assert not output.exists()


def test_runner_rejects_host_receipt_drift_before_api_call(tmp_path):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    receipt = _endpoint_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["selected_endpoint"]["returned_model"] = "wrong/model"
    _write_json(receipt, payload)
    client, calls = _client()
    with pytest.raises(ValueError, match="returned_model"):
        runner.run_review(
            manifest,
            tmp_path / "cache.jsonl",
            tmp_path / "reviews.jsonl",
            endpoint_receipt_path=receipt,
            client=client,
        )
    assert calls == []
    assert not (tmp_path / "cache.jsonl").exists()


def test_runner_rejects_missing_live_architecture_before_api_call(tmp_path):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    receipt = _endpoint_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    del payload["live_checks"]["model_info"]["architectures"]
    _write_json(receipt, payload)
    client, calls = _client()
    with pytest.raises(ValueError, match="model_info.architectures"):
        runner.run_review(
            manifest,
            tmp_path / "cache.jsonl",
            tmp_path / "reviews.jsonl",
            endpoint_receipt_path=receipt,
            client=client,
        )
    assert calls == []


@pytest.mark.parametrize("failure", ["missing", "duplicate", "scale"])
def test_runner_fails_closed_on_invalid_decision_sets(tmp_path, failure):
    manifest = _manifest(tmp_path, [_item(1), _item(2)], 2)

    def corrupt(rows):
        if failure == "missing":
            return rows[:-1]
        if failure == "duplicate":
            return [*rows, rows[0]]
        return [{**rows[0], "scale": "1000"}, *rows[1:]]

    client, _ = _client(corrupt)
    output = tmp_path / "reviews.jsonl"
    with pytest.raises(ValueError):
        _run_review(manifest, tmp_path / "cache.jsonl", output, client=client)
    assert not output.exists()
    event = json.loads((tmp_path / "cache.jsonl").read_text())
    assert event["attempt_status"] == "rejected"
    assert event["rejection_reason"].startswith("ValueError:")


def test_runner_retries_rejected_attempt_then_replays_accepted_cache(tmp_path):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    cache = tmp_path / "cache.jsonl"
    bad, _ = _client(lambda rows: [{**rows[0], "scale": "1000"}])
    with pytest.raises(ValueError, match="identity mapping"):
        _run_review(manifest, cache, tmp_path / "bad.jsonl", client=bad)

    good, calls = _client()
    accepted = tmp_path / "accepted.jsonl"
    result = _run_review(manifest, cache, accepted, client=good)
    assert result["api_requests"] == len(calls) == 1
    events = [json.loads(line) for line in cache.read_text().splitlines()]
    assert [event["attempt_status"] for event in events] == ["rejected", "accepted"]
    frozen = cache.read_bytes()

    def fail(**_):
        raise AssertionError("accepted cache replay called the API")

    replay = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    replayed = tmp_path / "replayed.jsonl"
    replay.base_url = runner.BASE_URL
    replay_result = _run_review(manifest, cache, replayed, client=replay)
    assert replay_result["api_requests"] == 0 and replay_result["cache_hits"] == 1
    assert cache.read_bytes() == frozen
    assert replayed.read_bytes() == accepted.read_bytes()


def test_runner_invalidates_cross_packet_alias_cycle_before_retry(tmp_path):
    manifest = _manifest(tmp_path, [_item(1), _item(2)], 1)
    cache, output = tmp_path / "cache.jsonl", tmp_path / "reviews.jsonl"

    def cycle(rows):
        changed = []
        for row in rows:
            target = "unit-0002" if row["input_unit"] == "unit-0001" else "unit-0001"
            changed.append(
                {
                    **row,
                    "mapping_kind": "spelling_alias",
                    "canonical_unit": target,
                    "equivalence_evidence": "claimed spelling equivalence",
                }
            )
        return changed

    bad, _ = _client(cycle)
    with pytest.raises(ValueError, match="lacks observed identity target"):
        _run_review(manifest, cache, output, client=bad)
    assert not output.exists()
    first_events = [json.loads(line) for line in cache.read_text().splitlines()]
    assert [event["attempt_status"] for event in first_events] == [
        "accepted",
        "accepted",
        "invalidated",
        "invalidated",
    ]

    good, calls = _client()
    result = _run_review(manifest, cache, output, client=good)
    assert result["api_requests"] == len(calls) == 2
    assert output.is_file()


@pytest.mark.parametrize(
    ("model", "provider"),
    [("deepseek/wrong", None), (runner.MODEL, "OpenInference")],
)
def test_runner_fails_closed_on_returned_route_drift(tmp_path, model, provider):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    client, _ = _client(returned_model=model, served_provider=provider)
    cache = tmp_path / "cache.jsonl"
    output = tmp_path / "reviews.jsonl"
    with pytest.raises(ValueError, match="response route mismatch"):
        _run_review(manifest, cache, output, client=client)
    assert not output.exists()
    event = json.loads(cache.read_text())
    assert event["attempt_status"] == "rejected"
    assert event["response"].get("provider") == provider
    good, calls = _client()
    result = _run_review(manifest, cache, output, client=good)
    assert result["api_requests"] == len(calls) == 1
    assert [
        json.loads(line)["attempt_status"] for line in cache.read_text().splitlines()
    ] == [
        "rejected",
        "accepted",
    ]


def test_runner_rejects_wrong_client_base_url_before_request(tmp_path):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    client, calls = _client(base_url="http://127.0.0.1:50001/v1")
    with pytest.raises(ValueError, match="client base URL mismatch"):
        _run_review(
            manifest,
            tmp_path / "cache.jsonl",
            tmp_path / "reviews.jsonl",
            client=client,
        )
    assert calls == []


def test_runner_builds_credential_free_local_client_with_512_connections(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path, [_item(1)], 1)
    client, calls = _client()
    client.with_options = lambda **_: client
    seen = {}

    def load_client(**kwargs):
        seen.update(kwargs)
        return client, ""

    monkeypatch.setattr(runner, "openai_compatible_client", load_client)
    _run_review(manifest, tmp_path / "cache.jsonl", tmp_path / "reviews.jsonl")
    assert len(calls) == 1
    assert seen == {
        "base_url": runner.BASE_URL,
        "provider": "local",
        "env_file": None,
        "max_connections": 512,
    }


def test_runner_submits_uncached_packets_concurrently(tmp_path):
    manifest = _manifest(tmp_path, [_item(1), _item(2)], 1)
    client, calls = _client()
    original = client.chat.completions.create
    rendezvous = threading.Barrier(2)

    def create(**request):
        rendezvous.wait(timeout=5)
        return original(**request)

    client.chat.completions.create = create
    result = _run_review(
        manifest, tmp_path / "cache.jsonl", tmp_path / "reviews.jsonl", client=client
    )
    assert result["api_requests"] == len(calls) == 2


def test_request_deduplicates_large_observed_alias_catalog():
    rows = [runner._wire_item(_item(index)) for index in range(4_000)]
    batches = [
        (f"packet_{start // 500 + 1:04d}", rows[start : start + 500])
        for start in range(0, len(rows), 500)
    ]
    requests, _ = runner._review_plan(batches, "a" * 64, runner.DEFAULT_MAX_TOKENS)
    request = requests["packet_0001"]
    user = json.loads(request["messages"][1]["content"])
    observed = user["allowed_alias_targets_by_endpoint"]["micronucleus_assay"]
    assert observed == sorted(item["input_unit"] for item in rows)
    assert all("allowed_alias_targets" not in item for item in user["items"])
    assert len(request["messages"][1]["content"].encode()) < 1_000_000


def test_manifest_partition_caps_disjoint_batches_at_500(tmp_path):
    manifest = _manifest(tmp_path, [_item(index) for index in range(501)], 500)
    _, batches, keys = runner.load_review_batches(manifest)
    assert [len(items) for _, items in batches] == [500, 1]
    assert len(keys) == len(set(keys)) == 501


def test_runner_functions_respect_repository_length_limit():
    tree = ast.parse(Path(runner.__file__).read_text())
    lengths = {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert max(lengths.values()) <= 60, lengths
