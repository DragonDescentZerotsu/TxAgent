from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_candidate_blind_audit as audit,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    candidate = tmp_path / "measurement_candidates.parquet"
    candidate.write_bytes(b"frozen candidate bytes")
    implementations = {}
    inputs = {}
    for name in audit.IMPLEMENTATION_INPUTS:
        path = tmp_path / f"{name}.py"
        path.write_text(name, encoding="utf-8")
        implementations[name] = inputs[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    manifest = {
        "candidate_path": str(candidate.resolve()),
        "candidate_sha256": file_sha256(candidate),
        "implementation": implementations,
        "validations": {name: True for name in audit.CANDIDATE_VALIDATIONS},
    }
    manifest_path = candidate.with_suffix(".manifest.json")
    _write(manifest_path, manifest)
    inputs |= {
        name: {"path": str(tmp_path / name), "sha256": digest}
        for name, digest in audit.FROZEN_AUDIT_HASHES.items()
    }
    inputs["candidate_artifact"] = {
        "path": str(candidate.resolve()),
        "sha256": file_sha256(candidate),
    }
    inputs["candidate_manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
    }
    return candidate, manifest_path, inputs


def _passing_report(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    candidate, _, inputs = _artifacts(tmp_path)
    inputs |= {
        name: {"path": str(tmp_path / name), "sha256": "a" * 64}
        for name in audit.REPORT_INPUTS - set(inputs)
    }
    rate = {"correct": 19, "total": 20, "rate": 0.95}
    report = {
        "score_version": audit.SCORE_VERSION,
        "audit_version": audit.AUDIT_VERSION,
        "inputs": inputs,
        "rows": 90,
        "label_status_counts": {"ok": 60, "unsure": 30},
        "label_reason_counts": {"absolute": 60, "missing_unit": 30},
        "exact_absolute": rate,
        "exact_absolute_by_source": {
            source: {"correct": 9, "total": 10, "rate": 0.9}
            for source in audit.BLIND_SOURCES
        },
        "sample_candidate_evidence_grounding_failures": 0,
        "candidate_inventory_validations": {
            name: True for name in audit.CANDIDATE_VALIDATIONS
        },
        "gates": {name: True for name in audit.GATES},
        "decision": "PASS",
    }
    receipt = tmp_path / "candidate_blind_audit_v3.score_report.json"
    _write(receipt, report)
    return receipt, candidate, report


def test_accepts_exact_pass_and_returns_compiler_reference(tmp_path: Path) -> None:
    receipt, candidate, _ = _passing_report(tmp_path)
    reference = audit.validate_blind_pass_receipt(receipt, candidate)
    assert reference["sha256"] == file_sha256(receipt)
    assert reference["candidate_sha256"] == file_sha256(candidate)
    assert reference["decision"] == "PASS"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update(decision="FAIL"), "PASS decision"),
        (
            lambda report: report["gates"].update(
                candidate_inventory_no_truncation=False
            ),
            "failed claim",
        ),
        (
            lambda report: report["exact_absolute"].update(correct=18, rate=0.9),
            "below its acceptance threshold",
        ),
        (
            lambda report: report["exact_absolute_by_source"].pop(
                "premutagenic_damage"
            ),
            "source coverage",
        ),
        (
            lambda report: report.update(
                sample_candidate_evidence_grounding_failures=1
            ),
            "grounding failures",
        ),
        (
            lambda report: report["inputs"]["sample"].update(path="relative"),
            "path is not absolute",
        ),
    ],
)
def test_rejects_failed_or_incomplete_score_claims(
    tmp_path: Path, mutation, message: str
) -> None:
    receipt, candidate, report = _passing_report(tmp_path)
    mutation(report)
    _write(receipt, report)
    with pytest.raises(ValueError, match=message):
        audit.validate_blind_pass_receipt(receipt, candidate)


@pytest.mark.parametrize("target", ["candidate", "manifest", "generator"])
def test_rejects_candidate_or_implementation_drift(tmp_path: Path, target: str) -> None:
    receipt, candidate, report = _passing_report(tmp_path)
    if target == "candidate":
        candidate.write_bytes(b"changed")
    elif target == "manifest":
        candidate.with_suffix(".manifest.json").write_text("{}\n")
    else:
        path = Path(report["inputs"]["generator"]["path"])
        path.write_text("changed", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        audit.validate_blind_pass_receipt(receipt, candidate)


def test_canonical_wrapper_uses_only_its_frozen_paths(tmp_path, monkeypatch) -> None:
    receipt, candidate, _ = _passing_report(tmp_path)
    monkeypatch.setattr(audit, "BLIND_PASS_RECEIPT_PATH", receipt)
    monkeypatch.setattr(audit, "CANDIDATE_PATH", candidate)
    monkeypatch.setattr(
        audit,
        "FROZEN_CANDIDATE_HASHES",
        {
            "candidate_sha256": file_sha256(candidate),
            "candidate_manifest_sha256": file_sha256(
                candidate.with_suffix(".manifest.json")
            ),
        },
    )
    assert audit.validate_canonical_blind_pass()["decision"] == "PASS"

    monkeypatch.setattr(audit, "BLIND_PASS_RECEIPT_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        audit.validate_canonical_blind_pass()
