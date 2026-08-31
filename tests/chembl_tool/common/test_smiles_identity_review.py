from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.chembl_tool.common.starling.smiles_identity_review import (
    consolidate,
    prepare,
    prepare_adjudication,
    select_review_targets,
)


def _row(
    candidate_id: str,
    *,
    decision: str = "override",
    confidence: str = "medium",
    name: str = "ethanol",
    canonical_smiles: str = "C",
    pubchem_smiles: str = "CCO",
    support_text: str = "The ethanol concentration was measured.",
    row_number: int = 1,
) -> dict:
    return {
        "id": candidate_id,
        "candidate_id": candidate_id,
        "task_id": "bbb_martins",
        "cleaned_record_id": candidate_id,
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": row_number,
        "source_record_id": str(row_number),
        "endpoint_name": "concentration",
        "measurement_text": "1.0",
        "support_text": support_text,
        "source_smiles": canonical_smiles,
        "canonical_smiles": canonical_smiles,
        "extracted_molecule_name": name,
        "extraction_evidence_span": name,
        "pubchem_cid": 702,
        "pubchem_title": name,
        "pubchem_smiles": pubchem_smiles,
        "stored_parent_inchi_key": "stored",
        "pubchem_parent_inchi_key": "reference",
        "decision": decision,
        "override_smiles": pubchem_smiles if decision == "override" else None,
        "reject_reason": None if decision == "override" else "ambiguous_context",
        "confidence": confidence,
        "rationale": "The original review rationale is long enough for the replay fixture.",
        "reviewer": "primary-reviewer",
    }


def _write_review(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_prepare_selects_expected_targets_and_keeps_context_groups_atomic(tmp_path):
    rows = [
        _row("medium-1", row_number=1),
        _row("medium-2", row_number=2),
        _row(
            "invalid-high",
            confidence="high",
            pubchem_smiles="not-a-smiles",
            support_text="CCNU was measured.",
            row_number=3,
        ),
        _row(
            "pfos-high",
            confidence="high",
            name="PfOS",
            pubchem_smiles="C(F)(F)F",
            support_text="PfOS was administered.",
            row_number=4,
        ),
        _row(
            "conflict-high",
            confidence="high",
            support_text="The named molecule was measured.",
            row_number=5,
        ),
        _row(
            "conflict-reject",
            decision="reject",
            confidence="high",
            support_text="The named molecule was measured.",
            row_number=6,
        ),
    ]
    frame = pd.DataFrame(rows)
    targets = select_review_targets(frame)
    assert set(targets["candidate_id"]) == {
        "medium-1",
        "medium-2",
        "invalid-high",
        "pfos-high",
        "conflict-high",
    }

    source = tmp_path / "v1.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "v2"
    manifest = prepare(source, output, batch_size=2)
    assert manifest["target_rows"] == 5
    packets = [
        json.loads(path.read_text(encoding="utf-8"))["items"]
        for path in sorted((output / "checker_packets").glob("*.json"))
    ]
    assert any(
        {item["target"]["candidate_id"] for item in packet}
        == {"medium-1", "medium-2"}
        for packet in packets
    )


def test_checker_agreement_promotes_medium_override(tmp_path):
    source = tmp_path / "v1.parquet"
    pd.DataFrame([_row("medium")]).to_parquet(source, index=False)
    output = tmp_path / "v2"
    prepare(source, output, batch_size=100)
    _write_review(
        output / "checker_reviews/checker.jsonl",
        {
            "candidate_id": "medium",
            "review_role": "checker",
            "reviewer": "checker-reviewer",
            "decision": "override",
            "override_smiles": "CCO",
            "rationale": "The support text unambiguously measures ethanol and the reference is valid.",
        },
    )
    assert prepare_adjudication(source, output, 100)["adjudication_rows"] == 0
    manifest = consolidate(source, output)
    result = pd.read_parquet(manifest["proposal"])
    assert result.iloc[0]["decision"] == "override"
    assert result.iloc[0]["confidence"] == "high"
    assert result.iloc[0]["review_resolution"] == "primary_checker_agree"


def test_checker_disagreement_requires_independent_adjudication(tmp_path):
    source = tmp_path / "v1.parquet"
    pd.DataFrame([_row("medium")]).to_parquet(source, index=False)
    output = tmp_path / "v2"
    prepare(source, output, batch_size=100)
    _write_review(
        output / "checker_reviews/checker.jsonl",
        {
            "candidate_id": "medium",
            "review_role": "checker",
            "reviewer": "checker-reviewer",
            "decision": "retain_original",
            "override_smiles": None,
            "rationale": "The row does not provide enough evidence to replace the stored structure.",
        },
    )
    assert prepare_adjudication(source, output, 100)["adjudication_rows"] == 1
    _write_review(
        output / "adjudicator_reviews/adjudicator.jsonl",
        {
            "candidate_id": "medium",
            "review_role": "adjudicator",
            "reviewer": "adjudicator-reviewer",
            "decision": "unresolved",
            "override_smiles": None,
            "rationale": "The available row context cannot identify a replacement with sufficient certainty.",
        },
    )
    manifest = consolidate(source, output)
    result = pd.read_parquet(manifest["proposal"])
    assert result.iloc[0]["decision"] == "reject"
    assert pd.isna(result.iloc[0]["override_smiles"])
    assert result.iloc[0]["review_resolution"] == "adjudicated"


def test_adjudicated_reject_propagates_to_identical_context(tmp_path):
    source = tmp_path / "v1.parquet"
    pd.DataFrame(
        [
            _row("medium", row_number=1),
            _row("prior-high", confidence="high", row_number=2),
        ]
    ).to_parquet(source, index=False)
    output = tmp_path / "v2"
    prepare(source, output, batch_size=100)
    _write_review(
        output / "checker_reviews/checker.jsonl",
        {
            "candidate_id": "medium",
            "review_role": "checker",
            "reviewer": "checker-reviewer",
            "decision": "retain_original",
            "override_smiles": None,
            "rationale": "The named molecule is only a comparator, so the stored structure is retained.",
        },
    )
    prepare_adjudication(source, output, 100)
    _write_review(
        output / "adjudicator_reviews/adjudicator.jsonl",
        {
            "candidate_id": "medium",
            "review_role": "adjudicator",
            "reviewer": "adjudicator-reviewer",
            "decision": "retain_original",
            "override_smiles": None,
            "rationale": "Independent review confirms that the name belongs only to a comparator.",
        },
    )

    manifest = consolidate(source, output)
    result = pd.read_parquet(manifest["proposal"]).set_index("candidate_id")
    assert set(result["decision"]) == {"reject"}
    assert result.at["prior-high", "review_resolution"] == "exact_context_propagated"
    assert result.at["prior-high", "propagated_from_candidate_id"] == "medium"
    assert manifest["exact_context_propagated_rows"] == 1
    assert manifest["net_override_delta"] == -2
