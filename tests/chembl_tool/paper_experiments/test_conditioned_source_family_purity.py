from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.benchmark_dataset import LabeledSourceRecord
from tools.chembl_tool.paper_experiments.audit_source_family_purity_gold_impact import (
    _skin_binary_label,
    _vote,
)
from tools.chembl_tool.paper_experiments.build_conditioned_source_family_purity import (
    BIO_DIRECT_GROUP,
    PURITY_VERSION,
    PuritySpec,
    _bio_reason,
    build_overlay,
)


def test_overlay_changes_only_family_and_appends_audit(tmp_path: Path):
    source = tmp_path / "source.parquet"
    rows = [
        {
            "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
            "canonical_bioavailability_evidence_scope": "relative_or_conditional",
            "canonical_endpoint_name": "bioavailability",
            "canonical_measurement_text": "F = 44.8%",
            "canonical_unit_text": "%",
            "support_text": "The overall oral bioavailability F was 44.8%.",
            "source_id": "fg",
            "canonical_record_id": "r1",
            "retrieval_eligible": True,
            "parent_smiles": "CCO",
        },
        {
            "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
            "canonical_bioavailability_evidence_scope": "mechanistic",
            "canonical_endpoint_name": "intestinal uptake",
            "canonical_measurement_text": "increased",
            "canonical_unit_text": "",
            "support_text": "CNT1 increased intestinal uptake.",
            "source_id": "fg",
            "canonical_record_id": "r2",
            "retrieval_eligible": True,
            "parent_smiles": "CCN",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    spec = PuritySpec("bioavailability_ma", source, BIO_DIRECT_GROUP, _bio_reason)

    manifest = build_overlay(spec, tmp_path / "out", batch_size=1)
    output = pq.read_table(tmp_path / "out" / "records.parquet").to_pylist()
    audit = pq.read_table(tmp_path / "out" / "moved_records.parquet").to_pylist()

    assert [row["group_id"] for row in output] == [
        BIO_DIRECT_GROUP,
        "Fg.gut_wall_efflux_intestinal_metabolism",
    ]
    assert output[0]["source_family_purity_version"] == PURITY_VERSION
    assert output[1]["source_family_purity_version"] is None
    assert output[0]["support_text"] == rows[0]["support_text"]
    assert output[1]["support_text"] == rows[1]["support_text"]
    assert len(audit) == 1
    assert audit[0]["source_record_id"] == "r1"
    assert manifest["n_moved_rows"] == 1
    assert manifest["gold_labels_modified"] is False


def test_gold_sensitivity_vote_uses_frozen_record_agreement_contract():
    rows = [
        LabeledSourceRecord("CCO", label, "source")
        for label in [1, 1, 1, 0]
    ]
    assert _vote(rows)["label"] == 1
    assert _vote([*rows, LabeledSourceRecord("CCO", 0, "candidate")])["label"] is None
    assert _skin_binary_label("positive") == 1
    assert _skin_binary_label("negative") == 0
    assert _skin_binary_label("inconclusive") is None
