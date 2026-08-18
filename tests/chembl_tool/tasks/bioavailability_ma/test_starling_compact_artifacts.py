from __future__ import annotations

import json

import pandas as pd

from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import retrieve_neighbors
from tools.chembl_tool.tasks.bioavailability_ma.starling_compact_artifacts import (
    BANNED_PERSISTED_FIELDS,
    build_relational_evidence_catalog,
    compact_persisted_records,
    load_compact_neighbor_index,
    write_compact_neighbor_index,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMNS,
)


def _record(record_id: str, value: str) -> dict:
    row = {column: None for column in SOURCE_COLUMNS["hf_bioavailability"]}
    row.update(
        {
            "normalized_record_id": record_id,
            "source_id": "hf_bioavailability",
            "source_name": "HF bioavailability",
            "source_row_number": int(record_id[-1]),
            "source_record_id": f"source-{record_id}",
            "canonical_smiles": "CCO",
            "group_id": "Tier 1.direct_oral_bioavailability",
            "assay_tier": "Tier 1",
            "endpoint_group": "direct_oral_bioavailability",
            "evidence_role": "direct_outcome",
            "target_pref_name": "oral bioavailability",
            "endpoint_name": "oral_bioavailability",
            "bioavailability_report_type": "absolute",
            "canonical_bioavailability_evidence_scope": "direct",
            "canonical_endpoint": "oral_bioavailability",
            "finite_scalar_value": float(value),
            "confidence": 0.9,
            "retrieval_eligible": True,
            "oral_bioavailability_value": value,
            "support_text": f"paid evidence {record_id}",
            "source_payload_json": json.dumps({"duplicate": True}),
            "llm_source_fields_json": json.dumps({"duplicate": True}),
        }
    )
    return row


def test_compact_catalog_and_index_hydrate_referenced_records(tmp_path):
    records = compact_persisted_records([_record("record-1", "40"), _record("record-2", "50")])
    assert not (set(records[0]) & BANNED_PERSISTED_FIELDS)

    root = tmp_path / "starling_normalized_v6"
    records_dir = root / "03_records"
    catalog_dir = root / "04_evidence_catalog"
    index_dir = root / "05_neighbor_index"
    records_dir.mkdir(parents=True)
    catalog_dir.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(records_dir / "records.parquet", index=False)
    families, bridge = build_relational_evidence_catalog(records)
    pd.DataFrame(families).to_parquet(
        catalog_dir / "molecule_families.parquet", index=False
    )
    pd.DataFrame(bridge).to_parquet(
        catalog_dir / "molecule_family_records.parquet", index=False
    )
    write_compact_neighbor_index(families=families, output_dir=index_dir)

    index = load_compact_neighbor_index(index_dir)
    result = retrieve_neighbors("CCN", index, min_similarity=0.0)
    evidence = result["groups"][0]["neighbors"][0]["evidence_rows"][0]
    assert evidence["source_record_count"] == 2
    assert "normalized_records" not in evidence
    examples = evidence["minimal_evidence"]["examples"]
    assert examples[0]["source_fields"]["support_text"].startswith("paid evidence")
    assert examples[0]["source_contract"]["source_or_simply_cleaned"]["support_text"] is True
    assert evidence["_representative_record_ids"] == ["record-1", "record-2"]
    assert "_representative_record_ids" not in str(evidence_for_llm(evidence))


def test_compact_index_contains_no_pickle(tmp_path):
    records = compact_persisted_records([_record("record-1", "40")])
    families, _ = build_relational_evidence_catalog(records)
    write_compact_neighbor_index(families=families, output_dir=tmp_path)
    assert not list(tmp_path.glob("*.pkl"))
    assert {path.name for path in tmp_path.iterdir()} == {
        "molecules.parquet",
        "group_membership.parquet",
        "fingerprints.npz",
        "manifest.json",
    }
