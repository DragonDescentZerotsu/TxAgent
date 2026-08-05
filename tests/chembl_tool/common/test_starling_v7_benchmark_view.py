from __future__ import annotations

import json

import pandas as pd

from tools.chembl_tool.common.starling.v7_benchmark_view import (
    DIRECT_NUMERIC_VIEW,
    FULL_VIEW,
    build_v7_benchmark_view,
)
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import load_index
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import POLICY


def _record(record_id: str, smiles: str, group_id: str, value):
    row = {
        "canonical_record_id": record_id,
        "retrieval_eligible": True,
        "canonical_smiles": smiles,
        "group_id": group_id,
        "finite_scalar_value": value,
        "confidence": 0.9,
        "endpoint_name": "oral_bioavailability",
        "canonical_endpoint_name": "oral_bioavailability",
        "source_id": "hf_bioavailability",
        "source_name": "HF",
        "source_row_number": int(record_id[-1]),
        "bioavailability_report_type": (
            "absolute" if "direct" in group_id and "nondirect" not in group_id else "relative_comparison"
        ),
        "canonical_bioavailability_evidence_scope": (
            "direct" if "direct" in group_id and "nondirect" not in group_id else "nondirect"
        ),
        "oral_bioavailability_value": str(value or "relative"),
        "support_text": "example",
        "measurement_text": str(value or "relative"),
        "unit_text": "%" if value is not None else None,
        "smiles": smiles,
    }
    for field in (
        "comparator",
        "dose",
        "extra_details",
        "molecule_name",
        "oral_exposure_mode",
        "pmid",
        "qualifying_conditions",
        "species_or_population",
    ):
        row[field] = None
    return row


def test_v7_paper_view_filters_heldout_parents_across_every_group(tmp_path):
    normalized_root = tmp_path / "v7"
    records_dir = normalized_root / "03_records"
    records_dir.mkdir(parents=True)
    records = [
        _record("record-1", "CCO", "Observed.nondirect_oral_bioavailability", None),
        _record("record-2", "CCN", "Observed.direct_oral_bioavailability", 42.0),
        _record("record-3", "CCC", "Observed.nondirect_oral_bioavailability", None),
    ]
    pd.DataFrame(records).to_parquet(records_dir / "records.parquet", index=False)
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"drug": "CCO", "Y": 1}) + "\n", encoding="utf-8")

    full_dir = tmp_path / "full"
    full = build_v7_benchmark_view(
        policy=POLICY,
        normalized_root=normalized_root,
        heldout_labels_jsonl=heldout,
        out_dir=full_dir,
        benchmark_split="random",
        view=FULL_VIEW,
    )

    assert full["heldout_filter"]["zero_parent_overlap"] is True
    assert full["heldout_filter"]["n_excluded_heldout_records"] == 1
    assert set(pd.read_parquet(full_dir / "06_records/records.parquet")["canonical_smiles"]) == {
        "CCN",
        "CCC",
    }
    assert set(load_index(full_dir / "08_neighbor_index")["group_to_molecule_indices"]) == {
        "Observed.direct_oral_bioavailability",
        "Observed.nondirect_oral_bioavailability",
    }

    direct_dir = tmp_path / "direct"
    direct = build_v7_benchmark_view(
        policy=POLICY,
        normalized_root=normalized_root,
        heldout_labels_jsonl=heldout,
        out_dir=direct_dir,
        benchmark_split="random",
        view=DIRECT_NUMERIC_VIEW,
    )
    direct_rows = pd.read_parquet(direct_dir / "06_records/records.parquet")
    assert direct["records"] == 1
    assert direct_rows["group_id"].tolist() == ["Observed.direct_oral_bioavailability"]
    assert direct_rows["finite_scalar_value"].tolist() == [42.0]
