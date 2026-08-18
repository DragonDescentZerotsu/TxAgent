from __future__ import annotations

import json

import pandas as pd

from tools.chembl_tool.common.starling.v7_benchmark_view import (
    ALL_SCAFFOLD_FILTER,
    DIRECT_SOURCE_ONLY_FILTER,
    DIRECT_NUMERIC_VIEW,
    FULL_VIEW,
    build_v7_benchmark_view,
)
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import (
    load_index,
    retrieve_neighbors,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_downstream_artifacts import (
    get_spec as get_bioavailability_downstream_spec,
)
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
    assert (full_dir / "06_records/manifest.json").is_file()
    assert (full_dir / "07_molecule_evidence/manifest.json").is_file()
    assert (full_dir / "09_audits/heldout_overlap.json").is_file()
    assert full["stage_manifests"] == {
        "06_records": "06_records/manifest.json",
        "07_molecule_evidence": "07_molecule_evidence/manifest.json",
        "08_neighbor_index": "08_neighbor_index/manifest.json",
        "09_audits": "09_audits/heldout_overlap.json",
    }
    index_manifest = json.loads(
        (full_dir / "08_neighbor_index/manifest.json").read_text(encoding="utf-8")
    )
    assert index_manifest["records_manifest"] == "../06_records/manifest.json"
    assert index_manifest["evidence_manifest"] == (
        "../07_molecule_evidence/manifest.json"
    )

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


def test_v7_direct_source_view_retains_heldout_mechanism_records(tmp_path):
    normalized_root = tmp_path / "v7"
    records_dir = normalized_root / "03_records"
    records_dir.mkdir(parents=True)
    records = [
        _record("record-1", "CCO", "Observed.direct_oral_bioavailability", 40.0),
        _record("record-2", "CCO", "Observed.nondirect_oral_bioavailability", None),
        _record("record-3", "CCN", "Observed.direct_oral_bioavailability", 42.0),
    ]
    pd.DataFrame(records).to_parquet(records_dir / "records.parquet", index=False)
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"drug": "CCO", "Y": 1}) + "\n", encoding="utf-8")

    output = tmp_path / "direct-source-only"
    manifest = build_v7_benchmark_view(
        policy=POLICY,
        normalized_root=normalized_root,
        heldout_labels_jsonl=heldout,
        out_dir=output,
        benchmark_split="scaffold",
        view=FULL_VIEW,
        heldout_filter_mode=DIRECT_SOURCE_ONLY_FILTER,
        downstream_spec=get_bioavailability_downstream_spec(),
    )

    rows = pd.read_parquet(output / "06_records/records.parquet")
    assert rows["canonical_record_id"].tolist() == ["record-2", "record-3"]
    heldout_filter = manifest["heldout_filter"]
    assert heldout_filter["filter_source_id"] == "hf_bioavailability"
    assert heldout_filter["filter_scope"] == {
        "field": "canonical_bioavailability_evidence_scope",
        "value": "direct",
    }
    assert heldout_filter["n_excluded_heldout_records"] == 1
    assert heldout_filter["n_retained_heldout_nonfilter_records"] == 1
    assert heldout_filter["n_retained_heldout_nonfilter_parent_identities"] == 1
    assert heldout_filter["zero_filter_scope_parent_overlap"] is True
    assert heldout_filter["zero_parent_overlap"] is False
    audit = json.loads(
        (output / "09_audits/heldout_overlap.json").read_text(encoding="utf-8")
    )
    assert all(audit["validations"].values())

    retrieval = retrieve_neighbors(
        "CCO",
        load_index(output / "08_neighbor_index"),
        top_k_per_group=3,
        min_similarity=0.0,
        neighbor_identity_policy="parent_disjoint",
    )
    assert all(
        neighbor["canonical_smiles"] != "CCO"
        for group in retrieval["groups"]
        for neighbor in group["neighbors"]
    )


def test_v7_all_scaffold_filter_applies_to_every_source(tmp_path):
    normalized_root = tmp_path / "v7"
    records_dir = normalized_root / "03_records"
    records_dir.mkdir(parents=True)
    hf = _record("record-1", "Nc1ccccc1", "Observed.direct_oral_bioavailability", 40.0)
    oral = _record("record-2", "Cc1ccccc1", "Observed.oral_auc_cmax_exposure", None)
    oral.update(source_id="oral_exposure", source_name="oral", endpoint_name="AUC")
    fa = _record("record-3", "CCc1ccccc1", "Fa.absorption_solubility_permeability", None)
    fa.update(source_id="fa", source_name="fa", endpoint_name="solubility")
    kept = _record("record-4", "C1CCCCC1", "Observed.direct_oral_bioavailability", 42.0)
    pd.DataFrame([hf, oral, fa, kept]).to_parquet(
        records_dir / "records.parquet", index=False
    )
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        json.dumps({"drug": "Oc1ccccc1", "Y": 1}) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "all-scaffolds"
    manifest = build_v7_benchmark_view(
        policy=POLICY,
        normalized_root=normalized_root,
        heldout_labels_jsonl=heldout,
        out_dir=output,
        benchmark_split="scaffold",
        heldout_filter_mode=ALL_SCAFFOLD_FILTER,
    )

    rows = pd.read_parquet(output / "06_records/records.parquet")
    assert rows["canonical_record_id"].tolist() == ["record-4"]
    heldout_filter = manifest["heldout_filter"]
    assert heldout_filter["n_excluded_scaffold_records"] == 3
    assert heldout_filter["zero_heldout_scaffold_overlap"] is True
    assert heldout_filter["source_counts"]["hf_bioavailability"]["excluded_heldout"] == 1
    assert heldout_filter["source_counts"]["oral_exposure"]["excluded_heldout"] == 1
    assert heldout_filter["source_counts"]["fa"]["excluded_heldout"] == 1
