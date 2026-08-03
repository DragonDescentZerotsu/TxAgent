import json

import pandas as pd
import pytest

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING
from tools.chembl_tool.tasks.bioavailability_ma import build_starling_evidence_library as direct_builder
from tools.chembl_tool.tasks.bioavailability_ma import build_starling_factor_evidence_library as factor_builder
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_evidence_library import (
    GROUP_ID,
    build_starling_evidence_rows,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_evidence_library import build_neighbor_index
from tools.chembl_tool.tasks.bioavailability_ma.retrieve_neighbors import retrieve_neighbors
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import _clean_evidence_row
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_sources import (
    DIRECT_HF_SOURCE_COLUMNS,
)


def test_build_starling_evidence_rows_aggregates_molecule_records(tmp_path):
    source = tmp_path / "records.parquet"
    _write_source_parquet(
        source,
        [
            _record(1, "CCCO", 10.0, "Rat", "absolute"),
            _record(2, "CCCO", 50.0, "Human", "unspecified"),
            _record(3, "CCCCO", 150.0, "Human", "absolute"),
        ],
    )

    rows, stats = build_starling_evidence_rows(source, max_source_rows=100)

    assert len(rows) == 1
    assert stats["n_source_rows"] == 3
    assert stats["n_source_rows_kept"] == 2
    assert rows[0]["group_id"] == GROUP_ID
    assert rows[0]["standard_value"] == 30.0
    assert rows[0]["source_record_count"] == 2
    assert rows[0]["source_value_min_percent"] == 10.0
    assert rows[0]["source_value_max_percent"] == 50.0
    assert rows[0]["source_record_examples"] == [
        {
            "source_index": 0,
            "molecule_name": "molecule_1",
            "oral_bioavailability_value_percent": 10.0,
            "parse_modifier": "",
            "condition_text": "species_or_population: Rat\ndose: not specified\noral_exposure_mode: not specified\nqualifying_conditions: not specified\ncomparator: not specified\nextra_details: not specified",
            "species_or_population": "Rat",
            "dose": "",
            "oral_exposure_mode": "",
            "qualifying_conditions": "",
            "comparator": "",
            "extra_details": "",
            "pmid": "1",
            "support_text": "Reported oral bioavailability was 10.0%.",
            "bioavailability_report_type": "absolute",
        },
        {
            "source_index": 1,
            "molecule_name": "molecule_2",
            "oral_bioavailability_value_percent": 50.0,
            "parse_modifier": "",
            "condition_text": "species_or_population: Human\ndose: not specified\noral_exposure_mode: not specified\nqualifying_conditions: not specified\ncomparator: not specified\nextra_details: not specified",
            "species_or_population": "Human",
            "dose": "",
            "oral_exposure_mode": "",
            "qualifying_conditions": "",
            "comparator": "",
            "extra_details": "",
            "pmid": "2",
            "support_text": "Reported oral bioavailability was 50.0%.",
            "bioavailability_report_type": "unspecified",
        },
    ]
    assert rows[0]["assay_description"].startswith(
        "oral_bioavailability_value_percent: 10\nspecies_or_population: Rat"
    )
    assert "oral_bioavailability_value_percent: 50" in rows[0]["assay_description"]
    clean_row = _clean_evidence_row(rows[0])
    assert "source_pmids" not in clean_row
    assert "pmid" not in clean_row["examples"][0]
    assert clean_row["examples"][0]["oral_bioavailability_value_percent"] == 10.0
    assert clean_row["annotations"]["evidence_role"] == "direct_outcome"


def test_numeric_examples_cover_value_distribution_with_at_most_six_records(tmp_path):
    source = tmp_path / "records.parquet"
    _write_source_parquet(
        source,
        [
            _record(index, "CCCO", float(value), f"Species {index}", "absolute")
            for index, value in enumerate(range(0, 100, 10), start=1)
        ],
    )

    rows, _ = build_starling_evidence_rows(source, max_source_rows=100)

    examples = rows[0]["source_record_examples"]
    assert len(examples) == 6
    assert [item["oral_bioavailability_value_percent"] for item in examples] == [
        0.0,
        20.0,
        40.0,
        50.0,
        70.0,
        90.0,
    ]


def test_qualitative_only_molecule_is_added_without_fake_numeric_value(tmp_path):
    source = tmp_path / "records.parquet"
    _write_source_parquet(
        source,
        [
            {
                "source_index": 7,
                    "molecule_name": "qualitative molecule",
                    "smiles": "CCCO",
                    "oral_bioavailability_value": "very low",
                    "bioavailability_report_type": "unspecified",
                    "species_or_population": "rats",
                    "dose": "10 mg/kg",
                    "oral_exposure_mode": "oral",
                    "qualifying_conditions": None,
                    "comparator": None,
                    "extra_details": "extensive first-pass metabolism",
                    "pmid": "should-not-reach-the-llm",
                    "support_text": "The compound exhibited very low oral bioavailability in rats.",
            }
        ],
    )

    rows, stats = build_starling_evidence_rows(source, max_source_rows=100)

    assert len(rows) == 1
    assert stats["n_qualitative_rows_kept"] == 1
    assert stats["n_molecules_with_qualitative_only_evidence"] == 1
    assert rows[0]["standard_value"] == ""
    assert rows[0]["source_numeric_record_count"] == 0
    assert rows[0]["source_qualitative_record_count"] == 1
    clean_row = _clean_evidence_row(rows[0])
    assert clean_row["examples"][0]["oral_bioavailability_value_text"] == "very low"
    assert "qualitative_only_no_numeric_measurement" in clean_row["annotations"]["uncertainty"]
    assert "pmid" not in clean_row["examples"][0]
    assert "source_pmids" not in clean_row


def test_starling_index_retrieval_excludes_exact_query(tmp_path):
    source = tmp_path / "records.parquet"
    _write_source_parquet(
        source,
        [
            _record(1, "CCO", 80.0, "Human", "absolute"),
            _record(2, "CCCO", 60.0, "Human", "absolute"),
        ],
    )
    rows, _ = build_starling_evidence_rows(source, max_source_rows=100)
    index = build_neighbor_index(rows)
    index["source"] = {"dataset": "starling-labs/Oral_Bioavailability"}

    result = retrieve_neighbors("CCO", index, top_k_per_group=3, min_similarity=0.0)

    neighbors = result["groups"][0]["neighbors"]
    assert len(neighbors) == 1
    assert neighbors[0]["canonical_smiles"] == "CCCO"
    assert result["evidence_source"]["dataset"] == "starling-labs/Oral_Bioavailability"


def test_evidence_builder_clis_reject_removed_pinned_hf_modes():
    with pytest.raises(SystemExit):
        direct_builder._parse_args(["--source-mode", "pinned-hf"])
    with pytest.raises(SystemExit):
        factor_builder._parse_args(["--direct-source-mode", "pinned-hf"])


def test_complete_parquet_qualitative_handling_keeps_not_allowed_report_type(tmp_path):
    source = tmp_path / "records.parquet"
    _write_source_parquet(source, [{
        "source_index": 11,
            "molecule_name": "relative report",
            "smiles": "CCN",
            "oral_bioavailability_value": "higher than reference",
            "bioavailability_report_type": "relative",
            "support_text": "Relative oral exposure was higher than the reference.",
    }])

    rows, stats = build_starling_evidence_rows(source, max_source_rows=100)

    assert len(rows) == 1
    assert stats["n_qualitative_rows_kept"] == 1
    assert rows[0]["source_report_types"] == ["relative"]
    assert rows[0]["source_qualitative_record_count"] == 1


def test_factor_builder_include_direct_hf_reads_complete_parquet(monkeypatch, tmp_path):
    records = tmp_path / "records.parquet"
    records.write_bytes(b"fixture")
    calls = []

    def fake_direct(source_parquet, **kwargs):
        calls.append((source_parquet, kwargs["include_qualitative"]))
        return [], {
            "n_source_rows": 163815,
            "n_source_rows_kept": 80808,
            "n_dropped_rows_scanned": 83007,
        }

    monkeypatch.setattr(factor_builder, "build_direct_f_rows", fake_direct)
    monkeypatch.setattr(factor_builder, "build_starling_parquet_evidence_rows", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(factor_builder, "build_neighbor_index", lambda *args, **kwargs: {
        "molecules": [], "group_to_molecule_indices": {}
    })

    assert factor_builder.main([
        "--out-dir", str(tmp_path / "index"),
        "--direct-source-parquet", str(records),
        "--expected-direct-clean-numeric-rows", "80808",
    ]) == 0
    assert calls == [(records, True)]
    meta = json.loads((tmp_path / "index" / factor_builder.META_FILENAME).read_text(encoding="utf-8"))
    assert meta["include_direct_hf"] is True
    assert meta["direct_hf_provenance"]["source_mode"] == "complete_parquet"
    assert meta["direct_source_stats"]["n_source_rows_kept"] == 80808
    assert meta["prepared_hf_row_counts"] == {
        "raw_rows": 163815,
        "clean_numeric_rows": 80808,
        "dropped_rows": 83007,
    }
    assert len(meta["underlying_sources"]) == 5


def test_prepared_hf_count_preflight_rejects_mismatched_artifacts():
    with pytest.raises(ValueError, match="Prepared HF artifact preflight failed"):
        factor_builder._validate_prepared_direct_hf_counts(
            {"n_source_rows": 163814, "n_source_rows_kept": 80808, "n_dropped_rows_scanned": 83006},
            expected_raw_rows=163815,
            expected_clean_numeric_rows=82496,
        )


def test_direct_family_combines_sources_without_duplicate_neighbor_slots(tmp_path):
    source = tmp_path / "records.parquet"
    _write_source_parquet(source, [_record(1, "CCCO", 40.0, "Human", "absolute")])
    direct_rows, _ = build_starling_evidence_rows(source, max_source_rows=100)
    auc_row = dict(direct_rows[0])
    auc_row["evidence_source"] = "starling-labs/bioavailability_ma/Oral_AUC-Cmax-Exposure"
    auc_row["activity_comment"] = "Oral AUC-Cmax direct bioavailability evidence"
    index = build_neighbor_index([*direct_rows, auc_row])
    index["source"] = {"dataset": "combined Starling direct sources"}

    result = retrieve_experiment_view("CCO", index, mode="direct", config=STARLING, top_k_per_group=3, min_similarity=0.0)
    neighbors = result["groups"][0]["neighbors"]
    assert len(neighbors) == 1
    assert neighbors[0]["n_evidence_rows"] == 2
    assert {row["evidence_source"] for row in neighbors[0]["evidence_rows"]} == {
        "starling-labs/Oral_Bioavailability", "starling-labs/bioavailability_ma/Oral_AUC-Cmax-Exposure"
    }


def _record(index: int, smiles: str, value: float, species: str, report_type: str) -> dict:
    return {
        "source_index": index,
        "molecule_id": f"row_{index}",
        "molecule_name": f"molecule_{index}",
        "smiles": smiles,
        "oral_bioavailability_value_percent": value,
        "condition_text": f"species_or_population: {species}",
        "parse_modifier": "",
        "metadata": {
            "pmid": str(index),
            "support_text": f"Reported oral bioavailability was {value}%.",
            "bioavailability_report_type": report_type,
            "species_or_population": species,
        },
    }


def _write_source_parquet(path, rows):
    source_rows = []
    for position, item in enumerate(rows):
        raw = {column: None for column in DIRECT_HF_SOURCE_COLUMNS}
        metadata = item.get("metadata") or {}
        raw.update(metadata)
        raw.update(
            {
                key: value
                for key, value in item.items()
                if key in DIRECT_HF_SOURCE_COLUMNS
            }
        )
        raw["source_index"] = position
        if "oral_bioavailability_value_percent" in item:
            raw["oral_bioavailability_value"] = str(
                item["oral_bioavailability_value_percent"]
            )
        source_rows.append(raw)
    pd.DataFrame(source_rows, columns=DIRECT_HF_SOURCE_COLUMNS).to_parquet(
        path, index=False
    )
