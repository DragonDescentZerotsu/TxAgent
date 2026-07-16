import json
import sys
from types import SimpleNamespace

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.starling import oral_bioavailability as oral_cleaning
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_evidence_library import (
    GROUP_ID,
    build_starling_evidence_rows,
    build_starling_evidence_rows_from_pinned_hf,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_evidence_library import build_neighbor_index
from tools.chembl_tool.tasks.bioavailability_ma.retrieve_neighbors import retrieve_neighbors
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import _clean_evidence_row


def test_build_starling_evidence_rows_aggregates_molecule_records(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(_record(1, "CCCO", 10.0, "Rat", "absolute")),
                json.dumps(_record(2, "CCCO", 50.0, "Human", "unspecified")),
                json.dumps(_record(3, "CCCCO", 150.0, "Human", "absolute")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, stats = build_starling_evidence_rows(source)

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
            "source_index": 1,
            "molecule_name": "molecule_1",
            "oral_bioavailability_value_percent": 10.0,
            "parse_modifier": "",
            "condition_text": "species_or_population: Rat",
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
            "source_index": 2,
            "molecule_name": "molecule_2",
            "oral_bioavailability_value_percent": 50.0,
            "parse_modifier": "",
            "condition_text": "species_or_population: Human",
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
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(_record(index, "CCCO", float(value), f"Species {index}", "absolute"))
            for index, value in enumerate(range(0, 100, 10), start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _ = build_starling_evidence_rows(source)

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
    source = tmp_path / "records.jsonl"
    source.write_text("", encoding="utf-8")
    dropped = tmp_path / "dropped.jsonl"
    dropped.write_text(
        json.dumps(
            {
                "source_index": 7,
                "drop_reason": "unparseable_or_non_numeric_value",
                "raw_row": {
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
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows, stats = build_starling_evidence_rows(source, dropped_jsonl=dropped)

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
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(_record(1, "CCO", 80.0, "Human", "absolute")),
                json.dumps(_record(2, "CCCO", 60.0, "Human", "absolute")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows, _ = build_starling_evidence_rows(source)
    index = build_neighbor_index(rows)
    index["source"] = {"dataset": "starling-labs/Oral_Bioavailability"}

    result = retrieve_neighbors("CCO", index, top_k_per_group=3, min_similarity=0.0)

    neighbors = result["groups"][0]["neighbors"]
    assert len(neighbors) == 1
    assert neighbors[0]["canonical_smiles"] == "CCCO"
    assert result["evidence_source"]["dataset"] == "starling-labs/Oral_Bioavailability"


def test_pinned_hf_loader_uses_frozen_revision(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    assert oral_cleaning.load_pinned_oral_bioavailability_dataset() == []
    assert calls == [
        (("starling-labs/Oral_Bioavailability",), {
            "revision": "01bbe3ee9cdd3dc081c39973529c9da0c814d465", "split": "train"
        })
    ]


def test_pinned_rows_filter_report_type_bounds_invalid_smiles_and_content_mode(monkeypatch):
    dataset = [
        _raw_hf("CCO", "0.5", "absolute"),
        _raw_hf("OCC", "20%", "systemic_availability"),
        _raw_hf("CCN", "very low", "unspecified"),
        _raw_hf("CCC", "40%", "relative"),
        _raw_hf("not smiles", "10%", "absolute"),
        _raw_hf("CCCC", "101%", "absolute"),
    ]
    monkeypatch.setattr(oral_cleaning, "load_pinned_oral_bioavailability_dataset", lambda: dataset)

    full_rows, full_stats = build_starling_evidence_rows_from_pinned_hf(evidence_content="full")
    numeric_rows, numeric_stats = build_starling_evidence_rows_from_pinned_hf(evidence_content="numeric_only")

    assert len(full_rows) == 2
    ethanol = next(row for row in full_rows if row["canonical_smiles"] == "CCO")
    assert ethanol["source_numeric_record_count"] == 2
    assert ethanol["standard_value"] == 35.0
    qualitative = next(row for row in full_rows if row["canonical_smiles"] == "CCN")
    assert qualitative["source_qualitative_record_count"] == 1
    assert len(numeric_rows) == 1
    assert numeric_rows[0]["canonical_smiles"] == "CCO"
    assert full_stats["revision"] == "01bbe3ee9cdd3dc081c39973529c9da0c814d465"
    assert numeric_stats["n_qualitative_rows_kept"] == 0


def test_direct_family_combines_sources_without_duplicate_neighbor_slots(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(_record(1, "CCCO", 40.0, "Human", "absolute")) + "\n", encoding="utf-8")
    direct_rows, _ = build_starling_evidence_rows(source)
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


def _raw_hf(smiles: str, value: str, report_type: str) -> dict:
    return {
        "molecule_name": "test molecule", "smiles": smiles, "oral_bioavailability_value": value,
        "bioavailability_report_type": report_type, "species_or_population": "human", "support_text": "source passage",
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
