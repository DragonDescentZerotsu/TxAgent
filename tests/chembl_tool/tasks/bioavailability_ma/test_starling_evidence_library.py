import json

from tools.chembl_tool.tasks.bioavailability_ma.build_starling_evidence_library import (
    GROUP_ID,
    build_starling_evidence_rows,
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
    assert "pmid" not in clean_row["source_record_examples"][0]
    assert clean_row["source_record_examples"][0]["oral_bioavailability_value_percent"] == 10.0


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
    assert rows[0]["evidence_direction"] == "argues_against_high_bioavailability"
    clean_row = _clean_evidence_row(rows[0])
    assert clean_row["source_qualitative_examples"][0]["oral_bioavailability_value_text"] == "very low"
    assert "pmid" not in clean_row["source_qualitative_examples"][0]
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
