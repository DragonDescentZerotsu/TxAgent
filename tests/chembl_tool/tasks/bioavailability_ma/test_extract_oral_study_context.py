from __future__ import annotations

import json

import pandas as pd
import pytest

from tools.chembl_tool.tasks.bioavailability_ma.data_processing import (
    extract_oral_study_context as extractor,
)


def test_distinct_contexts_deduplicate_and_drop_null_like_values():
    values = pd.Series([None, "unknown", " Rats, plasma ", "Rats, plasma"])
    assert extractor._distinct_contexts(values) == ["Rats, plasma"]


def test_api_key_falls_back_to_distillation_repo(tmp_path, monkeypatch):
    keys_path = tmp_path / "keys.py"
    keys_path.write_text('OPENAI_API_KEY = "distillation-key"\n', encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(extractor, "DISTILLATION_KEYS_PATH", keys_path)

    assert extractor._load_api_key("OPENAI_API_KEY") == "distillation-key"


def test_response_accepts_both_extractions_and_json_null():
    content = json.dumps(
        {
            "mapping": {
                "v0000": {
                    "canonical_species_context": "human",
                    "canonical_biological_matrix": "plasma",
                },
                "v0001": {
                    "canonical_species_context": None,
                    "canonical_biological_matrix": None,
                },
            }
        }
    )
    assert extractor._validate_response(content, {"v0000", "v0001"}) == {
        "v0000": {
            "canonical_species_context": "human",
            "canonical_biological_matrix": "plasma",
        },
        "v0001": {
            "canonical_species_context": None,
            "canonical_biological_matrix": None,
        },
    }


def test_response_normalizes_literal_null_to_json_null():
    content = json.dumps(
        {
            "mapping": {
                "v0000": {
                    "canonical_species_context": "null",
                    "canonical_biological_matrix": "NULL",
                }
            }
        }
    )
    assert extractor._validate_response(content, {"v0000"}) == {
        "v0000": {
            "canonical_species_context": None,
            "canonical_biological_matrix": None,
        }
    }


@pytest.mark.parametrize(
    "result",
    [
        {
            "canonical_species_context": "healthy_human",
            "canonical_biological_matrix": "plasma",
        },
        {
            "canonical_species_context": "human",
            "canonical_biological_matrix": "blood",
        },
        {"canonical_species_context": "human"},
    ],
)
def test_response_rejects_out_of_contract_results(result):
    content = json.dumps({"mapping": {"v0000": result}})
    with pytest.raises(ValueError):
        extractor._validate_response(content, {"v0000"})
