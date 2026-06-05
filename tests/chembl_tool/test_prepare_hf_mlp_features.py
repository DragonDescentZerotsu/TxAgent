from __future__ import annotations

import json

import numpy as np

from tools.chembl_tool.activity_transfer_benchmark.prepare_hf_mlp_features import (
    CleanRow,
    build_molecule_feature_table,
    collect_endpoints,
    completion_to_label,
    dummy_embeddings,
    parse_prompt,
    write_jsonl,
    write_row_indices,
)


PROMPT = """You are given two molecules and an assay endpoint. Decide whether the endpoint behavior should transfer between the molecules.

## Endpoint
- Measurement type: IC50
- Assay type: F (functional)
- Assay format: cell-based format
- Assay description: Tested in vitro for antineoplastic activity against MEL-28 (human melanoma) tumor cell line
- Assay organism: Homo sapiens
- Target name: SK-MEL-28

## Molecule A
- SMILES: CCO

## Molecule B
- SMILES: CCN

Should this assay endpoint transfer between Molecule A and Molecule B?
(A) transfer
(B) not transfer

Answer:"""


def test_parse_prompt_extracts_endpoint_and_smiles():
    parsed = parse_prompt(PROMPT)

    assert "Measurement type: IC50" in parsed.endpoint_text
    assert "Target name: SK-MEL-28" in parsed.endpoint_text
    assert "## Molecule A" not in parsed.endpoint_text
    assert parsed.molecule_a_smiles == "CCO"
    assert parsed.molecule_b_smiles == "CCN"
    assert completion_to_label("A") == "similar"
    assert completion_to_label("B") == "different"


def test_rdkit_feature_table_deduplicates_canonical_smiles():
    rows = [
        CleanRow(
            split="train",
            source_index=0,
            label="similar",
            endpoint_key="endpoint-1",
            endpoint_text="endpoint text",
            endpoint_text_hash="hash",
            molecule_a_smiles="CCO",
            molecule_b_smiles="OCC",
            metadata={"endpoint_id": "endpoint-1"},
        )
    ]

    table, descriptor_names = build_molecule_feature_table({"train": rows}, workers=1, progress_every=0)

    assert len(table["records"]) == 1
    assert table["records"][0]["canonical_smiles"] == "CCO"
    assert table["fingerprints"].shape == (1, 2048)
    assert table["fingerprints"].dtype == np.uint8
    assert table["descriptors"].shape == (1, len(descriptor_names))
    assert table["descriptors"].dtype == np.float32
    assert table["raw_to_index"]["CCO"] == table["raw_to_index"]["OCC"]


def test_row_indices_and_dummy_embeddings(tmp_path):
    rows = [
        CleanRow(
            split="validation",
            source_index=7,
            label="different",
            endpoint_key="103530|IC50|continuous",
            endpoint_text="- Measurement type: IC50\n- Assay description: example",
            endpoint_text_hash="hash",
            molecule_a_smiles="CCO",
            molecule_b_smiles="CCN",
            metadata={"endpoint_id": "103530|IC50|continuous", "similarity_bucket": 4},
        )
    ]
    clean_by_split = {"validation": rows}
    endpoints = collect_endpoints(clean_by_split)
    molecule_table, _ = build_molecule_feature_table(clean_by_split, workers=1, progress_every=0)

    stats = write_row_indices(tmp_path, clean_by_split, endpoints, molecule_table)
    arrays = np.load(tmp_path / "validation.npz")

    assert stats["validation"] == {"n_rows": 1, "n_indexed": 1, "n_skipped_invalid_molecule": 0}
    assert arrays["endpoint_index"].tolist() == [0]
    assert arrays["molecule_a_index"].tolist() == [0]
    assert arrays["molecule_b_index"].tolist() == [1]
    assert arrays["label"].tolist() == [0]
    assert arrays["source_index"].tolist() == [7]

    embeddings = dummy_embeddings([endpoints[0]["endpoint_text"]], dim=32)
    assert embeddings.shape == (1, 32)
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)


def test_write_jsonl_preserves_metadata(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"metadata": {"assay_type": "F"}, "label": "similar"}])

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["metadata"]["assay_type"] == "F"
    assert row["label"] == "similar"
