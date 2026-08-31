from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools.chembl_tool.common.starling import smiles_identity_audit

from tools.chembl_tool.common.starling.smiles_identity_audit import (
    NameExtractionItem,
    NameExtractionResponse,
    _call_name_extraction,
    _compare,
    _pubchem_lookup_name,
    _run_batches,
    _treatment_readout_ambiguity,
    _verbatim_name_span,
)


def _row(identifier: str, smiles: str) -> dict[str, object]:
    return {
        "id": identifier,
        "source_id": "direct_bbb",
        "source_smiles": smiles,
        "canonical_smiles": smiles,
    }


def test_identity_comparison_distinguishes_match_form_and_mismatch() -> None:
    extraction = {
        "status": "single_explicit_subject",
        "molecule_name": "ethanol",
        "evidence_span": "ethanol",
    }
    exact = _compare(
        _row("exact", "CCO"),
        extraction,
        {"status": "resolved", "smiles": "CCO"},
    )
    assert exact["identity_comparison"] == "exact_identity"

    salt = _compare(
        _row("salt", "CC[NH3+].[Cl-]"),
        extraction,
        {"status": "resolved", "smiles": "CCN"},
    )
    assert salt["identity_comparison"] == "same_parent_or_form"

    tautomeric_source = _compare(
        _row(
            "tautomer",
            "CCOc1ccc(-c2nc3ccc(-c4nc5ccc(N6CCN(C)CC6)cc5n4)cc3n2)cc1",
        ),
        extraction,
        {
            "status": "resolved",
            "smiles": (
                "CCOC1=CC=C(C=C1)C2=NC3=C(N2)C=C(C=C3)"
                "C4=NC5=C(N4)C=C(C=C5)N6CCN(CC6)C"
            ),
        },
    )
    assert tautomeric_source["identity_comparison"] == "same_parent_or_form"

    salt_component = _compare(
        _row("metal", "O=S(=O)([O-])[O-].[Cu+2]"),
        extraction,
        {"status": "resolved", "smiles": "[Cu]"},
    )
    assert salt_component["identity_comparison"] == "same_parent_or_form"

    same_formula = _compare(
        _row("isomer", "CC1=NSCC1=O"),
        extraction,
        {"status": "resolved", "smiles": "CN1C(=O)C=CS1"},
    )
    assert same_formula["identity_comparison"] == "same_formula_isomer_keep"

    wildcard = _compare(
        _row("mixture", "*[N+](C)(C)Cc1ccccc1.[Cl-]"),
        extraction,
        {"status": "resolved", "smiles": "CCCC[N+](C)(C)Cc1ccccc1.[Cl-]"},
    )
    assert wildcard["identity_comparison"] == "generic_or_mixture_keep"

    mismatch = _compare(
        _row("mismatch", "C"),
        extraction,
        {"status": "resolved", "smiles": "CCO"},
    )
    assert mismatch["identity_comparison"] == "different_parent"


def test_unresolved_pubchem_result_is_retained_without_intervention() -> None:
    result = _compare(
        _row("row", "CCO"),
        {
            "status": "subject_not_explicit",
            "molecule_name": None,
            "evidence_span": None,
        },
        {"status": "not_attempted"},
    )
    assert result["identity_comparison"] == "unresolved_keep"


def test_pubchem_lookup_rejects_ambiguous_codes_and_strips_explicit_aliases() -> None:
    assert _pubchem_lookup_name("Compound 3") is None
    assert _pubchem_lookup_name("compound 6c") is None
    assert _pubchem_lookup_name("3-OMG") is None
    assert _pubchem_lookup_name("dogs") is None
    assert _pubchem_lookup_name("Pregabalin (PGB)") == "Pregabalin"
    assert _pubchem_lookup_name("Berberine nanocrystals") == "Berberine"
    assert _pubchem_lookup_name("[11C]elacridar") == "elacridar"


def test_pubchem_cache_supports_bounded_concurrent_writes(
    tmp_path: Path, monkeypatch
) -> None:
    class Response:
        status_code = 404

    class Session:
        def get(self, url: str, timeout: float):
            return Response()

    monkeypatch.setattr(smiles_identity_audit.requests, "Session", Session)
    monkeypatch.setattr(smiles_identity_audit.time, "sleep", lambda _: None)
    path = tmp_path / "pubchem.sqlite"
    cache = smiles_identity_audit.PubChemCache(path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda name: cache.resolve(name, 1), map(str, range(8))))
    cache.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count(*) FROM lookup").fetchone()[0] == 8
    finally:
        connection.close()
    assert {result["status"] for result in results} == {"not_found"}


def test_name_smiles_comparison_emits_candidates_without_stage1_writes(
    tmp_path: Path, monkeypatch
) -> None:
    extraction_dir = tmp_path / "name_extraction/full"
    extraction_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            ("exact", "explicit", "Ethanol", "CCO"),
            ("conflict", "explicit", "Methane", "CCO"),
            ("code", "explicit", "Compound 3", "CCO"),
            ("absent", "absent", None, "CCO"),
        ],
        columns=[
            "id",
            "extraction_status",
            "extracted_molecule_name",
            "canonical_smiles",
        ],
    )
    rows["source_smiles"] = rows["canonical_smiles"]
    rows["task_id"] = "bioavailability_ma"
    rows["source_id"] = "fixture"
    rows["source_row_number"] = range(1, len(rows) + 1)
    rows["endpoint_name"] = "concentration"
    rows["measurement_text"] = "1 mg/L"
    rows["support_text"] = rows["extracted_molecule_name"].fillna("No subject named.")
    extraction_path = extraction_dir / "row_extractions.parquet"
    rows.to_parquet(extraction_path, index=False)
    (extraction_dir / "manifest.json").write_text(
        json.dumps({"automatic_gate_status": "pass", "selected_rows": len(rows)}),
        encoding="utf-8",
    )

    def fake_resolve(self, name: str, timeout_s: float):
        smiles = {"Ethanol": "CCO", "Methane": "C"}[name]
        return {"status": "resolved", "query_name": name, "smiles": smiles}

    monkeypatch.setattr(smiles_identity_audit.PubChemCache, "resolve", fake_resolve)
    output_dir = tmp_path / "audit"
    smiles_identity_audit._run_name_smiles_comparison(
        SimpleNamespace(
            output_dir=output_dir,
            name_extractions=extraction_path,
            pubchem_cache=tmp_path / "pubchem.sqlite",
            pubchem_timeout_s=1.0,
            pubchem_workers=2,
        )
    )

    comparison_dir = output_dir / "name_smiles_comparison/v1"
    comparisons = pd.read_parquet(comparison_dir / "comparisons.parquet")
    conflicts = pd.read_parquet(comparison_dir / "conflict_candidates.parquet")
    manifest = json.loads((comparison_dir / "manifest.json").read_text())
    assert dict(zip(comparisons["id"], comparisons["identity_comparison"])) == {
        "exact": "exact_identity",
        "conflict": "different_parent",
        "code": "unresolved_keep",
    }
    assert conflicts["id"].tolist() == ["conflict"]
    assert manifest["conflict_candidate_rows"] == 1
    assert manifest["llm_requests"] == 0
    assert manifest["writes_stage1_or_reviewed_ledgers"] is False


def test_treatment_is_ambiguous_when_endpoint_names_another_readout() -> None:
    row = {
        "endpoint_name": "EB ratio (Evans blue/plasma Evans blue)",
        "support_text": "The agmatine (Agm) treatment group had an EB ratio of 1.14.",
    }
    assert _treatment_readout_ambiguity(row, "agmatine")
    assert not _treatment_readout_ambiguity(
        {**row, "endpoint_name": "agmatine concentration"}, "agmatine"
    )
    assert _treatment_readout_ambiguity(
        {
            "endpoint_name": "intrinsic clearance",
            "support_text": "With 20 µM gefitinib, CLint for APC formation was 1.3.",
        },
        "gefitinib",
    )
    assert _treatment_readout_ambiguity(
        {
            "endpoint_name": "permeability",
            "support_text": "Verapamil (100 µM) decreased Papp to 3.53.",
        },
        "Verapamil",
    )
    assert _treatment_readout_ambiguity(
        {
            "endpoint_name": "skin permeability",
            "support_text": "Administration of isoproterenol increased skin permeability.",
        },
        "isoproterenol",
    )
    assert _treatment_readout_ambiguity(
        {
            "endpoint_name": "B cell infiltration",
            "support_text": "Challenge with atranol affected mice sensitized with chloroatranol.",
        },
        "chloroatranol",
    )


def test_batch_log_is_resumable_and_requires_complete_cardinality(tmp_path: Path) -> None:
    rows = [_row("a", "C"), _row("b", "CC")]
    calls: list[list[str]] = []

    def fake_call(client: object, batch: list[dict[str, object]], model: str):
        calls.append([str(row["id"]) for row in batch])
        return (
            [
                {
                    "id": row["id"],
                    "status": "subject_not_explicit",
                    "molecule_name": None,
                    "evidence_span": None,
                }
                for row in batch
            ],
            {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )

    kwargs = {
        "batch_size": 1,
        "workers": 2,
        "model": "fixture-model",
        "prompt_version": "fixture-prompt",
        "log_path": tmp_path / "batches.jsonl",
        "call": fake_call,
        "client": object(),
    }
    results, usage, requests = _run_batches(rows, **kwargs)
    assert set(results) == {"a", "b"}
    assert usage["total_tokens"] == 24
    assert requests == 2
    assert len(calls) == 2

    namespaced = {**kwargs, "request_namespace": "http://dgx019:8000/v1"}
    results, _, requests = _run_batches(rows, **namespaced)
    assert set(results) == {"a", "b"}
    assert requests == 2
    assert len(calls) == 4

    results, usage, requests = _run_batches(rows, **kwargs)
    assert set(results) == {"a", "b"}
    assert usage["total_tokens"] == 24
    assert requests == 2
    assert len(calls) == 4


def test_batches_continuously_distribute_across_endpoints(tmp_path: Path) -> None:
    rows = [_row(str(index), "C") for index in range(4)]
    calls: list[tuple[str, str]] = []

    def fake_call(client: str, batch: list[dict[str, object]], model: str):
        calls.append((client, str(batch[0]["id"])))
        return ([{"id": batch[0]["id"]}], {})

    results, _, requests = _run_batches(
        rows,
        batch_size=1,
        workers=2,
        model="fixture-model",
        prompt_version="fixture-prompt",
        log_path=tmp_path / "batches.jsonl",
        call=fake_call,
        client="a",
        endpoint_clients=[("a", "http://a/v1"), ("b", "http://b/v1")],
    )

    assert requests == 4
    assert set(calls) == {("a", "0"), ("b", "1"), ("a", "2"), ("b", "3")}
    assert results["0"]["_request_namespace"] == "http://a/v1"
    assert results["1"]["_request_namespace"] == "http://b/v1"


def test_discovery_inventories_candidates_without_external_calls(
    tmp_path: Path, monkeypatch
) -> None:
    stage1_path = tmp_path / "stage" / "01_cleaned" / "records.parquet"
    stage1_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "cleaned_record_id": "candidate",
                "source_id": "without_name",
                "source_row_number": 1,
                "source_record_id": "r1",
                "endpoint_name": "AUC",
                "measurement_text": "10",
                "support_text": "Compound A had an AUC of 10.",
                "molecule_name": None,
                "source_smiles": "CCO",
                "canonical_smiles": "CCO",
                "structure_status": "resolved",
                "smiles": "CCO",
            },
            {
                "cleaned_record_id": "missing_smiles",
                "source_id": "without_name",
                "source_row_number": 2,
                "source_record_id": "r2",
                "source_smiles": None,
                "molecule_name": None,
                "structure_status": "missing_structure",
            },
            {
                "cleaned_record_id": "already_named",
                "source_id": "with_name",
                "source_row_number": 3,
                "source_record_id": "r3",
                "molecule_name": "ethane",
                "support_text": "Ethane was measured.",
                "source_smiles": "CC",
                "canonical_smiles": "CC",
                "structure_status": "resolved",
            },
            {
                "cleaned_record_id": "missing_support",
                "source_id": "without_name",
                "source_row_number": 4,
                "source_record_id": "r4",
                "molecule_name": None,
                "support_text": None,
                "source_smiles": "CCC",
                "canonical_smiles": "CCC",
                "structure_status": "resolved",
            },
        ]
    ).to_parquet(stage1_path, index=False)
    inventory_path = stage1_path.parents[1] / "00_source/source_inventory.json"
    inventory_path.parent.mkdir()
    inventory_path.write_text(
        json.dumps({"source_hashes": {"without_name": "abc", "with_name": "def"}}),
        encoding="utf-8",
    )
    policy = SimpleNamespace(
        record_contract=SimpleNamespace(
            sources={
                "without_name": SimpleNamespace(source_columns=("smiles",)),
                "with_name": SimpleNamespace(source_columns=("smiles", "molecule_name")),
            }
        )
    )
    monkeypatch.setattr(smiles_identity_audit, "_policy", lambda task: policy)
    output_dir = tmp_path / "audit"
    smiles_identity_audit.run(
        argparse.Namespace(
            task="bbb_martins",
            stage1=stage1_path,
            output_dir=output_dir,
            discover_only=True,
            extraction_batch_size=50,
        )
    )

    candidates = pd.read_parquet(
        output_dir / "bbb_martins/candidates/candidates.parquet"
    )
    manifest = json.loads(
        (output_dir / "bbb_martins/candidates/manifest.json").read_text()
    )
    assert candidates["id"].tolist() == ["candidate"]
    assert manifest["candidate_rows"] == 1
    assert manifest["external_requests"] == 0
    assert manifest["candidate_definition"] == {
        "canonical_smiles_nonempty": True,
        "stage1_molecule_name_blank": True,
        "structure_status": "resolved",
        "support_text_nonempty": True,
    }


def test_name_extraction_sends_measurement_context_without_structure() -> None:
    captured: dict[str, object] = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=NameExtractionResponse(
                    items=[
                        NameExtractionItem(
                            id="r000", status="explicit", molecule_name="Aspirin"
                        )
                    ]
                ),
                usage=SimpleNamespace(
                    input_tokens=8, output_tokens=3, total_tokens=11
                ),
            )

    client = SimpleNamespace(responses=Responses())
    items, usage = _call_name_extraction(
        client,
        [
            {
                "id": "private-row-id",
                "support_text": "Aspirin was measured.",
                "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(O)=O",
                "endpoint_name": "IC50",
                "measurement_text": "12 nM",
            }
        ],
        "gpt-5.6-luna",
    )
    request = json.loads(captured["input"][1]["content"])
    assert request == {
        "items": [
            {
                "id": "r000",
                "endpoint_name": "IC50",
                "measurement_text": "12 nM",
                "support_text": "Aspirin was measured.",
            }
        ]
    }
    assert "canonical_smiles" not in request["items"][0]
    assert captured["reasoning"] == {"effort": "xhigh"}
    assert captured["store"] is False
    assert items == [
        {"id": "private-row-id", "status": "explicit", "molecule_name": "Aspirin"}
    ]
    assert usage["total_tokens"] == 11


def test_full_extraction_reuses_legacy_only_for_singleton_contexts(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "audit"
    pilot_dir = output_dir / "name_extraction/pilot"
    pilot_dir.mkdir(parents=True)
    support_a = "Aspirin had an AUC of 10."
    support_b = "Metformin had reported exposure measurements."
    pd.DataFrame(
        [
            {
                "support_id": hashlib.sha256(support_a.encode()).hexdigest(),
                "status": "explicit",
                "molecule_name": "Aspirin",
                "extraction_model": "gpt-5.6-luna",
                "extraction_base_url": "https://api.openai.com/v1",
                "extraction_prompt_version": "missing_molecule_name_extraction.v2",
            },
            {
                "support_id": hashlib.sha256(support_b.encode()).hexdigest(),
                "status": "explicit",
                "molecule_name": "Metformin",
                "extraction_model": "gpt-5.6-luna",
                "extraction_base_url": "https://api.openai.com/v1",
                "extraction_prompt_version": "missing_molecule_name_extraction.v2",
            },
        ]
    ).to_parquet(pilot_dir / "unique_support_extractions.parquet", index=False)
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    (gate_dir / "manifest.json").write_text(
        json.dumps({"automatic_gate_status": "pass"}), encoding="utf-8"
    )
    rows = pd.DataFrame(
        [
            ("a", support_a, "AUC", "10"),
            ("b1", support_b, "AUC", "20"),
            ("b2", support_b, "Cmax", "5"),
        ],
        columns=["id", "support_text", "endpoint_name", "measurement_text"],
    )
    rows["task_id"] = "bioavailability_ma"
    rows["source_id"] = "fixture"
    rows["source_row_number"] = range(1, 4)
    rows["molecule_name"] = None
    rows["pilot_kind"] = "target"
    monkeypatch.setattr(
        smiles_identity_audit, "_name_extraction_rows", lambda args: (rows, len(rows))
    )
    monkeypatch.setattr(
        smiles_identity_audit,
        "_manual_review_gate",
        lambda *args: {"status": "pass"},
    )
    monkeypatch.setattr(smiles_identity_audit, "openai_client", lambda **kwargs: object())
    called_rows: list[dict[str, object]] = []

    def fake_batches(batch_rows, **kwargs):
        called_rows.extend(batch_rows)
        return (
            {
                row["id"]: {
                    "id": row["id"],
                    "status": "explicit",
                    "molecule_name": "Metformin",
                }
                for row in batch_rows
            },
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            1,
        )

    monkeypatch.setattr(smiles_identity_audit, "_run_batches", fake_batches)
    smiles_identity_audit._run_name_extraction(
        SimpleNamespace(
            output_dir=output_dir,
            name_extraction_scope="full",
            name_extraction_gate_dir=gate_dir,
            pilot_manual_review=None,
            name_extraction_model="nvidia/DeepSeek-V4-Flash-NVFP4",
            name_extraction_base_url="http://dgx019:8000/v1",
            name_extraction_no_auth=True,
            keys_path=None,
            workers=2,
            extraction_batch_size=50,
        )
    )
    assert {row["endpoint_name"] for row in called_rows} == {"AUC", "Cmax"}
    manifest = json.loads(
        (output_dir / "name_extraction/full/manifest.json").read_text()
    )
    assert manifest["legacy_singleton_contexts_reused"] == 1
    assert manifest["legacy_multi_context_support_texts_rerun"] == 1


def test_verbatim_name_span_preserves_source_typography() -> None:
    support = "The permeability of SN‐38 was measured."
    assert _verbatim_name_span("SN-38", support) == "SN‐38"
    assert _verbatim_name_span("sn-38", support) == "SN‐38"
    assert _verbatim_name_span("amenamevir", "amamevir was measured") is None
