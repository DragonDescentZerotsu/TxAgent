from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.assay_reranking.build_v11_cache import (
    _catalog_record,
    _load_bridge,
    _selected_fp32_head_logits,
    _write_demand,
)
from tools.chembl_tool.common.assay_reranking.v11 import (
    BACKBONE_DTYPE,
    CANDIDATE_SCHEMA_VERSION,
    CATALOG_SCHEMA_VERSION,
    LOGIT_EXTRACTION_DTYPE,
    SCORING_CONTRACT_VERSION,
    PromptScore,
    V11CachedAssayReranker,
    V11PromptRenderer,
    V11ScoreCache,
    build_prompt_task,
    model_profile,
    verify_vendored_assets,
)
from tools.chembl_tool.common.task_workflows.reasoning_batch import _parse_args
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch import (
    CONFIG as BIO_CONFIG,
)
from tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch import (
    CONFIG as SKIN_CONFIG,
)
from tools.chembl_tool.tasks.skin_reaction.run_reasoning_pipeline import (
    _group_prompt_payload as skin_group_prompt_payload,
)


def _bio_record(record_id: str, measurement: str = "42") -> dict:
    return {
        "canonical_record_id": record_id,
        "task_id": "bioavailability_ma",
        "source_id": "fa",
        "group_id": "Fa.absorption_solubility_permeability",
        "canonical_smiles": "CCO",
        "measurement_kind": "non_scalar",
        "smiles": "CCO",
        "endpoint_name": "solubility",
        "assay_system": "aqueous assay",
        "condition_medium": "water",
        "biological_context": None,
        "formulation_or_solid_form": None,
        "qualifying_conditions": None,
        "unit_text": "mg/mL",
        "measurement_text": measurement,
        "support_text": "reported evidence",
        "extra_details": None,
        "pmid": "1",
        "extraction_id": "ext",
        "global_identifier": "record",
        "confidence": 0.8,
        "paragraph_idx": 1,
        # Deliberately not pair/calibration eligible. Retrieval scoring must ignore this.
        "canonicalization_status": "non_scalar_measurement",
        "bucket_eligible": False,
        "calibration_valid": False,
    }


def test_vendored_assets_match_pinned_hashes():
    observed = verify_vendored_assets()
    assert observed["prompt_projection.json"] == (
        "339f3bf95ae71e14fd5b288a0bdec8d66c5fadfd8d2aa13b76b9f9c13b48fd99"
    )


def test_bbb_multitask_model_is_pinned():
    profile = model_profile("bbb_martins")
    assert profile["model"] == (
        "jiosephlee/assay-transfer-tool-soft-v11-multitask-with-categorical"
    )
    assert profile["revision"] == "3220147ce9e56240670bb120e582f35cb66ec434"


def test_selected_output_head_uses_fp32_accumulation():
    torch = pytest.importorskip("torch")
    width = 4096
    hidden = torch.ones((1, width), dtype=torch.bfloat16)
    head = torch.nn.Linear(width, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.fill_(1.0)
        head.weight[1, 0] = 1.0078125

    bf16_a = (hidden * head.weight[0]).sum()
    bf16_b = (hidden * head.weight[1]).sum()
    fp32_a, fp32_b = _selected_fp32_head_logits(
        hidden, head, [0], [1], torch
    )

    assert bf16_a.item() == bf16_b.item()
    assert fp32_a.dtype == torch.float32
    assert fp32_b.dtype == torch.float32
    assert fp32_b.item() > fp32_a.item()
    assert fp32_b.item() - fp32_a.item() == pytest.approx(0.0078125)


@pytest.mark.parametrize("config", [BIO_CONFIG, SKIN_CONFIG])
def test_v11_assay_transfer_defaults_to_matching_compact_index(config):
    args = _parse_args(
        config,
        [
            "--retrieval-strategy",
            "assay_transfer_tool",
            "--assay-transfer-profile",
            "v11_with_categorical",
            "--group-prompt-format",
            "assay_transfer_tool",
        ],
    )

    assert args.index == config.v11_index_default
    assert args.assay_transfer_initial_morgan_filter == 50
    assert args.min_similarity == 0.0


@pytest.mark.parametrize("config", [BIO_CONFIG, SKIN_CONFIG])
def test_v11_assay_transfer_preserves_explicit_index(config):
    args = _parse_args(
        config,
        [
            "--retrieval-strategy",
            "assay_transfer_tool",
            "--assay-transfer-profile",
            "v11_with_categorical",
            "--group-prompt-format",
            "assay_transfer_tool",
            "--index",
            "custom-index",
        ],
    )

    assert args.index == "custom-index"


def test_query_context_copy_hides_retrieval_measurement():
    renderer = V11PromptRenderer("bioavailability_ma")
    record = {**_bio_record("record-1"), "task_id": "bioavailability_ma"}
    prompt = renderer.render(record, "CCN")

    assert "<SMILES>CCN</SMILES>" in prompt
    assert "<SMILES>CCO</SMILES>" in prompt
    assert prompt.count("known measurement: 42") == 1
    assert prompt.index("Molecule A (known)") < prompt.index("known measurement: 42")
    assert "(A) Transfer" in prompt and "(B) Does not transfer" in prompt


def test_v11_endpoint_identity_uses_prompt_constant_and_shared_canonicalizer():
    renderer = V11PromptRenderer("bioavailability_ma")
    hf_record = {
        **_bio_record("hf-record"),
        "source_id": "hf_bioavailability",
    }
    hf_record.pop("endpoint_name")
    fa_record = {**_bio_record("fa-record"), "endpoint_name": "Aqueous-Solubility"}

    assert renderer.canonical_endpoint_key(hf_record) == "oral_bioavailability"
    assert renderer.canonical_endpoint_key(fa_record) == "aqueous_solubility"


def test_v11_endpoint_identity_groups_missing_endpoint_within_source_family():
    renderer = V11PromptRenderer("bbb_martins")
    record = {"source_id": "direct_bbb", "endpoint_name": None}

    assert renderer.canonical_endpoint_key(record) == "unspecified_endpoint:direct_bbb"


def test_catalog_keeps_non_scalar_stage07_record_without_calibration():
    payload = _catalog_record(
        "bioavailability_ma",
        _bio_record("record-uncalibrated"),
        V11PromptRenderer("bioavailability_ma"),
        {"continuous", "binary"},
    )

    assert payload["record_id"] == "record-uncalibrated"
    assert payload["measurement_kind"] == "non_scalar"
    assert payload["training_measurement_kind_supported"] is False
    assert "bucket_eligible" not in payload
    assert "calibration_valid" not in payload


def test_prompt_demand_collapses_identical_cache_keys(tmp_path: Path):
    renderer = V11PromptRenderer("bioavailability_ma")
    profile = model_profile("bioavailability_ma")
    records = [
        _catalog_record(
            "bioavailability_ma",
            _bio_record(record_id),
            renderer,
            {"continuous", "binary"},
        )
        for record_id in ("record-1", "record-2")
    ]
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE refs(
            record_id TEXT, query_smiles TEXT, group_id TEXT, molecule_id TEXT
        );
        CREATE TABLE records(record_id TEXT PRIMARY KEY, payload TEXT);
        """
    )
    for record in records:
        connection.execute(
            "INSERT INTO records VALUES (?, ?)",
            (record["record_id"], json.dumps(record)),
        )
        connection.execute(
            "INSERT INTO refs VALUES (?, ?, ?, ?)",
            (
                record["record_id"],
                "CCN",
                "Fa.absorption_solubility_permeability",
                "STARLING_1",
            ),
        )
    path = tmp_path / "demand_rows.jsonl"

    n_scores, n_duplicates = _write_demand(
        task_id="bioavailability_ma",
        connection=connection,
        rows_path=path,
        renderer=renderer,
        model=profile["model"],
        revision=profile["revision"],
    )
    connection.close()

    assert (n_scores, n_duplicates) == (1, 1)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_stage07_bridge_includes_nonrepresentative_records(tmp_path: Path):
    path = tmp_path / "molecule_family_records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "evidence_id": "evidence",
                    "canonical_record_id": "representative",
                    "record_order": 1,
                    "representative_rank": 1,
                },
                {
                    "evidence_id": "evidence",
                    "canonical_record_id": "not-representative",
                    "record_order": 0,
                    "representative_rank": None,
                },
            ]
        ),
        path,
    )

    assert _load_bridge(path)["evidence"] == (
        "not-representative",
        "representative",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_record_level_reranker_allows_same_molecule_to_repeat(tmp_path: Path):
    renderer = V11PromptRenderer("bioavailability_ma")
    profile = model_profile("bioavailability_ma")
    records = [
        _catalog_record(
            "bioavailability_ma",
            _bio_record(f"record-{index}", measurement=str(index)),
            renderer,
            {"continuous", "binary"},
        )
        for index in (1, 2)
    ]
    catalog = tmp_path / "catalog.jsonl"
    _write_jsonl(
        catalog,
        [
            {
                "record_type": "catalog_metadata",
                "schema_version": CATALOG_SCHEMA_VERSION,
                "task_id": "bioavailability_ma",
                "model": profile["model"],
                "model_revision": profile["revision"],
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "backbone_dtype": BACKBONE_DTYPE,
                "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
                "template_hash": renderer.template_hash,
                "projection_hash": renderer.projection_hash,
            },
            *records,
        ],
    )
    manifest = tmp_path / "candidate_manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "record_type": "manifest_metadata",
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "task_id": "bioavailability_ma",
            },
            {
                "record_type": "candidate_group",
                "query_smiles": "CCN",
                "group_id": "Fa.absorption_solubility_permeability",
                "candidates": [
                    {
                        "molecule_id": "STARLING_1",
                        "record_ids": ["record-1", "record-2"],
                    }
                ],
            },
        ],
    )
    cache_path = tmp_path / "scores.sqlite3"
    cache = V11ScoreCache(cache_path, mode="read_write")
    tasks = [
        build_prompt_task(
            renderer,
            record,
            query_smiles="CCN",
            group_id="Fa.absorption_solubility_permeability",
            molecule_id="STARLING_1",
            model=profile["model"],
            model_revision=profile["revision"],
        )
        for record in records
    ]
    cache.write_batch(
        tasks,
        [
            PromptScore(tasks[0].cache_key, 1.0, 0.0, 0.7),
            PromptScore(tasks[1].cache_key, 2.0, 0.0, 0.9),
        ],
    )
    cache.close()

    reranker = V11CachedAssayReranker(
        task_id="bioavailability_ma",
        catalog_path=catalog,
        candidate_manifest_path=manifest,
        cache_path=cache_path,
    )
    selected = reranker.rerank_records(
        query_smiles="CCN",
        group_id="Fa.absorption_solubility_permeability",
        candidates=[
            {
                "molecule_chembl_id": "STARLING_1",
                "canonical_smiles": "CCO",
                "similarity": 0.5,
                "evidence_rows": [{}],
            }
        ],
    )
    reranker.cache.close()

    assert [row["transfer_winning_record_id"] for row in selected] == [
        "record-2",
        "record-1",
    ]
    assert [row["molecule_chembl_id"] for row in selected] == [
        "STARLING_1",
        "STARLING_1",
    ]
    assert [
        row["transfer_winning_record"]["canonical_endpoint_key"] for row in selected
    ] == ["solubility", "solubility"]


def test_skin_group_prompt_exposes_only_rounded_score_and_minimal_record():
    source_contract = {
        "contract_version": "source_column_contract.v1",
        "source_id": "direct_skin_reaction",
        "source_or_simply_cleaned": {"endpoint_name": True},
    }
    group = {
        "group_id": "Mechanism.tier_1",
        "tier": "Tier 1",
        "endpoint_group": "direct_skin_reaction",
        "transfer_neighbor_selection": {"selection_score_is_llm_visible": True},
        "neighbors": [
            {
                "rank": 1,
                "molecule_chembl_id": "STARLING_1",
                "canonical_smiles": "CCO",
                "similarity": 0.4,
                "similarity_bucket": "distant_analog",
                "transfer_selection_score": 0.876,
                "transfer_winning_record": {
                    "record_id": "record",
                    "source_contract": source_contract,
                    "source_fields": {"endpoint_name": "sensitization"},
                },
                "prefetched_comparisons": [],
                "evidence_rows": [
                    {
                        "evidence_source": "starling",
                        "canonical_smiles": "CCO",
                        "group_id": "Mechanism.tier_1",
                        "standard_type": "sensitization",
                    }
                ],
                "shared_assay_context": {},
            }
        ],
    }

    payload = skin_group_prompt_payload({"canonical_smiles": "CCN"}, group)

    assert payload["neighbors"][0]["assay_transfer_score"] == 0.88
    assert payload["neighbors"][0]["assay_transfer_record"]["contract_version"] == (
        "minimal_evidence.v1"
    )
    assert "transfer_selection_score" not in str(payload)
    assert "uncalibrated" in str(payload)


def test_skin_group_prompt_groups_multiple_records_without_audit_identifiers():
    source_contract = {
        "contract_version": "source_column_contract.v1",
        "source_id": "direct_skin_reaction",
        "source_or_simply_cleaned": {"endpoint_name": True},
    }
    record_one = {
        "record_id": "record-one",
        "canonical_endpoint_key": "sensitization",
        "source_contract": source_contract,
        "source_fields": {"endpoint_name": "sensitization"},
    }
    record_two = {
        "record_id": "record-two",
        "canonical_endpoint_key": "erythema",
        "source_contract": source_contract,
        "source_fields": {"endpoint_name": "erythema"},
    }
    group = {
        "group_id": "Mechanism.tier_1",
        "tier": "Tier 1",
        "endpoint_group": "direct_skin_reaction",
        "transfer_neighbor_selection": {"selection_score_is_llm_visible": True},
        "neighbors": [
            {
                "rank": 1,
                "molecule_chembl_id": "STARLING_1",
                "canonical_smiles": "CCO",
                "similarity": 0.4,
                "similarity_bucket": "distant_analog",
                "transfer_selection_score": 0.876,
                "transfer_winning_record": record_one,
                "transfer_selected_records": [
                    {
                        "record_rank": 1,
                        "transfer_selection_score": 0.876,
                        "transfer_winning_record_id": "audit-one",
                        "canonical_endpoint_key": "sensitization",
                        "transfer_winning_record": record_one,
                    },
                    {
                        "record_rank": 2,
                        "transfer_selection_score": 0.754,
                        "transfer_winning_record_id": "audit-two",
                        "canonical_endpoint_key": "erythema",
                        "transfer_winning_record": record_two,
                    },
                ],
                "prefetched_comparisons": [],
                "evidence_rows": [],
                "shared_assay_context": {},
            }
        ],
    }

    payload = skin_group_prompt_payload({"canonical_smiles": "CCN"}, group)
    neighbor = payload["neighbors"][0]

    assert "assay_transfer_score" not in neighbor
    assert "assay_transfer_record" not in neighbor
    assert [row["assay_transfer_score"] for row in neighbor["assay_transfer_records"]] == [
        0.88,
        0.75,
    ]
    serialized = json.dumps(payload, sort_keys=True)
    assert "sensitization" in serialized and "erythema" in serialized
    assert "audit-one" not in serialized and "audit-two" not in serialized
    assert "record-one" not in serialized and "record-two" not in serialized
    assert "endpoint-distinct" in serialized
