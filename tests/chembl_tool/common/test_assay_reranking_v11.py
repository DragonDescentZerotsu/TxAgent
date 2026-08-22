from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.assay_reranking import build_v11_cache
from tools.chembl_tool.common.assay_reranking.build_v11_cache import (
    _catalog_record,
    _eligible_record_ids,
    _finalize_compact_cache,
    _iter_compact_missing_batches,
    _load_bridge,
    _parse_args as parse_cache_args,
    _selected_fp32_head_logits,
    _task_paths,
    _write_demand,
)
from tools.chembl_tool.common.assay_reranking.v11 import (
    BACKBONE_DTYPE,
    COMPACT_CACHE_SCHEMA_VERSION,
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
        "430c561f0788eccfb03a3f7d44ae96e4184257bf8266a186b475fca4f9165af8"
    )


def test_bbb_task_specific_model_is_pinned():
    profile = model_profile("bbb_martins")
    assert profile["model"] == (
        "jiosephlee/assay-transfer-tool-soft-v11-bbb-martins-with-categorical"
    )
    assert profile["revision"] == "6b4795761fb7d2daf663f14927cdfb210a4d3e40"


def test_skin_v11_1_task_specific_model_is_pinned():
    profile = model_profile("skin_reaction")
    assert profile["model"] == (
        "jiosephlee/assay-transfer-tool-soft-v11.1-skin-reaction-with-categorical"
    )
    assert profile["revision"] == "4a0442ae2e1cdc20de102c3159766689a6704358"


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


def test_compact_cache_reranks_without_catalog_or_candidate_jsonl(tmp_path: Path):
    path = tmp_path / "scores.sqlite3"
    renderer = V11PromptRenderer("bioavailability_ma")
    profile = model_profile("bioavailability_ma")
    metadata = {
        "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "task_id": "bioavailability_ma",
        "model": profile["model"],
        "model_revision": profile["revision"],
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "backbone_dtype": BACKBONE_DTYPE,
        "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
        "template_hash": renderer.template_hash,
        "projection_hash": renderer.projection_hash,
        "candidate_contract": "tanimoto_identity_exclusion_then_eligible_pool.v1",
    }
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE queries(query_id INTEGER PRIMARY KEY, query_smiles TEXT UNIQUE);
        CREATE TABLE groups_dim(group_key INTEGER PRIMARY KEY, group_id TEXT UNIQUE);
        CREATE TABLE molecules(molecule_key INTEGER PRIMARY KEY, molecule_chembl_id TEXT UNIQUE);
        CREATE TABLE records(record_key INTEGER PRIMARY KEY, external_record_id TEXT UNIQUE, payload TEXT);
        CREATE TABLE scores(score_key INTEGER PRIMARY KEY, transfer_probability REAL);
        CREATE TABLE assignments(
            query_id INTEGER, group_key INTEGER, molecule_key INTEGER,
            record_key INTEGER, score_key INTEGER,
            PRIMARY KEY(query_id, group_key, molecule_key, record_key)
        ) WITHOUT ROWID;
        """
    )
    connection.executemany(
        "INSERT INTO cache_metadata VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in metadata.items()],
    )
    connection.execute("INSERT INTO queries VALUES (1, 'CCN')")
    connection.execute("INSERT INTO groups_dim VALUES (1, 'Fa.absorption_solubility_permeability')")
    connection.execute("INSERT INTO molecules VALUES (1, 'STARLING_1')")
    winning = {
        "record_id": "record-1",
        "canonical_endpoint_key": "aqueous_solubility",
        "canonical_smiles": "CCO",
        "measurement_kind": "continuous",
        "training_measurement_kind_supported": True,
        "source_contract": {"source_id": "fa"},
        "source_fields": {"measurement_text": "42"},
    }
    connection.execute(
        "INSERT INTO records VALUES (1, 'record-1', ?)", (json.dumps(winning),)
    )
    connection.execute("INSERT INTO scores VALUES (1, 0.8)")
    connection.execute("INSERT INTO assignments VALUES (1, 1, 1, 1, 1)")
    connection.commit()
    connection.close()

    reranker = V11CachedAssayReranker(
        task_id="bioavailability_ma",
        cache_path=path,
        model=profile["model"],
        model_revision=profile["revision"],
    )
    try:
        rows = reranker.rerank_records(
            query_smiles="CCN",
            group_id="Fa.absorption_solubility_permeability",
            candidates=[{"molecule_chembl_id": "STARLING_1", "similarity": 0.4}],
        )
    finally:
        reranker.cache.close()
    assert len(rows) == 1
    assert rows[0]["transfer_selection_score"] == pytest.approx(0.8)
    assert rows[0]["transfer_winning_record"] == winning


def test_explicit_lineage_paths_resolve_paper_stage_layout(tmp_path: Path):
    view = tmp_path / "paper" / "evidence" / "bioavailability_starling_v7"
    query = tmp_path / "valid.jsonl"
    cache = tmp_path / "paper" / "assay_transfer_rerank" / "valid"
    args = parse_cache_args(
        [
            "--tasks", "bioavailability_ma",
            "--task-evidence-view", f"bioavailability_ma={view}",
            "--task-query-jsonl", f"bioavailability_ma={query}",
            "--task-cache-dir", f"bioavailability_ma={cache}",
        ]
    )
    paths = _task_paths(args, "bioavailability_ma")
    assert paths["records"] == view / "06_records/records.parquet"
    assert paths["bridge"] == view / "07_molecule_evidence/molecule_family_records.parquet"
    assert paths["index"] == view / "08_neighbor_index"
    assert paths["queries"] == query
    assert paths["cache"] == cache / "scores.sqlite3"


def test_numeric_record_scope_requires_only_bioavailability():
    with pytest.raises(SystemExit):
        parse_cache_args(
            ["--tasks", "bbb_martins", "--record-scope", "numeric_direct"]
        )


def test_model_override_requires_one_task_and_an_immutable_pair():
    revision = "a" * 40
    args = parse_cache_args(
        [
            "--tasks", "bbb_martins",
            "--assay-transfer-model", "organization/model",
            "--assay-transfer-model-revision", revision,
        ]
    )
    assert args.assay_transfer_model == "organization/model"
    assert args.assay_transfer_model_revision == revision
    with pytest.raises(SystemExit):
        parse_cache_args(["--tasks", "bbb_martins", "--assay-transfer-model", "model"])
    with pytest.raises(SystemExit):
        parse_cache_args(
            [
                "--tasks", "bbb_martins", "skin_reaction",
                "--assay-transfer-model", "organization/model",
                "--assay-transfer-model-revision", revision,
            ]
        )

    local = parse_cache_args(
        [
            "--tasks", "bioavailability_ma",
            "--assay-transfer-local-model-dir", "checkpoint",
        ]
    )
    assert local.assay_transfer_local_model_dir == Path("checkpoint")
    with pytest.raises(SystemExit):
        parse_cache_args(
            [
                "--tasks", "bbb_martins", "skin_reaction",
                "--assay-transfer-local-model-dir", "checkpoint",
            ]
        )


def test_processed_gold_catalog_preserves_frozen_voter_labels():
    renderer = V11PromptRenderer("bbb_martins")
    record = {
        "canonical_record_id": "BBB_Martins:source_row:9",
        "source_id": "direct_bbb",
        "source_name": "starling-labs/BBB",
        "group_id": "Tier 1.starling_direct_bbb_evidence",
        "canonical_smiles": "CCO",
        "measurement_kind": "binary",
        "processed_gold_voting_record_key": "BBB_Martins:source_row:9",
        "processed_gold_lineage": "experimental_meaningful_cns_access_v2",
        "processed_gold_split": "train",
        "record_vote": 0,
        "molecule_Y": 1,
        "smiles": "CCO",
        "endpoint_name": "brain concentration",
        "measurement_text": "2",
        "unit_text": "ng/mL",
    }
    catalog = _catalog_record(
        "bbb_martins", record, renderer, {"continuous", "binary"}
    )
    assert catalog["record_vote"] == 0
    assert catalog["molecule_Y"] == 1
    assert catalog["source_fields"]["measurement_text"] == "2"
    assert catalog["source_contract"]["record_contract_version"] == (
        "processed_starling_gold.v1"
    )


def test_compact_resume_tasks_keep_frozen_model_provenance(tmp_path: Path):
    cache = tmp_path / "scores.sqlite3"
    connection = sqlite3.connect(cache)
    connection.executescript(
        """
        CREATE TABLE prompt_tasks(
            prompt_key INTEGER PRIMARY KEY, cache_key TEXT, prompt TEXT
        );
        CREATE TABLE prompt_scores(prompt_key INTEGER PRIMARY KEY, transfer_probability REAL);
        INSERT INTO prompt_tasks VALUES (1, 'cache-key', 'prompt');
        """
    )
    connection.commit()
    connection.close()
    revision = "b" * 40
    batches = list(
        _iter_compact_missing_batches(
            cache, task_id="bbb_martins", model="organization/model",
            revision=revision, batch_size=8,
        )
    )
    assert batches[0][0].model == "organization/model"
    assert batches[0][0].model_revision == revision


def test_bio_record_scope_filters_before_cache_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    records = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"canonical_record_id": "numeric", "source_id": "hf_bioavailability"},
                {"canonical_record_id": "qualitative", "source_id": "hf_bioavailability"},
                {"canonical_record_id": "nondirect", "source_id": "hf_bioavailability"},
                {"canonical_record_id": "mechanism", "source_id": "fa"},
            ]
        ),
        records,
    )
    projected = {
        "numeric": {
            "bioavailability_report_type": "absolute",
            "measurement_text": "42%",
            "species_or_population": "human",
            "qualifying_conditions": None,
        },
        "qualitative": {
            "bioavailability_report_type": "absolute",
            "measurement_text": "high bioavailability",
            "species_or_population": "human",
            "qualifying_conditions": None,
        },
        "nondirect": {
            "bioavailability_report_type": "relative",
            "measurement_text": "42%",
            "species_or_population": "human",
            "qualifying_conditions": None,
        },
    }
    monkeypatch.setattr(
        build_v11_cache,
        "_source_projection",
        lambda _task, row: {"source_fields": projected[row["canonical_record_id"]]},
    )

    assert _eligible_record_ids("bioavailability_ma", records, "labelable_direct") == {
        "numeric",
        "qualitative",
    }
    assert _eligible_record_ids("bioavailability_ma", records, "numeric_direct") == {
        "numeric"
    }


def test_compact_finalization_drops_rendered_prompts_and_build_joins(tmp_path: Path):
    cache = tmp_path / "scores.sqlite3"
    connection = sqlite3.connect(cache)
    connection.executescript(
        """
        CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE prompt_tasks(prompt_key INTEGER PRIMARY KEY, cache_key TEXT, prompt TEXT);
        CREATE TABLE prompt_scores(prompt_key INTEGER PRIMARY KEY, transfer_probability REAL);
        CREATE TABLE prompt_assignments(
            query_id INTEGER, group_key INTEGER, molecule_key INTEGER,
            record_key INTEGER, prompt_key INTEGER
        );
        INSERT INTO prompt_tasks VALUES (1, 'key', 'large rendered prompt');
        INSERT INTO prompt_scores VALUES (1, 0.75);
        INSERT INTO prompt_assignments VALUES (1, 1, 1, 1, 1);
        """
    )
    connection.commit()
    connection.close()
    renderer = V11PromptRenderer("bioavailability_ma")
    profile = model_profile("bioavailability_ma")
    version = {
        "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "status": "prepared",
        "task_id": "bioavailability_ma",
        "profile": "v11_with_categorical",
        "model": profile["model"],
        "model_revision": profile["revision"],
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "backbone_dtype": BACKBONE_DTYPE,
        "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
        "template_hash": renderer.template_hash,
        "projection_hash": renderer.projection_hash,
        "candidate_contract": "tanimoto_identity_exclusion_then_eligible_pool.v1",
        "record_scope": "all_stage07_retrieval_eligible_records",
        "n_prompt_scores": 1,
    }
    _finalize_compact_cache({"cache": cache}, version)
    connection = sqlite3.connect(cache)
    tables = {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"scores", "assignments"} <= tables
    assert not ({"prompt_tasks", "prompt_scores", "prompt_assignments"} & tables)
    assert connection.execute("SELECT transfer_probability FROM scores").fetchone()[0] == 0.75
    connection.close()


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
