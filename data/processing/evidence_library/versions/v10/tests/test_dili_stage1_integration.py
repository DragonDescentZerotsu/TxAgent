from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_source_rows,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import (
    NormalizedSourceProfile,
)
from data.processing.evidence_library.versions.v10 import (
    build_measurement_resolution_mapping as generator,
)
from data.processing.evidence_library.versions.v10 import (
    build_normalized_evidence_library as builder,
)
from data.processing.evidence_library.versions.v10.build_endpoint_unit_profile import (
    SUPPORTED_TASKS,
)
from data.processing.evidence_library.versions.v10.task_registry import (
    TASK_MODULES,
    import_task_module,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    build_measurement_resolution_mapping as dili_generator,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution as measurement_config,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_three_endpoint_256 as combined_config,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_three_endpoint_512 as combined_config_512,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_two_endpoint_512 as combined_config_two_endpoint,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_dgx007_dgx011_512 as combined_config_dgx007_dgx011,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_dgx007_550 as combined_config_dgx007,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_dgx005_8100_550 as combined_config_dgx005,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_dgx005_550 as combined_config_dgx005_50001,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_dgx005_50002_550 as combined_config_dgx005_50002,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution_openrouter_key_two_256 as combined_config_openrouter,
)
from data.processing.evidence_library.versions.v10.tasks.dili import starling_policy
from data.processing.evidence_library.versions.v10.tasks.dili.starling_schema import (
    ROLE_FIELDS,
)
from data.processing.paths import evidence_library_root, task_id


def test_dili_v10_registration_is_stage1_only() -> None:
    assert task_id("DILI") == task_id("dili") == "dili"
    assert evidence_library_root("DILI", "v10").as_posix().endswith(
        "data/evidence_libraries/dili/v10"
    )
    assert TASK_MODULES["dili"] == {
        "build_normalized_starling_evidence_library",
        "starling_measurement_resolution",
        "starling_policy",
    }
    assert "dili" in SUPPORTED_TASKS
    assert import_task_module("dili", "starling_policy").POLICY is starling_policy.POLICY
    assert generator.TaskConfig("dili").module is measurement_config


def test_source_inventory_supports_constant_and_literal_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constant_path = tmp_path / "constant.parquet"
    literal_path = tmp_path / "literal.parquet"
    constant_path.write_bytes(b"constant fixture")
    literal_path.write_bytes(b"literal fixture")
    constant_profile = NormalizedSourceProfile(
        source_id="constant",
        source_name="constant",
        endpoint_constant="human_dili_relation",
        source_path=str(constant_path),
    )
    literal_profile = NormalizedSourceProfile(
        source_id="literal",
        source_name="literal",
        endpoint_field="assay_category",
        literal_text_fields=("assay_category",),
        source_path=str(literal_path),
    )
    frames = {
        constant_path: pd.DataFrame(
            [{"source_row_uid": "sr_00000000000000000000000000000000"}]
        ),
        literal_path: pd.DataFrame(
            [
                {
                    "source_row_uid": "sr_00000000000000000000000000000001",
                    "assay_category": "null",
                }
            ]
        ),
    }
    seen: dict[str, list[str]] = {}

    def inventory(source_id: str, values: list[str], *, strict: bool) -> dict:
        assert strict
        seen[source_id] = values
        return {"endpoints": values}

    policy = SimpleNamespace(
        task_id="fixture",
        dataset_name="fixture",
        smiles_mapping=None,
        source_profiles=lambda _: (constant_profile, literal_profile),
        verify_source_digest=None,
        expected_source_rows={"constant": 1, "literal": 1},
        endpoint_inventory=inventory,
        load_extra_source=None,
    )
    args = SimpleNamespace(
        starling_data_dir=tmp_path,
        max_rows_per_source=None,
        strict_endpoint_inventory=True,
    )
    monkeypatch.setattr(builder, "validate_task_sources", lambda _: {})
    monkeypatch.setattr(
        builder.pd, "read_parquet", lambda path: frames[Path(path)].copy()
    )

    _, _, inventories, _ = builder._prepare_source_inputs(policy, args)

    assert seen == {
        "constant": ["human_dili_relation"],
        "literal": ["null"],
    }
    assert inventories["constant"]["endpoints"] == ["human_dili_relation"]
    assert inventories["literal"]["endpoints"] == ["null"]


def test_six_source_roles_use_only_declared_measurement_columns() -> None:
    expected_roles = {
        "dili_base": ("", "causal_status", ""),
        "dili_v1": ("endpoint_category", "result_value", "result_unit"),
        "dili_v2": ("assay_category", "reported_result", ""),
        "dili_v3": ("assay_category", "result_value", "result_unit"),
        "dili_v4": ("assay_family", "result_value", "result_unit"),
        "dili_v5": ("endpoint_class", "quantitative_value", "quantitative_unit"),
    }
    assert ROLE_FIELDS == expected_roles
    profiles = {
        profile.source_id: profile
        for profile in starling_policy.source_profiles(Path("unused"))
    }
    rules = measurement_config.source_routing_rules()
    assert set(profiles) == set(rules) == set(expected_roles)
    for source_id, (endpoint, measurement, unit) in expected_roles.items():
        profile = profiles[source_id]
        assert (profile.endpoint_field, profile.measurement_field, profile.unit_field) == (
            endpoint,
            measurement,
            unit,
        )
        assert rules[source_id].require_positive_value is False
        assert measurement_config.route_measurement(
            {"source_id": source_id, "measurement_text": "qualitative"}
        ).bucket == "reject"


def test_schema_specific_exact_result_routes_are_conservative() -> None:
    safe = measurement_config.route_measurement(
        {
            "source_id": "dili_v5",
            "measurement_text": "2.4",
            "unit_text": "µM",
            "quantitative_measure_type": "ec50",
            "specific_endpoint_name": "intracellular calcium level",
            "support_text": "The EC50 for the intracellular calcium response was 2.4 µM.",
        }
    )
    assert safe.bucket == "accept"
    assert safe.rule_id == "dili_v5_exact_ec50_pair.v1"
    assert (safe.measurement_text, safe.unit_text) == ("2.4", "µM")

    generic_absolute = measurement_config.route_measurement(
        {
            "source_id": "dili_v5",
            "measurement_text": "2.4",
            "unit_text": "nmol/mg protein",
            "quantitative_measure_type": "absolute_endpoint_value",
            "specific_endpoint_name": "hepatic glutathione content",
            "support_text": "Hepatic glutathione content was 2.4 nmol/mg protein.",
        }
    )
    assert generic_absolute.bucket == "extract"

    inferred_mec_unit = measurement_config.route_measurement(
        {
            "source_id": "dili_v5",
            "measurement_text": "50",
            "unit_text": "µM",
            "quantitative_measure_type": "mec",
            "specific_endpoint_name": "intracellular calcium level",
            "support_text": "The intracellular calcium MEC was 50.",
        }
    )
    assert inferred_mec_unit.bucket == "extract"

    relative = measurement_config.route_measurement(
        {
            "source_id": "dili_v5",
            "measurement_text": "2.4",
            "unit_text": "fold",
            "quantitative_measure_type": "fold_change",
            "specific_endpoint_name": "hepatic glutathione content",
            "support_text": "The treated value increased 2.4-fold over control.",
        }
    )
    assert relative.bucket == "extract"

    incomplete_percent = measurement_config.route_measurement(
        {
            "source_id": "dili_v4",
            "measurement_text": "23.4",
            "unit_text": "percent",
            "endpoint_and_comparator": "apoptotic hepatocytes",
            "support_text": "The treated group contained 23.4 percent apoptotic cells.",
        }
    )
    assert incomplete_percent.bucket == "extract"

    assert measurement_config.route_measurement(
        {
            "source_id": "dili_v2",
            "measurement_text": "IC50 12.5 µM",
            "unit_text": "",
        }
    ).bucket == "extract"


def _clean_dili_rows(
    source_id: str, rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    profile = next(
        profile
        for profile in starling_policy.source_profiles(Path("unused"))
        if profile.source_id == source_id
    )
    cleaned = clean_source_rows(rows, profile, smiles_mapping=None)
    result = starling_policy.POLICY.source_value_cleaner(
        cleaned, SimpleNamespace(starling_data_dir="unused")
    )
    return result.records


def test_v5_typed_potency_route_survives_stage1_cleaning() -> None:
    [cleaned] = _clean_dili_rows(
        "dili_v5",
        [
            {
                "source_row_uid": "sr_00000000000000000000000000000001",
                "extraction_id": "v5-safe-ec50",
                "SMILES": "CCO",
                "support_text": (
                    "The EC50 for the intracellular calcium response was 2.4 µM."
                ),
                "endpoint_class": "calcium_homeostasis",
                "endpoint_name": "intracellular calcium level",
                "assay_and_platform": "fluorescence assay",
                "effect_direction": "increased",
                "quantitative_measure_type": "ec50",
                "quantitative_value": "2.4",
                "quantitative_unit": "µM",
            }
        ],
    )

    assert cleaned["measurement_resolution_route"] == "accept"
    assert cleaned["measurement_resolution_rule_id"] == "dili_v5_exact_ec50_pair.v1"
    assert cleaned["measurement_resolution_exact_measurement"] == "2.4"
    assert cleaned["measurement_resolution_exact_unit"] == "µM"


def test_raw_missing_literal_null_and_off_schema_endpoints_are_preserved() -> None:
    rows = [
        {
            "source_row_uid": f"sr_{index:032x}",
            "extraction_id": str(index),
            "SMILES": "CCO",
            "assay_category": endpoint,
            "result_value": "0",
            "result_unit": "%",
        }
        for index, endpoint in enumerate(
            ("reviewed_value", "unexpected_new_value", None, "   ", "null"), start=1
        )
    ]

    cleaned = _clean_dili_rows("dili_v3", rows)

    assert [row["endpoint_name"] for row in cleaned] == [
        "reviewed_value",
        "unexpected_new_value",
        None,
        None,
        "null",
    ]
    assert [row["canonical_endpoint_name"] for row in cleaned] == [
        "reviewed_value",
        "unexpected_new_value",
        "missing_endpoint",
        "missing_endpoint",
        "null",
    ]
    assert [row["source_row_uid"] for row in cleaned] == [
        row["source_row_uid"] for row in rows
    ]
    inventory = starling_policy.endpoint_inventory(
        "dili_v3", ["reviewed_value", "unexpected_new_value", "", "null"], strict=True
    )
    assert inventory["endpoints"] == ["", "null", "reviewed_value", "unexpected_new_value"]
    assert inventory["endpoint_mapping_applied"] is False
    assert inventory["canonical_endpoints"] == []
    assert inventory["excluded_values"] == []


def test_dili_candidate_generation_rejects_a_persisted_endpoint_alias(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cleaned_record_id": "row-1",
                    "source_row_uid": "sr_00000000000000000000000000000001",
                    "source_id": "dili_v2",
                    "endpoint_name": "raw_assay_category",
                    "canonical_endpoint_name": "mapped_assay_category",
                    "measurement_text": "5",
                    "measurement_resolution_route": "extract",
                    "measurement_resolution_rule_id": None,
                    "measurement_resolution_exact_measurement": None,
                    "measurement_resolution_exact_unit": None,
                    "measurement_resolution_exact_unit_is_canonical": False,
                }
            ]
        ),
        path,
    )

    with pytest.raises(ValueError, match="differs from the raw endpoint"):
        generator.candidate_rows(path, generator.TaskConfig("dili"))


def test_dili_candidate_generation_rejects_a_stale_persisted_route(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cleaned_record_id": "row-1",
                    "source_row_uid": "sr_00000000000000000000000000000001",
                    "source_id": "dili_v3",
                    "endpoint_name": "assay",
                    "canonical_endpoint_name": "assay",
                    "measurement_text": "5",
                    "unit_text": "fold_vs_control",
                    "measurement_resolution_route": "accept",
                    "measurement_resolution_rule_id": "stale_rule.v1",
                    "measurement_resolution_exact_measurement": "5",
                    "measurement_resolution_exact_unit": "fold_vs_control",
                    "measurement_resolution_exact_unit_is_canonical": False,
                }
            ]
        ),
        path,
    )

    with pytest.raises(ValueError, match="differs from the active Stage-1 policy"):
        generator.candidate_rows(path, generator.TaskConfig("dili"))


def test_v5_specific_endpoint_name_cannot_overwrite_endpoint_class() -> None:
    [cleaned] = _clean_dili_rows(
        "dili_v5",
        [
            {
                "source_row_uid": "sr_00000000000000000000000000000001",
                "extraction_id": "v5",
                "SMILES": "CCO",
                "endpoint_class": "reactive_species",
                "endpoint_name": "total ROS formation",
                "quantitative_value": "-1",
                "quantitative_unit": "%",
            }
        ],
    )

    assert cleaned["endpoint_name"] == "reactive_species"
    assert cleaned["specific_endpoint_name"] == "total ROS formation"
    assert cleaned["canonical_endpoint_name"] == "reactive_species"


def test_direct_deepseek_adapter_uses_explicit_contract() -> None:
    request: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs):
            request.update(kwargs)
            message = SimpleNamespace(content='{"rows": []}', reasoning_content=None)
            return SimpleNamespace(
                id="response-1",
                model=measurement_config.DEEPSEEK_MODEL,
                provider=None,
                choices=[SimpleNamespace(message=message)],
                usage=None,
            )

    client = SimpleNamespace(
        base_url=measurement_config.DEEPSEEK_BASE_URL,
        chat=SimpleNamespace(completions=Completions()),
    )
    result = generator.openai_compatible_llm(client)(
        {"system": "system", "user": "user"},
        model=measurement_config.DEEPSEEK_MODEL,
        max_tokens=measurement_config.MAX_COMPLETION_TOKENS,
        temperature=measurement_config.TEMPERATURE,
        reasoning_effort=measurement_config.REASONING_EFFORT,
    )
    args = generator.parse_args(
        [
            "--task",
            "dili",
            "--base-url",
            measurement_config.DEEPSEEK_BASE_URL,
            "--model",
            measurement_config.DEEPSEEK_MODEL,
            "--provider",
            measurement_config.DEEPSEEK_PROVIDER,
            "--max-completion-tokens",
            str(measurement_config.MAX_COMPLETION_TOKENS),
        ]
    )

    assert args.base_url == measurement_config.DEEPSEEK_BASE_URL
    assert args.model == measurement_config.DEEPSEEK_MODEL
    assert args.api_key_env is None
    assert args.provider == measurement_config.DEEPSEEK_PROVIDER
    assert args.max_completion_tokens == measurement_config.MAX_COMPLETION_TOKENS
    assert request["reasoning_effort"] == "low"
    assert "extra_body" not in request
    assert request["max_tokens"] == measurement_config.MAX_COMPLETION_TOKENS
    assert request["temperature"] == measurement_config.TEMPERATURE
    assert result["api_metadata"] == {
        "api_response_id": "response-1",
        "returned_model": measurement_config.DEEPSEEK_MODEL,
        "served_provider": None,
        "requested_provider": None,
    }


def test_dili_generator_fixes_provider_pool_defaults() -> None:
    args = generator.parse_args(dili_generator.fixed_argv(["--inventory-only"]))

    assert args.task == "dili"
    assert args.base_url is None
    assert args.model == measurement_config.DEEPSEEK_MODEL
    assert args.api_key_env is None
    assert args.provider is None
    assert args.provider_pool_config == measurement_config.PROVIDER_POOL_CONFIG
    assert args.parallelism == measurement_config.PROVIDER_POOL_PARALLELISM
    assert args.max_completion_tokens == measurement_config.MAX_COMPLETION_TOKENS
    assert args.no_token_ledger is True
    assert args.require_complete is True
    assert measurement_config.prompt_manifest()["generation_temperature"] == 0.0

    gold_args = generator.parse_args(dili_generator.fixed_argv(["--gold-replay"]))
    assert gold_args.gold_fixture == measurement_config.DEFAULT_GOLD_FIXTURE

    with pytest.raises(SystemExit, match="remove these override flags"):
        dili_generator.fixed_argv(["--base-url", "https://openrouter.ai/api/v1"])

    subset_args = generator.parse_args(dili_generator.fixed_argv(["--gold-replay"]))
    with pytest.raises(SystemExit, match="subset_artifacts"):
        measurement_config.validate_generation_args(subset_args)

    bypass_argv = [
        "--task",
        "dili",
        "--base-url",
        "https://openrouter.ai/api/v1",
        "--provider",
        "openrouter",
        "--api-key-env",
        "OPEN_ROUTER_KEY",
        "--model",
        "wrong/model",
        "--no-token-ledger",
        "--require-complete",
    ]
    bypass = generator.parse_args(bypass_argv)
    with pytest.raises(SystemExit, match="generation contract mismatch"):
        measurement_config.validate_generation_args(bypass)
    with pytest.raises(SystemExit, match="generation contract mismatch"):
        generator.main(bypass_argv)


@pytest.mark.parametrize(
    "overlay",
    [
        combined_config,
        combined_config_512,
        combined_config_two_endpoint,
        combined_config_dgx007_dgx011,
        combined_config_dgx007,
        combined_config_dgx005,
        combined_config_dgx005_50001,
        combined_config_dgx005_50002,
        combined_config_openrouter,
    ],
)
def test_dili_combined_queue_overlay_preserves_prompt_contract(overlay: object) -> None:
    assert overlay.prompt_manifest() == measurement_config.prompt_manifest()
    args = generator.parse_args(
        [
            "--task",
            "dili",
            "--config-module",
            overlay.__name__,
            "--provider-pool-config",
            str(overlay.PROVIDER_POOL_CONFIG),
            "--parallelism",
            str(overlay.PROVIDER_POOL_PARALLELISM),
            "--model",
            overlay.DEEPSEEK_MODEL,
            "--max-completion-tokens",
            str(overlay.MAX_COMPLETION_TOKENS),
            "--retry-max-completion-tokens",
            "65536",
            "--no-token-ledger",
            "--require-complete",
        ]
    )
    overlay.validate_generation_args(args)


@pytest.mark.parametrize(
    ("module", "expected_temperature"),
    [
        (SimpleNamespace(), 1.0),
        (measurement_config, measurement_config.TEMPERATURE),
    ],
)
def test_submission_uses_task_temperature_or_historical_default(
    module: object, expected_temperature: float
) -> None:
    captured: dict[str, object] = {}

    class Executor:
        def submit(self, function, *args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(function=function, args=args)

    class Cache:
        def submit(self, *args, **kwargs) -> None:
            pass

        def _append(self, event) -> None:
            pass

    batch = generator.RequestBatch(
        request_id="temperature-contract",
        source_id="source",
        rows=({"id": "row-1", "source_id": "source"},),
        prompt="prompt",
        max_completion_tokens=32,
    )
    generator._submit_request(
        Executor(),
        batch,
        Cache(),
        None,
        "",
        "model",
        "https://example.test/v1",
        SimpleNamespace(task_id="fixture", module=module),
        object(),
    )

    assert captured["temperature"] == expected_temperature


def test_explicit_temperature_changes_request_identity_without_changing_legacy_ids() -> None:
    groups = [("endpoint", [{"id": "row-1"}])]
    common = {
        "task_id": "fixture",
        "PROMPT_VERSION": "prompt.v1",
    }

    def digest(module: object) -> str:
        return generator._batch_digest(
            groups,
            SimpleNamespace(**common, module=module),
            "source",
            "rendered prompt",
            "profile-digest",
            "model",
            32,
        )

    legacy_before = digest(SimpleNamespace())
    legacy_after = digest(SimpleNamespace())
    explicit_one = digest(SimpleNamespace(TEMPERATURE=1.0))
    explicit_zero = digest(SimpleNamespace(TEMPERATURE=0.0))

    assert legacy_before == legacy_after == "e70f290c8497a1f64a7e9b3d"
    assert len({legacy_before, explicit_one, explicit_zero}) == 3


def test_dili_gold_replay_requires_prompt_freeze_receipt(tmp_path: Path) -> None:
    lines = measurement_config.DEFAULT_GOLD_FIXTURE.read_text(encoding="utf-8").splitlines()
    manifest = json.loads(lines[0])
    manifest.pop("prompt_at_label_freeze")
    stale_gold = tmp_path / "dili-stale-gold.jsonl"
    stale_gold.write_text(
        "\n".join([json.dumps(manifest), *lines[1:]]) + "\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="gold freeze provenance mismatch"):
        measurement_config._validate_frozen_gold(stale_gold)


def _guarded_assignment(
    row: dict[str, str], measurement: str, unit: str
) -> dict[str, object]:
    raw = json.dumps(
        {
            "id": "r1",
            "status": "ok",
            "measurements": [{"measurement": measurement, "unit": unit}],
        }
    )
    return measurement_config.guard_model_assignment(
        row,
        {
            "status": "ok",
            "measurements_json": json.dumps(
                [{"measurement": measurement, "unit": unit}]
            ),
            "quantity_count": 1,
            "assignment_method": "model_single_pass",
            "rejected_response_json": None,
            "raw_response_json": raw,
        },
    )


@pytest.mark.parametrize(
    ("row", "measurement", "unit", "expected_status"),
    [
        (
            {"source_id": "dili_v1", "measurement_text": "IC50: 56.7", "unit_text": "umol_per_l"},
            "56.7",
            "umol_per_l",
            "ok",
        ),
        (
            {"source_id": "dili_v1", "measurement_text": "104.1 ± 1.6", "unit_text": "percent_of_vehicle"},
            "104.1",
            "percent_of_vehicle",
            "unsure",
        ),
        (
            {"source_id": "dili_v2", "measurement_text": "360 ± 47 pmol/mg DNA", "unit_text": ""},
            "360",
            "pmol/mg DNA",
            "ok",
        ),
        (
            {"source_id": "dili_v2", "measurement_text": "signal intensity 45 (substrate)", "unit_text": ""},
            "45",
            "substrate",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "0.066",
                "unit_text": "",
                "specific_endpoint_name": (
                    "hepatic oxidative stress index (OSI, TOS/TAC ratio)"
                ),
            },
            "0.066",
            "hepatic oxidative stress index (OSI, TOS/TAC ratio)",
            "ok",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "0.78",
                "unit_text": "",
                "specific_endpoint_name": "lysosome colocalization (Pearson's coefficient)",
            },
            "0.78",
            "Pearson's coefficient",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v4",
                "measurement_text": "183",
                "unit_text": "",
                "endpoint_and_comparator": "183 differentially expressed genes",
            },
            "183",
            "differentially expressed genes",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "80",
                "unit_text": "",
                "assay_detail": "viable cells counted after treatment",
            },
            "80",
            "% of viable cells",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "50 µM treatment inhibited respiration",
                "unit_text": "",
            },
            "50",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": (
                    "ATP depletion evident at the intermediate concentration (50 µM)"
                ),
                "unit_text": "",
            },
            "50",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": (
                    "sulfaphenazole (1 µM) modulated diclofenac-mediated "
                    "HO-1 mRNA induction"
                ),
                "unit_text": "",
            },
            "1",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": (
                    "10 µg/ml: no morphological differences; "
                    "25 µg/ml: cytolytic change"
                ),
                "unit_text": "",
            },
            "10",
            "µg/ml",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": (
                    "no significant change in TG accumulation through 0.6 mM"
                ),
                "unit_text": "",
            },
            "0.6",
            "mM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": (
                    "no obvious cytotoxic effect through 500 µM versus control"
                ),
                "unit_text": "",
            },
            "500",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "no inhibition by DOX (the inhibition was for Enoximone "
                    "0.02 mM vs control)"
                ),
                "unit_text": "",
            },
            "0.02",
            "mM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "No significant effect on malonyl-CoA I50 in inverted "
                    "vesicles (268 uM vs 199 uM control)"
                ),
                "unit_text": "",
            },
            "268",
            "uM",
            "ok",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "no induction of apoptosis at concentrations through 30 µM"
                ),
                "unit_text": "",
            },
            "30",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "N-Hcy-hemoglobin present at approximately 12.7 uM in blood"
                ),
                "unit_text": "",
            },
            "12.7",
            "uM",
            "ok",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": "1.33 ± 0.25",
                "unit_text": "ratio",
                "assay_and_readout": "AST/ALT ratio",
            },
            "1.33",
            "ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": "24 ± 3.0",
                "unit_text": "%",
                "assay_and_readout": "LDH release percent cytotoxicity",
            },
            "24",
            "%",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "ROS increased from 39 to 119 nmol/ml",
                "unit_text": "",
                "assay_detail": "Cellular ROS concentration in nmol/ml",
            },
            "119",
            "nmol/ml",
            "ok",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": (
                    "dramatically decreased net ATP levels when INH combined with "
                    "piericidin A (30 nM, nontoxic alone)"
                ),
                "unit_text": "",
                "assay_and_readout": "Net cellular ATP levels",
            },
            "30",
            "nM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": "dose-dependent stimulation; 1 mM most effective",
                "unit_text": "",
                "assay_and_readout": "thymidine incorporation into DNA",
            },
            "1",
            "mM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "4 mM IA decreased mitochondrial ROS production and mitochondrial "
                    "ability to accumulate calcium"
                ),
                "unit_text": "",
                "assay_detail": "mitochondrial calcium and ROS outcomes",
            },
            "4",
            "mM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "ATP IC50 82.39 uM in glucose medium vs 0.005 uM in "
                    "galactose medium"
                ),
                "unit_text": "",
                "assay_detail": "ATP IC50 in glucose versus galactose medium",
            },
            "82.39",
            "uM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "360 ± 47 pmol/mg DNA",
                "unit_text": "",
            },
            "360",
            "pmol/mg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 ± 1 U/L",
                "unit_text": "",
            },
            "1",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "IC50: 1e-5 mol/L",
                "unit_text": "",
            },
            "5",
            "mol/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "1.7",
                "unit_text": "",
                "specific_endpoint_name": "treated/control ratio",
                "quantitative_measure_type": "relative_to_control",
            },
            "1.7",
            "treated/control ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "2-fold increase",
                "unit_text": "",
            },
            "2",
            "fold_change",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity changed by 20 percent",
                "unit_text": "",
            },
            "20",
            "percent_change",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "control was 5 U/L",
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity was 5 U/L at 24 h",
                "unit_text": "",
            },
            "5",
            "U/L",
            "ok",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 U/L or less",
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "viability decreased at 50 µM",
                "unit_text": "",
            },
            "50",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/kg was administered; ALT increased",
                "unit_text": "",
            },
            "5",
            "mg/kg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v1",
                "measurement_text": "5 mg/kg administered to rats",
                "unit_text": "mg_per_kg",
            },
            "5",
            "mg_per_kg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 µg/mL compound caused cell death",
                "unit_text": "",
            },
            "5",
            "µg/mL",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg administered before the assay",
                "unit_text": "",
            },
            "5",
            "mg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "50 µM was used; viability decreased",
                "unit_text": "",
            },
            "50",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "10 mg/kg/day was administered; ALT increased",
                "unit_text": "",
            },
            "10",
            "mg/kg/day",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "viability fell following exposure at a concentration of 50 µM"
                ),
                "unit_text": "",
            },
            "50",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "rats received 10 mg/kg; ALT increased",
                "unit_text": "",
            },
            "10",
            "mg/kg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "10 mg/kg was injected; ALT increased",
                "unit_text": "",
            },
            "10",
            "mg/kg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "compound concentration was 50 µM",
                "unit_text": "",
            },
            "50",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "oral dosage: 5 mg/kg/week",
                "unit_text": "",
            },
            "5",
            "mg/kg/week",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "animals consumed 5 mg per kg",
                "unit_text": "",
            },
            "5",
            "mg per kg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "exposure level was 5 mg/kg bw/day",
                "unit_text": "",
            },
            "5",
            "mg/kg bw/day",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "working concentration was 5 mg/m2",
                "unit_text": "",
            },
            "5",
            "mg/m2",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "animals received 5 IU/kg",
                "unit_text": "",
            },
            "5",
            "IU/kg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "lowest effective concentration 0.3 µM",
                "unit_text": "",
            },
            "0.3",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "highest concentration tested 10 µM",
                "unit_text": "",
            },
            "10",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "ATP depletion at lowest significant concentration 1 mM"
                ),
                "unit_text": "",
            },
            "1",
            "mM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "2",
                "unit_text": "",
                "assay_detail": "ATP readout",
                "support_text": (
                    "After 2 mg/kg treatment, ATP decreased qualitatively."
                ),
            },
            "2",
            "mg/kg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "IC50 not determined; 10 µM treatment decreased ATP"
                ),
                "unit_text": "",
            },
            "10",
            "µM",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "baseline 1 mg/L; response increased from 5 to 20 mg/L"
                ),
                "unit_text": "",
            },
            "20",
            "mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "Drug A was 10 mg/L vs vehicle 5 mg/L and Drug B 7 mg/L"
                ),
                "unit_text": "",
            },
            "10",
            "mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "1.7",
                "unit_text": "",
                "specific_endpoint_name": "KO/WT ratio",
                "quantitative_measure_type": "ratio",
            },
            "1.7",
            "KO/WT ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "1.7",
                "unit_text": "",
                "specific_endpoint_name": "mutant/wild-type ratio",
                "quantitative_measure_type": "ratio",
            },
            "1.7",
            "mutant/wild-type ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "0.44 ± 0.26, p = 0.029",
                "unit_text": "",
                "specific_endpoint_name": "complex I + III/CS Mut/WT ratio",
                "quantitative_measure_type": "ratio",
            },
            "0.44",
            "complex I + III/CS Mut/WT ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "KO-to-WT ratio: 1.7",
                "unit_text": "",
                "assay_detail": "KO-to-WT ratio",
            },
            "1.7",
            "KO-to-WT ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "KO:WT ratio: 1.7",
                "unit_text": "",
                "assay_detail": "KO:WT ratio",
            },
            "1.7",
            "KO:WT ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "WT-normalized activity ratio: 1.7",
                "unit_text": "",
                "assay_detail": "WT-normalized activity ratio",
            },
            "1.7",
            "WT-normalized activity ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "1.7 (SCO2 KO)/WT ratio",
                "unit_text": "",
                "assay_detail": "(SCO2 KO)/WT ratio",
            },
            "1.7",
            "(SCO2 KO)/WT ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "SCO2-deficient/WT ratio: 1.7",
                "unit_text": "",
                "assay_detail": "SCO2-deficient/WT ratio",
            },
            "1.7",
            "SCO2-deficient/WT ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "drug-A/drug-B ratio: 1.7",
                "unit_text": "",
                "assay_detail": "drug-A/drug-B ratio",
            },
            "1.7",
            "drug-A/drug-B ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "high-dose/low-dose ratio: 1.7",
                "unit_text": "",
                "assay_detail": "high-dose/low-dose ratio",
            },
            "1.7",
            "high-dose/low-dose ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity ratio relative to WT was 1.7",
                "unit_text": "",
                "assay_detail": "activity ratio",
            },
            "1.7",
            "activity ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity ratio was 2 times control",
                "unit_text": "",
                "assay_detail": "activity ratio",
            },
            "2",
            "activity ratio",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "360 pmol/mg of DNA",
                "unit_text": "",
            },
            "360",
            "pmol/mg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "360 pmol/mg (DNA)",
                "unit_text": "",
            },
            "360",
            "pmol/mg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "2.4 nmol/min/mg (microsomal protein)",
                "unit_text": "",
            },
            "2.4",
            "nmol/min/mg",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L at 24 h",
                "unit_text": "",
            },
            "5",
            "mg/L at 24 h",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "0.066",
                "unit_text": "",
                "specific_endpoint_name": (
                    "hepatic oxidative stress index (OSI, TOS/TAC ratio)"
                ),
            },
            "0.066",
            "stress index",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "phagocytic index K: 3",
                "unit_text": "",
                "specific_endpoint_name": "phagocytic index K",
            },
            "3",
            "phagocytic index",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "phagocytic index PI: 3",
                "unit_text": "",
                "specific_endpoint_name": "phagocytic index PI",
            },
            "3",
            "phagocytic index",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "phagocytic index alpha: 3",
                "unit_text": "",
                "specific_endpoint_name": "phagocytic index alpha",
            },
            "3",
            "phagocytic index",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "5",
                "unit_text": "",
                "specific_endpoint_name": "ALT score or AST score",
            },
            "5",
            "ALT score",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "5",
                "unit_text": "",
                "specific_endpoint_name": "ALT score, AST score",
            },
            "5",
            "ALT score",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "2.4",
                "unit_text": "",
                "specific_endpoint_name": "toxicity score in treated mice",
            },
            "2.4",
            "toxicity score",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "2.4",
                "unit_text": "",
                "specific_endpoint_name": "toxicity score in treated mice",
            },
            "2.4",
            "toxicity score in treated mice",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "5",
                "unit_text": "",
                "specific_endpoint_name": "ALT score and AST score",
            },
            "5",
            "ALT score",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "5",
                "unit_text": "",
                "specific_endpoint_name": "ALT score and AST score",
            },
            "5",
            "ALT score and AST score",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "group A 5 U/L vs group B 2 U/L",
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "Low-dose group 10 U/L vs high-dose group 20 U/L"
                ),
                "unit_text": "",
            },
            "10",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "ALT was 5 U/L vs AST was 2 U/L in the control group"
                ),
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "sham group mean was 10 U/L",
                "unit_text": "",
            },
            "10",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "untreated cohort mean was 10 U/L",
                "unit_text": "",
            },
            "10",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "reference group mean was 10 U/L",
                "unit_text": "",
            },
            "10",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": (
                    "ALT level was 5 U/L vs AST level was 2 U/L in control group"
                ),
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity was 5 U/L or more",
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity was more than 5 U/L",
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity was under 5 U/L",
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "activity was not exceeding 5 U/L",
                "unit_text": "",
            },
            "5",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "values ranged from 5 to 20 U/L",
                "unit_text": "",
            },
            "20",
            "U/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L vs control 10 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L vs control 10 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L and 7 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L and 7 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L (n=8)",
                "unit_text": "",
            },
            "5",
            "mg/L (n=8)",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L; second result 7 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L; second result 7 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L. second result 7 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L. second result 7 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L: 7 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L: 7 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L\nsecond result 7 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L\nsecond result 7 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L | 7 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L | 7 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L - second result 7 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L - second result 7 mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L (second result 7 mg/L)",
                "unit_text": "",
            },
            "5",
            "mg/L (second result 7 mg/L)",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L [95% CI 4-6 mg/L]",
                "unit_text": "",
            },
            "5",
            "mg/L [95% CI 4-6 mg/L]",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L (± 1 mg/L)",
                "unit_text": "",
            },
            "5",
            "mg/L (± 1 mg/L)",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L (p=0.01)",
                "unit_text": "",
            },
            "5",
            "mg/L (p=0.01)",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "5 mg/L in treated cells",
                "unit_text": "",
            },
            "5",
            "mg/L in treated cells",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "placebo value was 10 mg/L",
                "unit_text": "",
            },
            "10",
            "mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "control median, 10 mg/L",
                "unit_text": "",
            },
            "10",
            "mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "10 mg/L, control",
                "unit_text": "",
            },
            "10",
            "mg/L",
            "unsure",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "IC50 was 5 mg/L",
                "unit_text": "",
            },
            "5",
            "mg/L",
            "ok",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "360 pmol/mg of DNA",
                "unit_text": "",
            },
            "360",
            "pmol/mg of DNA",
            "ok",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "360 pmol/mg (DNA)",
                "unit_text": "",
            },
            "360",
            "pmol/mg (DNA)",
            "ok",
        ),
        (
            {
                "source_id": "dili_v2",
                "measurement_text": "Drug A was 10 mg/L vs vehicle 5 mg/L",
                "unit_text": "",
                "assay_detail": "Drug A concentration in the treated sample",
            },
            "10",
            "mg/L",
            "ok",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "GSH/GSSG ratio was 2",
                "unit_text": "",
                "specific_endpoint_name": "GSH/GSSG ratio",
            },
            "2",
            "GSH/GSSG ratio",
            "ok",
        ),
    ],
)
def test_dili_model_assignment_guard_fails_closed(
    row: dict[str, str], measurement: str, unit: str, expected_status: str
) -> None:
    guarded = _guarded_assignment(row, measurement, unit)

    assert guarded["status"] == expected_status
    assert guarded["raw_response_json"]
    assert guarded["rejected_response_json"] is None
    if expected_status == "unsure":
        assert guarded["measurements_json"] == "[]"
        assert guarded["quantity_count"] == 0
        assert guarded["assignment_method"] == "model_guarded_unsure"
        assert guarded["assignment_guard_reason"]
    else:
        assert guarded["assignment_guard_reason"] is None


_DILI_REVIEWED_PREDICTIONS = {
    "0087d816c6d6c2af8793a1af74496da10a6078291aaaf6a1eb40c80bf245c244": (
        "795.7", "% of saline", "relative_value"
    ),
    "00a599607d50081313f5a9375ac645c4e05bb82d2a5b5ea98fbb8b76e078b9ab": (
        "2", "peroxyl radicals scavenged per antioxidant molecule", None
    ),
    "02fcc8b0ddfcb2c8f96514e5099543cc9eedba3213b6e3628265b43c3647b5de": (
        "20", "% viability", "conflicting_focal_statement"
    ),
    "07857e27505810c03a37237a2258063e00f4acdcdf998adefdc496f57924af7a": (
        "90", "% dead by day 3", "unit_contains_time_context"
    ),
    "10390b3dfcf9b91e0c26c2ecc2c7a6197861328f4a5a27c83c6e0fe2611352d3": (
        "0.785",
        "Manders' M1 coefficient (KEAP1-NRF2 colocalization, pSJD mice)",
        "control_or_comparator_not_outcome",
    ),
    "121ea045829eaeedfca05ee9c35e08e93685ab3299e63d76ac0c972b05942beb": (
        "22", "% positive cells", "multiple_or_missing_point_candidates"
    ),
    "1fc7c89fe7ee39a3eb30ff88b7a3261db4b43a631f0ee35cda2c6aaa952b5840": (
        "180", "nmol_glucose_per_10_min_per_2.5e6_cells", "relative_value"
    ),
    "230fba69d126ed0350fee803df426a8d4502cc20d7cd4cc92451d9fbc42cfbfd": (
        "1.07", "RBI", "unsupported_or_uncertain_unit"
    ),
    "277dbd08942f9c83700363246662d94a0f0ad7dec27c0ffd729ed6bea3273519": (
        "20", "% (percent survival)", "conflicting_focal_statement"
    ),
    "29b67bae3de55152c06e4bc297542252a6fea8d6e65ea286e378e9e111eaa9de": (
        "0.44", "% U/mg protein", "unsupported_or_uncertain_unit"
    ),
    "2c3293cb357f8210f026470ba297c84b59704f1ec43dcb08fb4a7eb9a094a09f": (
        "6.2", "mmole/l", "multiple_or_missing_point_candidates"
    ),
    "2cd28bfafb1ecc880f533ef9bb2ba143b8e016d89d85c0b5fae495c54f7c3fa4": (
        "11", "% per 10-hour circulation", "bound_not_point"
    ),
    "2fb1541b59553fae5fec974a3ab1147ed51b4ee207e3ca6b69a380ea1c7645d6": (
        "30.6", "m-equiv/l", "relative_value"
    ),
    "358995376954f405f39ebd0df5422bfd6d863045c00666307b600221c265e613": (
        "0.95", "index (dimensionless)", "unsupported_or_uncertain_unit"
    ),
    "371350c2b0393dada25ebd1d60497caeae1741e115a5f08b3b0189ad01e37505": (
        "88", "% of given amount", "multiple_or_missing_point_candidates"
    ),
    "3e71828497f5013d9d531c6526cb1d126ed23eb91ce6fe8a9801729564d7905c": (
        "18.71", "% activity", "relative_value"
    ),
    "3f65a1f0bdcfee0f6790a6c4459101396c9c82ea18b9cf97387f3681f350fb00": (
        "40", "% per day", "bound_not_point"
    ),
    "40b209d2bfeeb0cffaddc2f0cb02cb647d14ac364438e0b74f00188a9b75349c": (
        "30", "% immunostaining", "unsupported_or_uncertain_unit"
    ),
    "440cbb3deefeff4fdb6ff28eaa8308f5b29e0dde86491d1188905f1af72e0f96": (
        "30", "% of original ATP level", "relative_value"
    ),
    "4c551ffa6c31d919ee4cf503757d7292ef0e07a5b0a59a19a3103bee897c8b70": (
        "27.3", "H2O2 decomposition/min/mg protein", "unsupported_or_uncertain_unit"
    ),
    "563e7a652948c05db5ca6ef079bd357cc9abdd0c58195dba4876f2fb9fdee9cc": (
        "12.4",
        "E-5 (reported dimensionless mole-fraction scale)",
        "multiple_or_missing_point_candidates",
    ),
    "5a67da3c8b711ce0ae8782c68bc238cf157a1f09ee902280be9e788b5cebe9da": (
        "13", "cycles/day", "bound_not_point"
    ),
    "634e68af943bd340f47a684ebfe46cf12f064b0e88098f30e617113968f5e374": (
        "56.7", "% of total", "unsupported_or_uncertain_unit"
    ),
    "696fbd93f706b63bbcfd7206d2e2653eb524d315ef52b38c873375b8cc28e34b": (
        "43.86", "AI value", "unsupported_or_uncertain_unit"
    ),
    "73d4eee076ed791eefa155d449e8aa973cb906b71de33d41e928ef04bccc4265": (
        "48.6", "CAT activity unit", "unsupported_or_uncertain_unit"
    ),
    "7fb5d0beb1db7fdd138775bac99c296a7c7f2f2179beea1ed5ce177fb3cb30fc": (
        "26.42", "CDNB-GSH/min/mg protein", "unsupported_or_uncertain_unit"
    ),
    "8e52fd34522f26caa6929510b61e78656485b85119cfd4040d0e72e9f23584f3": (
        "0.24", "Pearson correlation coefficient", "incomplete_named_metric"
    ),
    "966894dfbe735633316d93ea32e1f9db7001faa26d458a8bbcfcb3f882fa2375": (
        "77", "% of total LC", "bound_not_point"
    ),
    "9afadd1cedcc60894a0a8c9bae9a7e8981f079eebe3e4f2a1e95b1fd6ef4c42e": (
        "1.43", "IC50", "unsupported_or_uncertain_unit"
    ),
    "a4d313ea2e4dc4e3d92439c073630ca60698a102f5cdcaaba10202d093a4af1e": (
        "106.9", "ALT (units not stated)", "unsupported_or_uncertain_unit"
    ),
    "a6e99c5b7d4e24e7ac44a9a64faabf418331e0d9de65e3e6fe23b52c555011d6": (
        "50", "µM/mL", "dose_or_concentration_not_outcome"
    ),
    "ad9e2b462b18ce6ee196250d26f2b138184faf090344c92ee9868fff6118fe19": (
        "91.8", "% quenching", "relative_value"
    ),
    "adcb92d403ddd019f22fdac057a3554cee3dfc3beea9b8f3833f53ce8425c9aa": (
        "1.0", "cells_per_high-power_field", "unsupported_or_uncertain_unit"
    ),
    "b3804e23757d0cf978a47441eed540fdca0595a9f6140c3aef99089641f89ab2": (
        "2.3", "mmol day-1", "bound_not_point"
    ),
    "bc4e763f8dbe079aa8b80d389e2216827894985e49a3727b0c8856ec028b127b": (
        "0.066", "hepatic oxidative stress index (OSI, TOS/TAC ratio)", None
    ),
    "bd0cd31aef3786f1bcba8b1fb3207325d83149dbd651c71da6f9690860a97a19": (
        "0.78", "Pearson's coefficient", "incomplete_named_metric"
    ),
    "bf1c0974a058fa441fe9d9809c732db569ee565123d45d9f3ea2a57eee1a70ab": (
        "13.8", "%/g liver", "unsupported_or_uncertain_unit"
    ),
    "c5a24455bbcc88fd6938fcecf3f1dc8d292dd3c1a4a1da2e97e10a44d6738119": (
        "7", "meq/l", "relative_value"
    ),
    "c33483ae4cb54022ca5f200bf2db5ed0fddf79c8e7f9b6e31ace4ac06fb8c41b": (
        "60", "% viability increase", "relative_value"
    ),
    "c9a8bed94e8e48dccd98669a049dff983a4b0ee495104afa0c764b5103211dc1": (
        "24.26", "GSH-CDNB conjugate/min/mg protein", "unsupported_or_uncertain_unit"
    ),
    "ccecab83d56749022905957cd1b97f9311a7999624d47275450f153b24d3bd9d": (
        "0.5", "Pearson colocalization coefficient R", "incomplete_named_metric"
    ),
    "d33631307f8e98fa9d6c7b30247960245891b5bbf600017a1a1f3d32f41b6192": (
        "1.305", "RI", "unsupported_or_uncertain_unit"
    ),
    "d33ff4832af05b178e09cfc3183bab1667258831712dc31419876a0ff919d1d6": (
        "25", "% DPPH quenching", "relative_value"
    ),
    "d350106149053ab5f1065d612318d8666f924a3aeb919b455829faa78892fb9a": (
        "0.8", "dimensionless DICI", "bound_not_point"
    ),
    "d3fa46de99458c5173844eafaa410067985b59b392daff0186ce9c1e53a1149a": (
        "8.02", "GPX-4 fluorescence intensity (arbitrary)", "unsupported_or_uncertain_unit"
    ),
    "dd372d1c73cfe33981acce891117be82a6fa400ebf0fae7c54c4c7d9633d211d": (
        "73.3", "% specific cytotoxicity", "relative_value"
    ),
    "e13a85eaa39cff4967bcc141a6fb8202fdec8e7b78b432af24b8923ca0d544d2": (
        "0.246", "RBD", "unsupported_or_uncertain_unit"
    ),
    "e9ccf066a5cc0ff25da0d9eb00c69072908cbb53e43ca58b5c6d1d780b31bb0f": (
        "1.76", "H2O2/min/g Hb", "unsupported_or_uncertain_unit"
    ),
    "f9333e01df7f6537df583ee23ebe37c59630e8c42f5a9fe04bb1758f6f2f19a9": (
        "13", "ASA/DHA ratio in leaves", None
    ),
    "f3c8a6646f9bc4d31207985ed35689fc3fa0974c1371a4804897279362753a39": (
        "100",
        "% of stained cells with reaction strength +/++/+++",
        "unsupported_or_uncertain_unit",
    ),
    "f99fcfb7dde27e1b02ab455bcf67fdf7d40480eccf6d722d99f77c1b47ed443a": (
        "0.8", "Pearson's coefficient", "incomplete_named_metric"
    ),
    "fa2bb538f569c41784623c23c17b0f82baec8496dcadd98eb7562329aa4f2d53": (
        "18", "% of LLC-NTCP", "relative_value"
    ),
    "fe69f8e0ad155d04455cb96ed6ae038d360f2324ffd251a05cf654f0fde0766a": (
        "21.9", "% reversal", "relative_value"
    ),
}


def _dili_gold_cases() -> dict[str, dict[str, object]]:
    historical_gold = (
        measurement_config.REPO_ROOT
        / "tests/chembl_tool/common/measurement_resolution_quality/gold/dili.v10.jsonl"
    )
    entries = [
        json.loads(line)
        for line in historical_gold.read_text(
            encoding="utf-8"
        ).splitlines()[1:]
    ]
    return {
        str(case["audit_case_id"]).split(":", 1)[-1]: case
        for case in entries
    }


def test_dili_assignment_guard_reviewed_false_ok_audit() -> None:
    cases = _dili_gold_cases()
    assert set(_DILI_REVIEWED_PREDICTIONS) <= set(cases)

    for record_id, (measurement, unit, expected_reason) in (
        _DILI_REVIEWED_PREDICTIONS.items()
    ):
        case = cases[record_id]
        row = {**case["input"], "source_id": case["source_id"]}
        guarded = _guarded_assignment(row, measurement, unit)
        assert guarded["assignment_guard_reason"] == expected_reason, record_id
        assert guarded["status"] == (
            "ok" if expected_reason is None else "unsure"
        ), record_id


def test_dili_assignment_guard_preserves_reviewed_gold_ok_pairs() -> None:
    reviewed_ok = 0
    for case in _dili_gold_cases().values():
        if case["expected"]["status"] != "ok":
            continue
        row = {**case["input"], "source_id": case["source_id"]}
        for entry in case["expected"]["measurements"]:
            reviewed_ok += 1
            assert (
                measurement_config._source_specific_guard_reason(
                    row,
                    str(entry["measurement"]),
                    str(entry["unit"]),
                )
                is None
            ), case["audit_case_id"]
    assert reviewed_ok == 475


def test_relative_percent_unit_does_not_match_changed_gene_counts() -> None:
    assert measurement_config._RELATIVE_RESULT_UNIT.search("% viability increase")
    assert measurement_config._RELATIVE_RESULT_UNIT.search("% loss of cell viability")
    assert not measurement_config._RELATIVE_RESULT_UNIT.search(
        "percent of changed genes upregulated"
    )
    assert not measurement_config._RELATIVE_RESULT_UNIT.search(
        "percent of genes changed on array"
    )


@pytest.mark.parametrize(
    ("row", "measurement", "unit"),
    [
        (
            {
                "source_id": "dili_v4",
                "measurement_text": "2",
                "unit_text": "% survival",
                "support_text": "Mortality was 12%.",
            },
            "2",
            "% survival",
        ),
        (
            {
                "source_id": "dili_v5",
                "measurement_text": "5",
                "unit_text": "µM/mL",
                "support_text": "The IC50 concentration was 50 µM/mL.",
            },
            "5",
            "µM/mL",
        ),
        (
            {
                "source_id": "dili_v4",
                "measurement_text": "2",
                "unit_text": "% survival",
                "support_text": "Mortality was 1e2%.",
            },
            "2",
            "% survival",
        ),
        (
            {
                "source_id": "dili_v4",
                "measurement_text": "2",
                "unit_text": "% survival",
                "support_text": "Mortality was 1e-2%.",
            },
            "2",
            "% survival",
        ),
        (
            {
                "source_id": "dili_v4",
                "measurement_text": "2",
                "unit_text": "% survival",
                "support_text": "Mortality was 10−2%.",
            },
            "2",
            "% survival",
        ),
    ],
)
def test_dili_guard_v9_numeric_equivalence_respects_token_boundaries(
    row: dict[str, str], measurement: str, unit: str
) -> None:
    assert (
        measurement_config._source_specific_guard_reason(row, measurement, unit)
        is None
    )


@pytest.mark.parametrize(
    ("record_id", "measurement", "unit", "expected_reason"),
    [
        (
            "0d8ae5beead3aaefc1030f4453f40a88cce4dc2f2aa0579b164e4a6f55654352",
            "34",
            "% inactivation",
            "relative_value",
        ),
        (
            "33ff45f96d92a51b4976461bcb576d6171202fe558b0b741dbaa69c6c2f19ccd",
            "0.34",
            "Pearson's correlation coefficient",
            "incomplete_named_metric",
        ),
        (
            "6f9f5057edb9a5d325a90d0e37b47fe7793111a94d767f20d3405d4a9d076b4b",
            "30.8",
            "% superoxide radical scavenged",
            "relative_value",
        ),
        (
            "7bc4dc8e90f2d0e1cd580e427f960fd54f4ac94369e91cecde581813b182fccf",
            "1.2",
            "mmoles/liter",
            "relative_value",
        ),
        (
            "8dcba74aab6f70166b2b9ca0ba5db31e4b843db98b74a427ec151391b026da3f",
            "33",
            "µM",
            "conflicting_focal_statement",
        ),
        (
            "faad663b60cc81b494284937aff7f159e7f5131fcfd1d17db3c7bab27e48a2dd",
            "1",
            "1e-6 mol/l",
            "bound_not_point",
        ),
        (
            "00a599607d50081313f5a9375ac645c4e05bb82d2a5b5ea98fbb8b76e078b9ab",
            "2",
            "stoichiometric factor n (peroxyl radicals scavenged per antioxidant molecule)",
            None,
        ),
    ],
)
def test_dili_assignment_guard_v9_fresh_replay_regressions(
    record_id: str,
    measurement: str,
    unit: str,
    expected_reason: str | None,
) -> None:
    case = _dili_gold_cases()[record_id]
    row = {**case["input"], "source_id": case["source_id"]}
    guarded = _guarded_assignment(row, measurement, unit)

    assert guarded["assignment_guard_reason"] == expected_reason
    assert guarded["status"] == ("ok" if expected_reason is None else "unsure")


@pytest.mark.parametrize(
    (
        "candidate_id",
        "source_row_uid",
        "source_id",
        "measurement_text",
        "basis_field",
        "basis",
        "measurement",
        "unit",
        "expected_status",
    ),
    [
        (
            "2cc6fd24544054e452c9d8389fbed9f4c214d27cda95a979f15f460ce337b6da",
            "sr_4d47d5a0127442a28c525e69f04e38e3",
            "dili_v2",
            "ATP 70.9 ng/ml (L-carnitine-balanced) vs. 12.8 ng/ml (NaCl-balanced); choline chloride not significantly different from control",
            "assay_detail",
            "Cellular ATP content, ng/ml after storage",
            "70.9",
            "ng/ml",
            "ok",
        ),
        (
            "2320a3667aa1647bf5c72a9c765491b6b9614e38ecb542fb756ec8dd99b4a280",
            "sr_2536cde2a1f541acb9b32513369b7f44",
            "dili_v2",
            "max D-SG concentration 1 nM after 4 min",
            "assay_detail",
            "D-SG (diclofenac-S-acyl-glutathione thioester) formation measured by LC-MS/MS; maximum D-SG concentration reached",
            "1",
            "nM",
            "ok",
        ),
        (
            "c7bc4c649794ad00c80b970d1dc5c433ecaa42253dd8afad931a926bc9df38ac",
            "sr_d7768c4213b64d059eb512987cca54fc",
            "dili_v2",
            "max D-1-O-G concentration 14.6 uM",
            "assay_detail",
            "Formation of the reactive acyl glucuronide D-1-O-G measured by LC-MS/MS; maximum concentration reached",
            "14.6",
            "uM",
            "ok",
        ),
        (
            "63e728279ca6aa8ed75fbb9838de67a40b8bdeabe02f28dfb0fac6d10198ae66",
            "sr_934c9a62035e4863af157dc0d5f35554",
            "dili_v2",
            "N-Hcy-hemoglobin present at approximately 12.7 uM in human blood, somewhat exceeding plasma tHcy",
            "assay_detail",
            "Steady-state N-Hcy-hemoglobin adduct concentration in human blood",
            "12.7",
            "uM",
            "ok",
        ),
        (
            "0efaf2348913e813518e84bf33e683d275b751e65266ba88e7db23d83f848424",
            "sr_86058ecd2d914d0c803c80095b541173",
            "dili_v2",
            "inhibition of sorbate oxidation by dinitrophenol: sorbate oxidized 0.2 uM at 10 min versus 1.2 uM without dinitrophenol; inhibition not overcome by phosphocreatine system or activating factors",
            "assay_detail",
            "Inhibition of oxidation of sorbate in washed liver particles; sorbate oxidized (uM) measured at 10 min",
            "0.2",
            "uM",
            "ok",
        ),
        (
            "2455d9192c021dfd1935c2cc24616f4b9121916be8937b181db83e79fcf2f945",
            "sr_c1d3ee631c744c088f0822de6fc570da",
            "dili_v1",
            "apoptotic morphology observed; mean diameter 20 µm",
            "assay_and_readout",
            "light microscopy—morphology (cell shrinkage, membrane blebbing, apoptotic bodies, cell diameter)",
            "20",
            "µm",
            "ok",
        ),
        (
            "78e13fda60303c8857870096e7999d10a7b170eb81cf1aa8b7891f425c878942",
            "sr_1011ff9640ed476595d2e940850cbe50",
            "dili_v2",
            "dec-2-ynal reacted with GSH, forming an absorbance peak at 303 nm (GSH adduct formed)",
            "assay_detail",
            "GSH conjugation monitored as new UV absorbance at 303 nm",
            "303",
            "nm",
            "ok",
        ),
        (
            "53c57c0a04de9b97f3608aea4c45946f03287686e71ac6c6de130bce314399b8",
            "sr_67620f771e1844619536d37ed8570713",
            "dili_v1",
            "no prevention of cell death at ~1 µM (caspases blocked); death prevention only at much higher concentrations",
            "assay_and_readout",
            "cytotoxicity/cell death assay—prevention of DR-agonist-induced cell death",
            "1",
            "µM",
            "unsure",
        ),
        (
            "a8fa2e0a2faec0b178c63e05228dcd59892d72f13bd27ebacc61084948966291",
            "sr_e5cbd0e6489f4daa83de3cfb67bda1d1",
            "dili_v1",
            "ER stress induced in hepatocytes at the high DCOIT concentration (300 µg/L); no qualitative detail for lower concentrations given in abstract",
            "assay_and_readout",
            "measurement of ER stress level in hepatocytes",
            "300",
            "µg/L",
            "unsure",
        ),
        (
            "4a1f835646cf9da5dc4ea8881939ce14771749095a32cfe56f1086a24b88916e",
            "sr_0b59af047e11426aa183b9e14175a7ed",
            "dili_v2",
            "Acetate detected in cysticerci only in initial-stage cysticerci treated with ABZSO 13 µM; mostly detected in culture medium rather than within cysticerci",
            "assay_detail",
            "Quantification of acetate as a fatty-acid-oxidation readout in cysticerci and culture medium",
            "13",
            "µM",
            "unsure",
        ),
        (
            "019ea1659ec56f1be81e871c3a1163b7c4bbcceed1cef8b16e5cf401bd4163e8",
            "sr_8200eb471f0e41c1bfa479d6420db30f",
            "dili_v2",
            "no change in MMP (majority of eugenol-treated cells showed emission at ~590 nm)",
            "assay_detail",
            "JC-1 flow cytometry (488 nm excitation, 530/30 and 585/42 nm emission)",
            "590",
            "nm",
            "unsure",
        ),
        (
            "138cbe673abd53f700709977ca140e663b037e05bcc9a725792d559e31b9b1e8",
            "sr_0e524a4702ba4d2495bab473ac3ffaaf",
            "dili_v2",
            "individual complex II and complex III activities were not affected, even at highest tissue propofol concentration of 180 µM",
            "assay_detail",
            "Individual complex II and complex III catalytic activities",
            "180",
            "µM",
            "unsure",
        ),
        (
            "6c28ab038eb57077b17ef4459d86f4c13694923fd1ab388000ba2b2b8fab0991",
            "sr_d8ad20d7c04640cebc7eb7709a766152",
            "dili_v1",
            "no complete cell killing observed; minimum concentration for complete killing exceeded 400.0 µM",
            "assay_and_readout",
            "cytotoxicity assessment—minimum concentration providing complete cell killing",
            "400.0",
            "µM",
            "unsure",
        ),
        (
            "21f2627cb12bb3d54f49b092afd167e3e4d19241a1d1b9c22d5566cd78bbf6ed",
            "sr_024f41ec184d4d37971002022f4b6abb",
            "dili_v2",
            "rate of ketogenesis from octanoate not affected by MCHP 0.05 mM",
            "assay_detail",
            "Ketogenesis from octanoate measured as rate of ketone-body production",
            "0.05",
            "mM",
            "unsure",
        ),
        (
            "e83f5570bd7f4fd464af1d07fe91eb4d801eb617b2ab64a69c75c08d236db66c",
            "sr_349a67b014594077a565e1aedb01cc37",
            "dili_v2",
            "induces mitochondrial permeability transition when dye concentration exceeds 20 µM",
            "assay_detail",
            "Mitochondrial permeability transition as a function of dye concentration",
            "20",
            "µM",
            "unsure",
        ),
        (
            "bb67cddd57f063e7afa1fc73909ff1081ca3c4e2ec5ca7e7e4537403aac57b85",
            "sr_4f784472fe0d48509b97f1f991677970",
            "dili_v2",
            "Abnormal Arrhenius profile of liver mitochondrial succinoxidase activity partly corrected 12 h after 3 µg thyroxine/g",
            "assay_detail",
            "Arrhenius profile of liver mitochondrial succinoxidase activity after in-vivo thyroxine treatment",
            "3",
            "µg thyroxine/g",
            "unsure",
        ),
        (
            "e708b770a5ab3d2c32bd7fc9b53e6d80b3774571547f31855b03dbf1db888d43",
            "sr_327ecb756da7461e8c8179c9ceb79b5b",
            "dili_v2",
            "ATP concentration decreased by about 15 ng/µL vs control",
            "assay_detail",
            "ATP concentration change in ng/µL",
            "15",
            "ng/µL",
            "unsure",
        ),
        (
            "90492c126c4c3a636da5250b8d758141509e890437bc48d86dea87f07c051228",
            "sr_8a66da0ab589473dbe3692848bb64f5d",
            "dili_v1",
            "further significant decrease in GSH content within 30 min at both concentrations; cells initially contained 0.7 µg GSH/10^6 cells",
            "assay_and_readout",
            "cellular glutathione content—further GSH depletion in pre-depleted cells",
            "0.7",
            "µg GSH/10^6 cells",
            "unsure",
        ),
    ],
    ids=lambda value: value[:12] if isinstance(value, str) and len(value) == 64 else None,
)
def test_dili_reviewed_concentration_boundaries(
    candidate_id: str,
    source_row_uid: str,
    source_id: str,
    measurement_text: str,
    basis_field: str,
    basis: str,
    measurement: str,
    unit: str,
    expected_status: str,
) -> None:
    row = {
        "id": candidate_id,
        "source_row_uid": source_row_uid,
        "source_id": source_id,
        "measurement_text": measurement_text,
        basis_field: basis,
    }

    guarded = _guarded_assignment(row, measurement, unit)

    assert guarded["status"] == expected_status
    assert bool(guarded["assignment_guard_reason"]) is (expected_status == "unsure")


def test_dili_retry_returns_validation_feedback_to_the_model() -> None:
    batch = generator.RequestBatch(
        request_id="request-1",
        source_id="dili_v5",
        rows=({"id": "row-1", "source_id": "dili_v5"},),
        prompt="fixture prompt",
        max_completion_tokens=256,
    )
    calls: list[dict] = []

    def fake_llm(messages, **kwargs):
        payload = json.loads(messages["user"])
        calls.append(payload)
        response = (
            {
                "id": "r1",
                "status": "ok",
                "measurements": [{"measurement": "2", "unit": ""}],
            }
            if len(calls) == 1
            else {"id": "r1", "status": "unsure", "measurements": []}
        )
        return {"content": json.dumps({"rows": [response]})}

    rows, _, status = generator.query_batch(
        batch,
        llm=fake_llm,
        model=measurement_config.DEEPSEEK_MODEL,
        task="dili",
        max_measurements=1,
        retry_validation_feedback=True,
        retry_semantic_errors=True,
    )

    assert status == "valid_after_structural_retry"
    assert rows[0]["status"] == "unsure"
    assert calls[1]["validation_feedback"] == [
        {"id": "r1", "error": "entry_without_a_unit"}
    ]


def test_dili_mapping_provenance_accepts_reviewed_provider_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path = tmp_path / "records.parquet"
    profile_path = tmp_path / "endpoint_unit_profile.json"
    mapping_path = tmp_path / "measurement_resolution.parquet"
    stage1_row = {
        "source_id": "dili_v3",
        "endpoint_name": "off_schema_endpoint",
            "measurement_text": "1",
            "measurement_resolution_route": "extract",
            "measurement_resolution_rule_id": None,
            "measurement_resolution_exact_measurement": None,
            "measurement_resolution_exact_unit": None,
            "measurement_resolution_exact_unit_is_canonical": False,
        }
    for source_id in measurement_config.SOURCE_IDS:
        for field in measurement_config.prompt_row_fields(source_id):
            stage1_row.setdefault(field, None)
    provider_contracts = [
        *(
            (base_url, provider_name, measurement_config.DEEPSEEK_MODEL, "")
            for base_url, provider_name in zip(
                measurement_config.PROVIDER_POOL_BASE_URLS,
                measurement_config.PROVIDER_POOL_NAMES,
                strict=True,
            )
        ),
        (
            "http://dgx005:50002/v1",
            "dgx005_50002",
            measurement_config.DEEPSEEK_MODEL,
            "",
        ),
        (
            "http://dgx005:8100/v1",
            "dgx005_8100",
            "deepseek-v4-flash",
            "",
        ),
        (
            "http://dgx007:50001/v1",
            "dgx007_50001",
            measurement_config.DEEPSEEK_MODEL,
            "",
        ),
        (
            "https://openrouter.ai/api/v1",
            "OpenInference",
            "deepseek/deepseek-v4-flash-0731",
            "OPEN_ROUTER_KEY_TWO",
        ),
    ]
    stage1_rows = []
    for index in range(len(provider_contracts)):
        row = dict(stage1_row)
        row["cleaned_record_id"] = f"row-{index + 1}"
        row["source_row_uid"] = f"sr_{index + 1:032d}"
        stage1_rows.append(row)
    accepted_stage1_row = {field: None for field in stage1_rows[0]}
    accepted_stage1_row.update(
        {
            "cleaned_record_id": "row-accepted",
            "source_id": "dili_v5",
            "source_row_uid": "sr_00000000000000000000000000000002",
            "endpoint_name": "calcium_homeostasis",
            "measurement_text": "2.4",
            "unit_text": "µM",
            "support_text": (
                "The EC50 for the intracellular calcium response was 2.4 µM."
            ),
            "specific_endpoint_name": "intracellular calcium level",
            "quantitative_measure_type": "ec50",
            "assay_and_platform": "fluorescence assay",
            "effect_direction": "increased",
            "measurement_resolution_route": "accept",
            "measurement_resolution_rule_id": "dili_v5_exact_ec50_pair.v1",
            "measurement_resolution_exact_measurement": "2.4",
            "measurement_resolution_exact_unit": "µM",
            "measurement_resolution_exact_unit_is_canonical": False,
        }
    )
    pq.write_table(
        pa.Table.from_pylist([*stage1_rows, accepted_stage1_row]),
        records_path,
    )
    profile_path.write_text("{}\n", encoding="utf-8")
    assignments = {}
    provenance = {}
    events = []
    candidates = []
    for index, (base_url, provider_name, model, credential_env) in enumerate(
        provider_contracts, start=1
    ):
        record_id = f"row-{index}"
        assignments[record_id] = {
            "cleaned_record_id": record_id,
            "source_id": "dili_v3",
            "status": "unsure",
            "measurements_json": "[]",
            "quantity_count": 0,
            "assignment_method": "model_single_pass",
            "rejected_response_json": None,
            "raw_response_json": '{"status":"unsure","measurements":[]}',
            "api_response_id": f"response-{index}",
            "returned_model": model,
            "served_provider": provider_name,
            "requested_provider": None,
        }
        provenance[record_id] = {
            "inference_model": model,
            "inference_base_url": base_url,
            # Simulate a later local resume overwriting paid-provider provenance.
            "inference_credential_env": "",
        }
        events.append(
            {
                "status": "api_response",
                "api_response_id": f"response-{index}",
                "returned_model": model,
                "base_url": base_url,
                "credential_env": credential_env,
                "response": {"usage": None},
            }
        )
        candidates.append(
            {
                "id": record_id,
                "source_row_uid": f"sr_{index:032d}",
                "source_id": "dili_v3",
                "canonical_endpoint_name": "off_schema_endpoint",
            }
        )
    cache = SimpleNamespace(
        assignments=assignments,
        attempted=set(assignments),
        provenance=provenance,
        events=events,
    )
    config = generator.TaskConfig("dili")
    monkeypatch.setattr(measurement_config, "DEFAULT_CLEANED_RECORDS", records_path)

    generator.materialize(
        candidates,
        cache,
        config,
        mapping_path=mapping_path,
        records_path=records_path,
        profile_path=profile_path,
        model=measurement_config.DEEPSEEK_MODEL,
        api_base_url="mixed",
        max_completion_tokens=measurement_config.MAX_COMPLETION_TOKENS,
        reasoning_mode=measurement_config.REASONING_EFFORT,
    )

    measurement_config.validate_mapping_provenance(mapping_path)
    rows = pq.read_table(mapping_path).to_pylist()
    assert {row["inference_base_url"] for row in rows} == set(
        base_url for base_url, _, _, _ in provider_contracts
    )
    manifest_path = mapping_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["inference"]["temperature"] == measurement_config.TEMPERATURE
    assert manifest["assignment_guard"] == {
        "version": measurement_config.ASSIGNMENT_GUARD_VERSION,
        "guarded_rows": 0,
        "reason_counts": {},
    }
    manifest["api_base_url"] = "https://openrouter.ai/api/v1"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="measurement mapping provenance mismatch"):
        measurement_config.validate_mapping_provenance(mapping_path)

    manifest["api_base_url"] = "mixed"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    table = pq.read_table(mapping_path).to_pylist()
    table[0]["requested_provider"] = "baidu/fp8"
    pq.write_table(pa.Table.from_pylist(table), mapping_path)
    manifest["mapping_sha256"] = measurement_config.file_sha256(mapping_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row_provider_provenance"):
        measurement_config.validate_mapping_provenance(mapping_path)

    table[0]["requested_provider"] = None
    pq.write_table(pa.Table.from_pylist(table), mapping_path)
    manifest["mapping_sha256"] = measurement_config.file_sha256(mapping_path)
    table[0]["source_id"] = "dili_v4"
    pq.write_table(pa.Table.from_pylist(table), mapping_path)
    manifest["mapping_sha256"] = measurement_config.file_sha256(mapping_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_identity"):
        measurement_config.validate_mapping_provenance(mapping_path)

    table[0]["source_id"] = "dili_v3"
    pq.write_table(pa.Table.from_pylist(table), mapping_path)
    manifest["mapping_sha256"] = measurement_config.file_sha256(mapping_path)
    manifest["assignment_guard"]["version"] = "wrong_guard"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="assignment_guard"):
        measurement_config.validate_mapping_provenance(mapping_path)


def test_failed_dili_validation_preserves_the_canonical_mapping(tmp_path: Path) -> None:
    canonical = tmp_path / "measurement_resolution.parquet"
    canonical_manifest = canonical.with_suffix(".manifest.json")
    pending = tmp_path / "measurement_resolution.pending.parquet"
    pending_manifest = pending.with_suffix(".manifest.json")
    canonical.write_bytes(b"prior valid mapping")
    canonical_manifest.write_bytes(b"prior valid manifest")
    pending.write_bytes(b"invalid pending mapping")
    pending_manifest.write_bytes(b"invalid pending manifest")

    def reject(*args, **kwargs):
        raise ValueError("invalid returned model")

    with pytest.raises(ValueError, match="invalid returned model"):
        generator.promote_validated_mapping(
            pending_path=pending,
            final_path=canonical,
            manifest={},
            expected_record_ids={"row-1"},
            validator=reject,
        )

    assert canonical.read_bytes() == b"prior valid mapping"
    assert canonical_manifest.read_bytes() == b"prior valid manifest"
    assert not pending.exists()
    assert not pending_manifest.exists()


_DILI_GUARD_V6_REVIEWED_CASES = ({'row': {'id': 'b6d8a607d4c194016481f599056d3e7e316c08af5e8a1a621bd771bd5b5749cc',
          'source_row_uid': 'sr_027eaf9bdf774c3498e97935a0e44113',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (Srxn1-GFP %GFP positive 2 m, preceding the CHOP-GFP '
                              'response; also enhanced TNFα-induced ICAM1-GFP)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (Srxn1-GFP oxidative-stress reporter) '
                               'versus time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': 'e463820d8f61e4d9301b809a150af80e1cf897f612e0f0ce29b690cc945a3a85',
          'source_row_uid': 'sr_08c485b67d004be3aca5b69d85754074',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (CHOP-GFP %GFP positive 2 m one '
                              'sensitive-descriptor-driven positive detection)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (CHOP-GFP ER-stress reporter)'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': 'dfbfce98157291025fc9d9e292d91eb62eebd3bfcbd9114d87e6f990888baab8',
          'source_row_uid': 'sr_0bcd9001cf4843b589db71d4e41dbb2e',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (p21-GFP %GFP positive 2 m positive detection)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (p21-GFP DNA-damage reporter)'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': '63deaff70d6b6acfd219298997eed2a2206571e9e1f346d0c683956d6b569d22',
          'source_row_uid': 'sr_1328175df33040b99bf75e2a165aa6bc',
          'source_id': 'dili_v1',
          'measurement_text': 'EC50 significantly lower than SAL alone or SAL + 1 µg/ml tiamulin',
          'assay_and_readout': 'LDH release assay—membrane integrity/lysis as % of solvent '
                               'control'},
  'measurement': '1',
  'unit': 'µg/ml tiamulin',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': '0895c417e7e646919ad4ee0de80e35a51f7f04eb28a59b839e79a8f3cb725275',
          'source_row_uid': 'sr_13b1ed2b57d743efb200d145ce4580be',
          'source_id': 'dili_v2',
          'measurement_text': 'GSH-dependent formation of M3-M5 and the AM-GSH conjugate with '
                              'saturation kinetics, K_m = 0.3 mM for GSH; negligible product '
                              'formation in the absence of GSH',
          'assay_detail': 'GSH trapping of the reactive metabolite during P450-mediated '
                          'bioactivation; GSH-concentration-dependent formation of metabolites '
                          'M3-M5 and the AM-GSH conjugate, with K_m for GSH'},
  'measurement': '0.3',
  'unit': 'mM',
  'expected_reason': None},
 {'row': {'id': '82d2234e68176e187a73b85e0758d813155a45b80d8b3f5a97081c92abe5e51c',
          'source_row_uid': 'sr_15916df63fbb42e0bef07524fa4c077c',
          'source_id': 'dili_v1',
          'measurement_text': 'prevented (GSH maintained, GSSG decreased vs glyoxal 5 mM)',
          'assay_and_readout': 'intracellular GSH depletion and extracellular GSSG formation '
                               '(thiol redox)'},
  'measurement': '5',
  'unit': 'mM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '588d8c69af5003eb1382b4ba691987f0847e51dcbfde43b0970da04bca1ef16a',
          'source_row_uid': 'sr_169931a4f7b74f19b8d75a038da50e8a',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (p21-GFP %GFP positive 2 m)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (p21-GFP DNA-damage reporter) versus '
                               'time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': '5c242d251e2e5d8bb10d33bb624daa1c4ae79aad2e50a511b2b5ac10e80971a4',
          'source_row_uid': 'sr_16e4be43f18d4897bcf002e7bc991f02',
          'source_id': 'dili_v2',
          'measurement_text': 'Cyclosporin A largely blocks Ca2+-induced pore opening by '
                              'interacting with the low-capacity binding sites (K_d cyclosporin 8 '
                              'nM); cyclosporin also brings about closure of the pre-opened pore',
          'assay_detail': 'Ca2+-induced inner-membrane pore opening measured by [14C]sucrose entry '
                          'into the matrix space; cyclosporin A inhibition of pore opening'},
  'measurement': '8',
  'unit': 'nM',
  'expected_reason': None},
 {'row': {'id': '1f8782bb555b368a0f0cb9fadee7b33cd23bda7b8fdd13c5280c91ba0a38bbff',
          'source_row_uid': 'sr_20bfad0f28cd427ab5a1a5c7e6e28889',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (initial CHOP-GFP %GFP positive 2 m at 100-fold C-max)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (CHOP-GFP ER-stress reporter) versus '
                               'time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': 'd85fe8fa51177c5c009397f6b621e253faddb9d13d3eb9344bca3e4022d7da19',
          'source_row_uid': 'sr_24eebec2ed2140e2aa66d620a46ee6bc',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (CHOP-GFP %GFP positive 2 m)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (CHOP-GFP ER-stress reporter) versus '
                               'time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': '6ef2c07dc02c95a0fc266b572f573546dc305b7f914440e2348948ab465901cd',
          'source_row_uid': 'sr_2b683070f2804635b8c3d14066592026',
          'source_id': 'dili_v2',
          'measurement_text': 'Progressive formation of methemoglobin observed (peak at 405 nm), '
                              'indicating NO release from GTN during denitration',
          'assay_detail': 'NO release quantitated by its conversion of oxyhemoglobin to '
                          'methemoglobin (absorbance at 405 nm) during anaerobic GTN denitration '
                          'by microsomes'},
  'measurement': '405',
  'unit': 'nm',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': 'f3693faa6276517fcaee9d6bcf373ae1ee0303a5cfc4cebe106e92f32402ca6a',
          'source_row_uid': 'sr_3329662bb314404eaea8a9af7e1e74d1',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (Srxn1-GFP %GFP positive 2 m descriptor more sensitive '
                              'than mean GFP intensity, which was not defined as positive)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (Srxn1-GFP oxidative-stress '
                               'reporter); intensity-based feature alone was negative'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': '6d54ffd39cbe5c8f7a290564f9242ec8bb67b43ae2ac0a839a300f91fd22f627',
          'source_row_uid': 'sr_3d41a162e5a24d759f7c5df02307afc1',
          'source_id': 'dili_v2',
          'measurement_text': 'potent uncoupling activity; more potent than the C4- and C8-analogs '
                              '(EC50 approximately 100 nM for C4/C8)',
          'assay_detail': 'Uncoupling assay: uncoupled respiratory rate after addition of '
                          'uncoupler, expressed as percent of basal respiratory rate'},
  'measurement': '100',
  'unit': 'nM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': 'd12e6cf1259249dbfe7418c1d5f36a73f20b4dd419c61bab45baf8452a1e1f42',
          'source_row_uid': 'sr_3fd7393720cc420b8000d388d084d920',
          'source_id': 'dili_v2',
          'measurement_text': '17g decreased the MMP of A549 cells',
          'assay_detail': 'JC-1 fluorescence microscopy of mitochondrial membrane potential (MMP)'},
  'measurement': '17',
  'unit': 'g decreased the MMP of A549 cells',
  'expected_reason': 'unit_contains_context_clause'},
 {'row': {'id': '2f8a909914475ac62b36ea9551ffda6fdbe97fafd2dd175df0b09a9a6528d05b',
          'source_row_uid': 'sr_416976b42dc544fab23e116c231a7b03',
          'source_id': 'dili_v1',
          'measurement_text': 'less toxic to HepG2 than to MN9D cells; IC50 ~30 nM in MN9D',
          'assay_and_readout': 'cell viability/cytotoxicity assay—IC50 determination'},
  'measurement': '30',
  'unit': 'nM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '83a927d6be41b535426d68c847743a4f8e682267b37562baca9d1b63f64cfd4b',
          'source_row_uid': 'sr_4a501f5e9216474cbee6049b5111421c',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (Srxn1-GFP %GFP positive 2 m, onset from 50-fold C-max as '
                              'primary and only stress type; negative on mean-intensity feature)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (Srxn1-GFP oxidative-stress '
                               'reporter); no response at intensity-feature level'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': '95a5db1dc150a1a5b42cc96cc441ab7b5f7201e9dbe889af5b7af5672097f1c5',
          'source_row_uid': 'sr_4fbef7fc9e1f4183a268d31aef6ae42b',
          'source_id': 'dili_v1',
          'measurement_text': 'prevented lipid peroxidation (significant vs glyoxal 5 mM, P<0.05)',
          'assay_and_readout': 'lipid peroxidation measurement in isolated rat hepatocytes—lipid '
                               'peroxidation level, prevention relative to glyoxal 5 mM'},
  'measurement': '5',
  'unit': 'mM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '500b554f389c68a1406b5206b46aca6224495049251065d72819d4b1087b09b4',
          'source_row_uid': 'sr_5410b72a7ff94934a06e146f276faad2',
          'source_id': 'dili_v2',
          'measurement_text': 'IC50 in the range of 50 µM',
          'assay_detail': '[3H]-carnitine-ex/carnitine-in antiport in proteoliposomes; '
                          'dose-response transport IC50'},
  'measurement': '50',
  'unit': 'µM',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': '469f83023f6ccae1e3bd8c6b20398dd96cdd984ca49359d2aede8887e99bc453',
          'source_row_uid': 'sr_54da9b0a93e848e9ab43215e97ae27d3',
          'source_id': 'dili_v1',
          'measurement_text': 'decreased (small but significant) vs no-substrate control; basal '
                              'ATP 2.6 mM',
          'assay_and_readout': 'ATP content—total intracellular ATP concentration (mM)'},
  'measurement': '2.6',
  'unit': 'mM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '9bf1d89da3692a2bc946f4257af4da69a4b1cf3568fa5e05161c99cd0562d93a',
          'source_row_uid': 'sr_55dd75085c1f4953b93db974bc8389a1',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (initial CHOP-GFP %GFP positive 2 m at 100-fold C-max)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (CHOP-GFP ER-stress reporter) versus '
                               'time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': 'a8cb014a59d38fb154b3933ae444b57427f794489bb3b2508718301595296110',
          'source_row_uid': 'sr_5b22ee9df2d64135bb1d3ef04b79408f',
          'source_id': 'dili_v2',
          'measurement_text': 'Ki of CsA for inhibition of MPT in the range of 5 nM',
          'assay_detail': 'Ki for inhibition of mitochondrial permeability transition (MPT) pore '
                          'opening'},
  'measurement': '5',
  'unit': 'nM',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': '7bb68d1fe0b779e19f3a16da95215301de012488b2bac5cd76f6c9e36b6cab1e',
          'source_row_uid': 'sr_5f2bb7d1523f40d1acea721e3296dbcc',
          'source_id': 'dili_v2',
          'measurement_text': 'IC50 about 0.8 µg/ml in presence of 500 µg/ml erythromycin',
          'assay_detail': 'mitochondrial protein synthesis rate, [14C]leucine incorporation into '
                          'protein as % of control, in presence of erythromycin'},
  'measurement': '0.8',
  'unit': 'µg/ml in presence of 500',
  'expected_reason': 'unit_contains_context_clause'},
 {'row': {'id': 'b13a6655149c607488f2dfa128cc91674c43942e8b31e9fc5b877bfb8584b6ef',
          'source_row_uid': 'sr_63b6162f0ab94a84947b5dd93acbd082',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (initial CHOP-GFP %GFP positive 2 m at 100-fold C-max)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (CHOP-GFP ER-stress reporter) versus '
                               'time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '243fe7efee4ae5f2cc2093d9a60e4dadb107abd37fe35b951379cc3edd7f9bc1',
          'source_row_uid': 'sr_66a83d4ef8ba46bbaa3aed02c35db217',
          'source_id': 'dili_v1',
          'measurement_text': 'EC50: 0.08 uM (half-maximally effective concentration for '
                              'dose-dependent stimulation of ureagenesis)',
          'unit_text': 'umol_per_l',
          'assay_and_readout': 'ureagenesis assay—rate of urea biosynthesis measured in isolated '
                               'rat hepatocyte incubations'},
  'measurement': '0.08',
  'unit': 'umol_per_l',
  'expected_reason': None},
 {'row': {'id': 'f5a252c6524fc84e8b8919e9317b1fd18baa630958090c1180f49b7f71c0d008',
          'source_row_uid': 'sr_6b90b950f4d545e397fd518075658e7b',
          'source_id': 'dili_v1',
          'measurement_text': 'no effect on LDH or viability (89 U/L vs control 76 U/L)',
          'unit_text': 'U/L',
          'assay_and_readout': 'lactate dehydrogenase (LDH) activity in culture medium—membrane '
                               'leakage'},
  'measurement': '89',
  'unit': 'U/L',
  'expected_reason': 'different_qualitative_focal_outcome'},
 {'row': {'id': '56f432bc3a67b745990e9c4e4726dd93d5fce99ac3c474b6a98bfca299349f96',
          'source_row_uid': 'sr_6c2dc3d4a6b7413b959ab47741e6d813',
          'source_id': 'dili_v1',
          'measurement_text': 'prevented lipid peroxidation (significant vs glyoxal 5 mM, P<0.05)',
          'assay_and_readout': 'lipid peroxidation measurement in isolated rat hepatocytes—lipid '
                               'peroxidation level, prevention relative to glyoxal 5 mM'},
  'measurement': '5',
  'unit': 'mM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': 'c709f534d207121edf577d91f729c083b2cb4e7b9d8b012e03d9314f393bf984',
          'source_row_uid': 'sr_6e52372c44874cbdb3cd298d018e6464',
          'source_id': 'dili_v1',
          'measurement_text': 'synergistic mitochondrial damage; giant mitochondria 3 um in length',
          'assay_and_readout': 'transmission electron microscopy—mitochondrial ultrastructure '
                               '(enlargement, swelling, cristae disruption, giant mitochondria)'},
  'measurement': '3',
  'unit': 'um',
  'expected_reason': None},
 {'row': {'id': '09306b9a77c9a39fb4990129e031f27f3b3a2929e98a8700c3e721b76da681c9',
          'source_row_uid': 'sr_6f3aff84194f4600a0d3b863d8d102aa',
          'source_id': 'dili_v1',
          'measurement_text': 'prevented lipid peroxidation (significant vs glyoxal 5 mM, P<0.05)',
          'assay_and_readout': 'lipid peroxidation measurement in isolated rat hepatocytes—lipid '
                               'peroxidation level, prevention relative to glyoxal 5 mM'},
  'measurement': '5',
  'unit': 'mM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': 'b3e364783b78f380159fb9c00a8e37d7c3094594daa6ee5ea06367cc0fcce815',
          'source_row_uid': 'sr_6fda5a2e9a354ed9b2471fba3034edeb',
          'source_id': 'dili_v1',
          'measurement_text': 'prevented lipid peroxidation (significant vs glyoxal 5 mM, P<0.05)',
          'assay_and_readout': 'lipid peroxidation measurement in isolated rat hepatocytes—lipid '
                               'peroxidation level, prevention relative to glyoxal 5 mM'},
  'measurement': '5',
  'unit': 'mM',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '44e38dd9969d1fd7f632d9a6c6a86a9ba388f818acf82ade5b1aeae10f564e0c',
          'source_row_uid': 'sr_7551cb68f21a429cb8b617c608480a10',
          'source_id': 'dili_v2',
          'measurement_text': 'SPD bound to MTP with Kd = 0.1 µM and allosterically enhanced its '
                              'enzymatic FAO activities',
          'assay_detail': 'Enzymatic fatty acid β-oxidation (FAO) activity of the MTP complex; '
                          'allosteric activation readout; binding affinity Kd by biochemical '
                          'binding assay'},
  'measurement': '0.1',
  'unit': 'µM',
  'expected_reason': None},
 {'row': {'id': 'e4563a7add6b36d61abfe99563d021355a40943caa82114dc534cc465fa461d2',
          'source_row_uid': 'sr_7741fba0e4644a16bbb3c3be19d26444',
          'source_id': 'dili_v2',
          'measurement_text': 'disappearance of the peak at 279 nm, assigned to progressive '
                              'oxidation without conversion to the quinone methide (negative for '
                              'quinone methide formation)',
          'assay_detail': 'UV-Vis spectral monitoring of enzymatic oxidation; assessed conversion '
                          'to a quinone methide derivative'},
  'measurement': '279',
  'unit': 'nm',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': '5c64253787c3ed926cd9f10ed1b3220b3dc383f4ca1a3b8651776286ac5c5b8b',
          'source_row_uid': 'sr_7d1e957402014a458ddfc6f7b8f3c34a',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (Srxn1-GFP %GFP positive 2 m, preceding the CHOP-GFP '
                              'response)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (Srxn1-GFP oxidative-stress reporter) '
                               'versus time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': 'b808a64186d91ce3bbc33058e218a2474a1f9c3b733909d52de538a628fa2043',
          'source_row_uid': 'sr_961da23e78ec43dbb05fe3472ee8c6ac',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (Srxn1-GFP %GFP positive 2 m, reported qualitatively; BMC '
                              'for severe-DILI group lower than non-severe)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (GFP intensity ≥2x control mean) '
                               'versus time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': '75d4f7127ca5eba9273bb5878872fa5dfc3a420036c40d77aa21f4cd86363df0',
          'source_row_uid': 'sr_9f79fa0a23124a6bbb71b47017c325ac',
          'source_id': 'dili_v1',
          'measurement_text': 'EC50 significantly lower than SAL alone and than SAL with 1 µg/mL '
                              'tiamulin, in all assays',
          'assay_and_readout': 'Cytotoxicity panel (NRU, MTT, TPC, LDH)—EC50'},
  'measurement': '1',
  'unit': 'µg/mL tiamulin',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': '93749651b676ea61558e6cbc936042a0bfe7278d31c85f8459aa6b7d8a9eaf06',
          'source_row_uid': 'sr_9f86e853a8114b3f9a0e43518253f348',
          'source_id': 'dili_v2',
          'measurement_text': 'Km in the range of 1 µM; Vmax much lower than that of DB',
          'assay_detail': 'NADH-CoQ reductase (complex I) activity with CoQ2 as electron acceptor '
                          '- Km and Vmax determination'},
  'measurement': '1',
  'unit': 'µM',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': '690849d021f588936b9dd2f3e792cd33699c2d36d9e832469c0b7246cd05f0f7',
          'source_row_uid': 'sr_ab7295a22d254c3bbdcdaffc55f1d91d',
          'source_id': 'dili_v2',
          'measurement_text': 'MMP first-responding HCS endpoint at MEC 130 µM',
          'assay_detail': 'Live-cell high-content imaging of mitochondrial membrane potential '
                          '(MMP) perturbation in 3D spheroid; MEC value'},
  'measurement': '130',
  'unit': 'µM',
  'expected_reason': None},
 {'row': {'id': 'c832ff2235b242ac8acbe6a6279166929a77db4dd0f24cde1dcc4a5b2d8f4f1c',
          'source_row_uid': 'sr_b134110271fb469595b3885062f48369',
          'source_id': 'dili_v2',
          'measurement_text': 'toxicity threshold detected by JC-1 staining was 10 ng/ml',
          'assay_detail': 'JC-1 staining; dissipation of mitochondrial membrane potential (Δψm) '
                          'toxicity threshold'},
  'measurement': '10',
  'unit': 'ng/ml',
  'expected_reason': None},
 {'row': {'id': 'adb01b41cb67b4f8133df9f5820cb8c1cf8a628f8028a3bc1a350f554d92df1a',
          'source_row_uid': 'sr_ba697555c9df4acfb964af7cb7755ba3',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (Srxn1-GFP %GFP positive 2 m; PoD for this reporter '
                              'higher in C-max than for ICAM1-GFP)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (Srxn1-GFP oxidative-stress reporter) '
                               'versus time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': '614a4d99a53beea5d9a67fdc2009c3d01412852554867661c408b2a6a9cc26f8',
          'source_row_uid': 'sr_c1f658fdeb5548f98739596c59f6cbe2',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (Srxn1-GFP %GFP positive 2 m, preceding the CHOP-GFP '
                              'response)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (Srxn1-GFP oxidative-stress reporter) '
                               'versus time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'},
 {'row': {'id': '2cb50d359a7148c83d8c0d0188ba73d2d85c874223ff722e88a69184f5dac9f2',
          'source_row_uid': 'sr_ccf8167905444636bf45e12e4f3d33e2',
          'source_id': 'dili_v2',
          'measurement_text': 'IC50 in the range of 50 µM on both rat and human recombinant '
                              'protein',
          'assay_detail': '[3H]-carnitine-ex/carnitine-in antiport in proteoliposomes; '
                          'dose-response transport IC50'},
  'measurement': '50',
  'unit': 'µM',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': '43fbb3b6c9d22f1a6e546fa8f34e167630ac817e0be8dd1099fe2281873bab1b',
          'source_row_uid': 'sr_e1da4f1783854e1c819e3b0d4e83d392',
          'source_id': 'dili_v1',
          'measurement_text': 'LOEC: 140 µM (lowest observed effect concentration causing LDH '
                              'leakage)',
          'unit_text': 'umol_per_l',
          'assay_and_readout': 'LDH leakage—lactate dehydrogenase release as a cytotoxicity '
                               'endpoint'},
  'measurement': '140',
  'unit': 'umol_per_l',
  'expected_reason': None},
 {'row': {'id': 'a067d4c626b5a8aeb6b658eaa8639af1323f03d65c4b6d0a6ba7a27ffd883ddd',
          'source_row_uid': 'sr_f7d3bfacc635499fa17e7cf123020c7b',
          'source_id': 'dili_v2',
          'measurement_text': 'GEA effects on MPT less than diamide; ATP accelerated pore opening '
                              'after GEA treatment (K_a for ATP 1.9 uM in GEA-treated '
                              'mitochondria)',
          'assay_detail': 'Calcium-dependent mitochondrial permeability transition pore-opening '
                          'response to ATP after NO-donor pretreatment'},
  'measurement': '1.9',
  'unit': 'uM',
  'expected_reason': None},
 {'row': {'id': '3fb173e83ca59ac4d66458324f129e41160fec352bf2773de3888fa8b812f58d',
          'source_row_uid': 'sr_fb961415b7a4464ea5101e16c8fd4d09',
          'source_id': 'dili_v2',
          'measurement_text': 'Plasma ketone bodies 0.81 mM (ethylhexanol) vs 1.59 ± 0.07 mM '
                              '(control), P < 0.001',
          'assay_detail': 'Plasma ketone bodies (acetoacetate + β-hydroxybutyrate) by standard '
                          'enzymatic methods, in vivo; ketogenesis/FAO flux readout'},
  'measurement': '0.81',
  'unit': 'mM',
  'expected_reason': None},
 {'row': {'id': '08e857789e3eeb2d73b013c36810cf2ec043bc7ada819d2f2793e076f42e3efa',
          'source_row_uid': 'sr_fef50ad79bd94f1b97848fffe4a1af00',
          'source_id': 'dili_v1',
          'measurement_text': 'increased (CHOP-GFP %GFP positive 2 m)',
          'assay_and_readout': 'High-content live-cell confocal BAC reporter screen—fraction of '
                               'cells with %GFP positive 2 m (CHOP-GFP ER-stress reporter) versus '
                               'time'},
  'measurement': '2',
  'unit': 'm',
  'expected_reason': 'unsupported_or_uncertain_unit'})


_DILI_GUARD_V6_REVIEWED_CASES += ({'row': {'id': '1a3d68ed13dddcf30ffe7c638ace2c0f3c5282652581db477e8edb7c9108e2a0',
          'source_row_uid': 'sr_50df859343f84941bd642df861f9da9e',
          'source_id': 'dili_v1',
          'measurement_text': 'optimal DMSO concentration: 15%',
          'unit_text': 'percent_vv',
          'support_text': 'We cryopreserved single and spheroid porcine hepatocytes in various '
                          'concentrations of dimethyl sulfoxide (DMSO) using hormonally defined '
                          'medium (HDM) and presentation solutions such as University of Wisconsin '
                          '(UW) and fetal bovine serum. After thawing we measured viability, '
                          'plating efficiency, ammonia removal, urea synthesis, and albumin '
                          'secretion. The optimal DMSO concentration for porcine hepatocyte '
                          'cryopreservation was 15%.',
          'assay_and_readout': 'Post-thaw viability, plating efficiency, ammonia removal, urea '
                               'synthesis, and albumin secretion measured to identify the optimal '
                               'cryoprotectant concentration'},
  'measurement': '15',
  'unit': 'percent_vv',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': 'cdc129c949b7c4544466a7f4888985eac680104711a2cf84e2ff636436507e3d',
          'source_row_uid': 'sr_da7705b4426447dead897d470af577fd',
          'source_id': 'dili_v1',
          'measurement_text': 'comparable to KD 100 µM (reduced LDH/MDA, improved viability vs '
                              'iron-alone)',
          'support_text': 'In all tested experimental conditions, the efficacy of 100 µM of the '
                          'dihydroxamate chelator KD was close to that of 50 µM of the '
                          'trihydroxamate chelator DFO in protecting rat hepatocytes against the '
                          'toxic (LDH and MDA release) effect of iron-citrate and in improving '
                          'viability.',
          'assay_and_readout': 'LDH release, MDA production and MTT reduction (combined protective '
                               'effect against iron-citrate)'},
  'measurement': '100',
  'unit': 'µM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': '7d5714ef9b2b8311b6ccfb4aeec47780ec878f7f95d54ec009be64922e3ce76d',
          'source_row_uid': 'sr_bf9a0d7165c2453dbb2446cd5bda54b0',
          'source_id': 'dili_v1',
          'measurement_text': 'induced proliferation of hepatocytes (growth-promoting effect); '
                              'dose-dependently inhibited by TGF-alpha-neutralizing antibody (IC50 '
                              'for inhibition 35 ng/ml)',
          'support_text': 'Hepatocytes were plated at 3.3x10^4 cells/cm^2 and cultured; after the '
                          'medium change hepatocytes were treated with 10^-8 M PGI2 or 10^-9 M '
                          'carbaprostacyclin for 4 h in the presence or absence of TGF-alpha- or '
                          'IGF-I-neutralizing antibody (2.5-100 ng/ml). Hepatocyte proliferation '
                          'is expressed as the percent increase in total number of nuclei compared '
                          'with the control culture. Results are means +/- S.E.M. of three '
                          'independent experiments.',
          'assay_and_readout': 'IP-receptor agonist-induced hepatocyte proliferation measured as '
                               'percent increase in total number of nuclei compared with the '
                               'control culture'},
  'measurement': '35',
  'unit': 'ng/ml',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '89e6f129f173cf3301147d07cb8ec4863e71068f9836d7179a1fe4e657ab9535',
          'source_row_uid': 'sr_a1a4a9c1abc84529a6eaf7f72242a79c',
          'source_id': 'dili_v1',
          'measurement_text': 'induced proliferation of hepatocytes (growth-promoting effect); '
                              'dose-dependently inhibited by TGF-alpha-neutralizing antibody (IC50 '
                              'for inhibition 35 ng/ml)',
          'support_text': 'Hepatocytes were plated at 3.3x10^4 cells/cm^2 and cultured as '
                          'described in the legend to Fig. 1. After the medium change, hepatocytes '
                          'were treated with 10^-9 M carbaprostacyclin for 4 h in the presence or '
                          'absence of TGF-alpha-neutralizing antibody or IGF-I-neutralizing '
                          'antibody (2.5-100 ng/ml). Hepatocyte proliferation is expressed as the '
                          'percent increase in total number of nuclei compared with the control '
                          'culture.',
          'assay_and_readout': 'IP-receptor agonist-induced hepatocyte proliferation measured as '
                               'percent increase in total number of nuclei compared with the '
                               'control culture'},
  'measurement': '35',
  'unit': 'ng/ml',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': 'c5e2b48729e9c73b186bf2da646614bb55d06f45399065f3856b101100d5af92',
          'source_row_uid': 'sr_0f6032d411e44019ba0f8601dcad3a15',
          'source_id': 'dili_v1',
          'measurement_text': 'induced an increase in hepatocyte DNA synthesis (growth-promoting '
                              'effect); dose-dependently inhibited by TGF-alpha-neutralizing '
                              'antibody (IC50 for inhibition 25 ng/ml)',
          'support_text': 'Hepatocytes were plated at a density of 3.3x10^4 cells/cm^2 and '
                          'cultured as described. After the medium change, hepatocytes were '
                          'treated with 10^-8 M PGI2 or 10^-9 M carbaprostacyclin for 4 h in the '
                          'presence or absence of TGF-alpha-neutralizing antibody or '
                          'IGF-I-neutralizing antibody (2.5-100 ng/ml). The rate of hepatocyte DNA '
                          'synthesis is expressed as dpm/(h.mg protein).',
          'assay_and_readout': 'hepatocyte DNA synthesis rate expressed as dpm/(h.mg protein) '
                               'after 4 h of culture'},
  'measurement': '25',
  'unit': 'ng/ml',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': 'eea71682cfcac03874c9ad34833a63de710c6c911d611b67f2353156567afc79',
          'source_row_uid': 'sr_148e79625cbf4239aef8071ede1a7f14',
          'source_id': 'dili_v1',
          'measurement_text': 'induced an increase in hepatocyte DNA synthesis (growth-promoting '
                              'effect); dose-dependently inhibited by TGF-alpha-neutralizing '
                              'antibody (IC50 for inhibition 25 ng/ml)',
          'support_text': 'Hepatocytes were plated at a density of 3.3x10^4 cells/cm^2 and '
                          'cultured as described. After the medium change, hepatocytes were '
                          'treated with 10^-9 M carbaprostacyclin for 4 h in the presence or '
                          'absence of TGF-alpha-neutralizing antibody or IGF-I-neutralizing '
                          'antibody (2.5-100 ng/ml). The rate of hepatocyte DNA synthesis is '
                          'expressed as dpm/(h.mg protein).',
          'assay_and_readout': 'hepatocyte DNA synthesis rate expressed as dpm/(h.mg protein) '
                               'after 4 h of culture'},
  'measurement': '25',
  'unit': 'ng/ml',
  'expected_reason': 'control_or_comparator_not_outcome'},
 {'row': {'id': '34c53622fe99c18704dff84d23136c9c4e4430c243c6a8b940ae1100ba4eac71',
          'source_row_uid': 'sr_0a19295a7e2d44d1a55371057ee1e9ab',
          'source_id': 'dili_v2',
          'measurement_text': 'optimal concentration for activation of MGST activity was 5 mM; '
                              'activation magnitude lower than equimolar NEM',
          'support_text': 'After 30 s incubation, the optimal concentration for acrolein as well '
                          'as NEM in activation of MGST activity was 5 mM (Fig. 1). The magnitude '
                          'of activation by NEM was higher than that of acrolein.',
          'assay_detail': 'MGST glutathione transferase activity as functional readout of covalent '
                          'thiol adduction; concentration dependency of enzyme activation'},
  'measurement': '5',
  'unit': 'mM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': 'b3e15003e17d5d98a06b9525f7ac55b7f91cdbc054c619822c38f35eb8942603',
          'source_row_uid': 'sr_c5ed900ee8ea4a05a204f54b32eb85e3',
          'source_id': 'dili_v2',
          'measurement_text': 'optimal concentration for activation of MGST activity was 5 mM; '
                              'activation magnitude higher than equimolar acrolein',
          'support_text': 'After 30 s incubation, the optimal concentration for acrolein as well '
                          'as NEM in activation of MGST activity was 5 mM (Fig. 1). The magnitude '
                          'of activation by NEM was higher than that of acrolein.',
          'assay_detail': 'MGST glutathione transferase activity as functional readout of covalent '
                          'thiol adduction; concentration dependency of enzyme activation'},
  'measurement': '5',
  'unit': 'mM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': 'fc7ca66d40bf51bb358bf3c3fa65967a0523c90c237e091311fbc547b26d0b1d',
          'source_row_uid': 'sr_36ed35d8d7be43a68264aea20bb131f3',
          'source_id': 'dili_v2',
          'measurement_text': 'sensitivity similar to rat liver (I50 comparable to 2.7 uM); no '
                              'exact I50 value reported',
          'support_text': 'At a fixed carnitine concentration of 200 uM, CPT I of human foetal '
                          'liver mitochondria showed sensitivity to malonyl-CoA similar to that of '
                          'rat liver, which exhibited an I50 of 2.7 uM.',
          'assay_detail': 'Carnitine palmitoyltransferase I (CPT I) activity inhibition; I50 for '
                          '50% suppression at fixed 200 uM carnitine'},
  'measurement': '2.7',
  'unit': 'uM',
  'expected_reason': 'missing_exact_focal_value'},
 {'row': {'id': '37b137e6677834bb3ebe41657f705c7da26c13f69a8d4d89400baa70a4b8a13a',
          'source_row_uid': 'sr_130796d06a304cf4b6e341425e2d3f3a',
          'source_id': 'dili_v2',
          'measurement_text': 'no reduction in mitochondrial membrane potential for mCPG 500 µM',
          'support_text': '(2) (A) mCPG does not reduce mitochondrial membrane potential. Stallion '
                          'spermatozoa were incubated in presence of mCPG or mCPG and Cyss, and '
                          'mitochondrial membrane potential evaluated after 3 h of incubation at '
                          '37°C. (E) are samples supplemented with mCPG 500 µM.',
          'assay_detail': 'Mitochondrial membrane potential (delta-psi-m) flow-cytometry assay, '
                          'readout of the percentage of spermatozoa with high mitochondrial '
                          'membrane potential'},
  'measurement': '500',
  'unit': 'µM',
  'expected_reason': 'different_qualitative_focal_outcome'},
 {'row': {'id': '5e1ef499bca269feaa738e1ac02b60ef1d685332d4bed58e4f78b331d61f2723',
          'source_row_uid': 'sr_53bd2b3fd5774934b2165627cebf6266',
          'source_id': 'dili_v2',
          'measurement_text': 'highest concentration that did not cause mitochondrial membrane '
                              'potential decrease was 0.5 µM',
          'support_text': 'In the in vitro experiments, we used only mitoTEMPO due to its better '
                          'solubility and accessibility for cultured cells. We tested the '
                          'influence of different mitoTEMPO concentrations on the mitochondrial '
                          'membrane potential of hepatocytes. The highest concentration that did '
                          'not cause membrane potential decrease was 0.5 µM, while for '
                          'non-targeted TEMPO this concentration was 100 times higher.',
          'assay_detail': 'Mitochondrial membrane potential of hepatocytes; highest concentration '
                          'without membrane-potential decrease as the endpoint'},
  'measurement': '0.5',
  'unit': 'µM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': '71b4d68ab91e6e3b03afc1de8c43f1f1cea216865c8db20b1b09f84faaf2d5da',
          'source_row_uid': 'sr_013fb12c10624f568aefeec79c62f131',
          'source_id': 'dili_v2',
          'measurement_text': 'depolarization of the mitochondrial membrane observed when '
                              'concentrations exceed 5 uM',
          'support_text': 'A study performed on mice and humans with isolated mitochondria '
                          'incubated in the presence of palmitoyl carnitine, palmitoyl-CoA and '
                          'oleoyl-CoA reported stimulation of mitochondrial ATP production; '
                          'however, if their concentrations exceed 5 µM, mitochondrial dysfunction '
                          'occurs, ATP synthesis is inhibited and a depolarization of the '
                          'mitochondrial membrane is observed [270].',
          'assay_detail': 'isolated-mitochondria membrane polarization readout'},
  'measurement': '5',
  'unit': 'uM',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': '979b775482ec1032036aaacc23e2f2355775c63906eac51c43d3c90852ad6513',
          'source_row_uid': 'sr_9681b7a54c76426c9d3632a933465f35',
          'source_id': 'dili_v2',
          'measurement_text': 'depolarization of the mitochondrial membrane observed when '
                              'concentrations exceed 5 uM',
          'support_text': 'A study performed on mice and humans with isolated mitochondria '
                          'incubated in the presence of palmitoyl carnitine, palmitoyl-CoA and '
                          'oleoyl-CoA reported stimulation of mitochondrial ATP production; '
                          'however, if their concentrations exceed 5 µM, mitochondrial dysfunction '
                          'occurs, ATP synthesis is inhibited and a depolarization of the '
                          'mitochondrial membrane is observed [270].',
          'assay_detail': 'isolated-mitochondria membrane polarization readout'},
  'measurement': '5',
  'unit': 'uM',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': 'aab665c72b5ed89fe132540a704bf0236b5ae5bb52e151eb8beb1cfea85a1c03',
          'source_row_uid': 'sr_92da295cf2ee4d33a51029f16e3cc537',
          'source_id': 'dili_v2',
          'measurement_text': 'optimal concentration 2 uM (S3 maintained mitochondrial membrane '
                              'potential and cell viability; p < 0.05)',
          'support_text': 'To screen for the optimal concentration of S3, cultured RGCs were '
                          'incubated with different concentrations of S3 for 24 h and then '
                          'assessed for cell viability and mitochondrial membrane potential. '
                          'According to the LDH and JC-1 assay results (p < 0.05), the optimal '
                          'concentration of S3 for RGCs was 2 µM, which was chosen for further '
                          'experiments.',
          'assay_detail': 'JC-1 assay of mitochondrial membrane potential (JC-1 ratio), used '
                          'together with LDH viability to select an optimal non-toxic, '
                          'membrane-potential-preserving concentration'},
  'measurement': '2',
  'unit': 'uM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': 'b07235687b51b6ede2d2ac3db1331616f8b3b8a4500f3ff6f3bd648e492350f1',
          'source_row_uid': 'sr_d108bfdb26094996b30dbb544a3dcf4d',
          'source_id': 'dili_v2',
          'measurement_text': 'optimal concentration 2.0 uM; optimal duration 30 min based on JC-1 '
                              'ratio levels',
          'support_text': 'The authors determined the optimal concentration and incubation time of '
                          'fucoxanthin for retinal ganglion cells (RGCs) according to the levels '
                          'of mitochondrial membrane potential in RGCs. As reflected by the JC-1 '
                          'ratio, the optimal concentration of fucoxanthin for RGCs was 2.0 µM, '
                          'and the optimal duration was 30 min. Fucoxanthin was dissolved in 48 '
                          'mg/ml DMSO, which served as the control due to its poor water '
                          'solubility.',
          'assay_detail': 'JC-1 kit mitochondrial membrane potential; JC-1 red/green ratio readout '
                          'used to select optimal concentration and duration'},
  'measurement': '2.0',
  'unit': 'uM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': 'd74127cd4c2cf4c62929558bfc20b4e4c79105bd0cf9500ed241d6c005ddc8a2',
          'source_row_uid': 'sr_6be112ac19914d6d9cd30b88cd936571',
          'source_id': 'dili_v2',
          'measurement_text': 'depolarization of the mitochondrial membrane observed when '
                              'concentrations exceed 5 uM',
          'support_text': 'A study performed on mice and humans with isolated mitochondria '
                          'incubated in the presence of palmitoyl carnitine, palmitoyl-CoA and '
                          'oleoyl-CoA reported stimulation of mitochondrial ATP production; '
                          'however, if their concentrations exceed 5 µM, mitochondrial dysfunction '
                          'occurs, ATP synthesis is inhibited and a depolarization of the '
                          'mitochondrial membrane is observed [270].',
          'assay_detail': 'isolated-mitochondria membrane polarization readout'},
  'measurement': '5',
  'unit': 'uM',
  'expected_reason': 'bound_not_point'},
 {'row': {'id': '5eca2a36d7469e530a421e10e862d0b7c4d34513c5856bdfe3d9182622e186b8',
          'source_row_uid': 'sr_9d368112be494ba6a864ca04575c1e06',
          'source_id': 'dili_v2',
          'measurement_text': 'optimal stimulatory concentration about 1 mM',
          'support_text': 'The optimal concentration for a stimulatory effect of phencyclidine on '
                          'succinoxidase activity was about 1 mM, and marked inhibition was '
                          'obtained at 7 mM.',
          'assay_detail': 'succinoxidase (complex II) activity, stimulatory concentration'},
  'measurement': '1',
  'unit': 'mM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': 'c2ef543fc9a7d3e4c522028f41dbbc204b42a281f9ea0decc869d87b10ed0832',
          'source_row_uid': 'sr_910dd57c022d4a57b9c052b6a4d7a66f',
          'source_id': 'dili_v2',
          'measurement_text': 'optimal stimulatory concentration 0.4 mM',
          'support_text': 'The optimal concentration for a stimulatory effect of imipramine on '
                          'succinoxidase activity was 0.4 mM, and marked inhibition was obtained '
                          'at 0.8 mM imipramine.',
          'assay_detail': 'succinoxidase (complex II) activity, stimulatory concentration'},
  'measurement': '0.4',
  'unit': 'mM',
  'expected_reason': 'dose_or_concentration_not_outcome'},
 {'row': {'id': '5944479e0173215eab1111c6de1946d712cd72cba8d7549696f40dd0dc219a45',
          'source_row_uid': 'sr_570800dad803478fbc58d4eb92a375de',
          'source_id': 'dili_v2',
          'measurement_text': 'NAC enhanced complex I activity in old synaptic mitochondria; the '
                              'optimum NAC concentration for maximum complex I activity was 10 mM '
                              'in old synaptic preparations',
          'support_text': 'The paper reports that NAC enhances complex I activity in vitro in '
                          'synaptic mitochondria isolated from old mice, and that the optimum NAC '
                          'concentration for maximum complex I activity was 10 mM in old synaptic '
                          'preparations. The authors argue mitochondrial thiolic groups essential '
                          'to oxidative phosphorylation are impaired by aging.',
          'assay_detail': 'Complex I (NADH:ubiquinone oxidoreductase) enzymatic activity measured '
                          'by spectrophotometric assay, expressed as specific activity on a '
                          'protein basis (nmol/min/mg mitochondrial protein)'},
  'measurement': '10',
  'unit': 'mM',
  'expected_reason': 'dose_or_concentration_not_outcome'})

_DILI_GUARD_V6_REVIEWED_CASES += ({'row': {'id': '4aae973bc1ba8340336b351c2906ce63a6c0a0be0cb8a19ac341a9ff490facca',
          'source_row_uid': 'sr_1d2af5dd35204f0d98821bbd0c147b81',
          'source_id': 'dili_v2',
          'measurement_text': 'ATP maintained at a steady concentration of approximately 0.0015 M',
          'support_text': 'Rat-liver homogenate, cyclophorase or mitochondria maintain ATP at a '
                          'steady concentration of approximately 0.0015 M when oxidizing '
                          'succinate.',
          'assay_detail': 'steady-state ATP concentration maintained in tissue preparation during '
                          'substrate oxidation'},
  'measurement': '0.0015',
  'unit': 'M',
  'expected_reason': None},
 {'row': {'id': '2721cf3b7aeb58431b535fa61d1879d57d07458101378a046460736ab9b4e214',
          'source_row_uid': 'sr_4ba9abd1e4694c2ea32a76c532347537',
          'source_id': 'dili_v2',
          'measurement_text': 'competitive Ki vs carnitine 2.8 uM',
          'support_text': 'HPC inhibited the forward reaction in heart mitochondria; its '
                          'competitive inhibitory constant vs carnitine was 2.8 µM in heart '
                          'mitochondria.',
          'assay_detail': 'inhibition of CPT forward reaction in mitochondria, competitive Ki vs '
                          'carnitine'},
  'measurement': '2.8',
  'unit': 'uM',
  'expected_reason': None},
 {'row': {'id': 'c27e122980d642b0e0dfd234e176f27ea4a9fd34e40601c07d8cc1966c7b705a',
          'source_row_uid': 'sr_d36ec41622044a06a235d3480a99f7a1',
          'source_id': 'dili_v2',
          'measurement_text': 'competitive Ki vs carnitine 4.2 uM',
          'support_text': 'HPC inhibited the forward reaction in liver mitochondria; its '
                          'competitive inhibitory constant vs carnitine was 4.2 µM in liver '
                          'mitochondria.',
          'assay_detail': 'inhibition of CPT forward reaction in mitochondria, competitive Ki vs '
                          'carnitine'},
  'measurement': '4.2',
  'unit': 'uM',
  'expected_reason': None})

@pytest.mark.parametrize(
    "case",
    _DILI_GUARD_V6_REVIEWED_CASES,
    ids=lambda case: case["row"]["id"][:12],
)
def test_dili_assignment_guard_v6_reviewed_regressions(
    case: dict[str, object],
) -> None:
    guarded = _guarded_assignment(
        case["row"],
        str(case["measurement"]),
        str(case["unit"]),
    )
    expected_reason = case["expected_reason"]

    assert guarded["status"] == ("ok" if expected_reason is None else "unsure")
    assert guarded["assignment_guard_reason"] == expected_reason
    assert guarded["measurements_json"] == (
        json.dumps(
            [{"measurement": case["measurement"], "unit": case["unit"]}]
        )
        if expected_reason is None
        else "[]"
    )
