from __future__ import annotations

from pathlib import Path

import yaml

from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder
from predict.retrieval.assay_reranking import build_skin_v27_ranked_retrieval as cli
from predict.retrieval.assay_reranking.runtime import (
    ACTIVE_CACHE_PROFILES, CANONICAL_CACHE_ROOT, DATA_ACTIVE_CACHE_ROOT,
    cache_profile_root,
)
from predict.retrieval.assay_reranking.v27_skin import MODEL, TRAINING_MODEL, SkinV27PromptRenderer
from predict.retrieval.assay_reranking.v27_safety import SafetyV27PromptRenderer


def _record(**values) -> dict:
    return {
        "task_id": "skin_reaction",
        "progressive_level": "L2",
        "record_id": "record-1",
        "parent_id": "parent-1",
        "canonical_smiles": "CCO",
        "source_id": "direct_skin_reaction",
        "source_fields": {
            "endpoint_name": "sensitization",
            "measurement_text": "source measurement",
            "unit_text": "source unit",
            "assay_or_test": "LLNA",
            "outcome_label": "positive",
        },
        "display_measurement_text": "canonical measurement",
        "display_unit_text": "%",
        **values,
    }


def test_skin_v27_renderer_uses_atomic_display_and_hides_query_result() -> None:
    prompt = SkinV27PromptRenderer().render(_record(), "CCN")
    known, query = prompt.split("Experiment B (query measurement hidden)")
    assert "Known reported measurement: canonical measurement" in known
    assert "Reported unit: %" in known
    assert "source measurement" not in prompt and "source unit" not in prompt
    assert "Outcome label: positive" in known and "Outcome label" not in query
    assert "Known reported measurement" not in query


def test_skin_v27_renderer_drops_unit_for_non_scalar_source_text() -> None:
    prompt = SkinV27PromptRenderer().render(
        _record(display_measurement_text="positive", display_unit_text=None), "CCN"
    )
    assert "Known reported measurement: positive" in prompt
    assert "Reported unit" not in prompt


def test_skin_v27_profiles_are_data_backed_and_configure_query_cohort(monkeypatch) -> None:
    for profile in cli.PROFILES.values():
        assert cache_profile_root(profile).parent == DATA_ACTIVE_CACHE_ROOT
    monkeypatch.setattr(builder, "PROFILE", builder.PROFILE)
    monkeypatch.setattr(builder, "QUERY_BENCHMARK", builder.QUERY_BENCHMARK)
    monkeypatch.setattr(builder, "LABEL_RELEASE", builder.LABEL_RELEASE)
    monkeypatch.setattr(builder, "TASK_LEVELS", builder.TASK_LEVELS)
    cli._configure("tdc")
    assert builder.QUERY_BENCHMARK == "tdc"
    assert builder.TASK_LEVELS == {"skin_reaction": ("L2", "L3")}
    assert builder._model_spec("skin_reaction", "L2") == MODEL
    assert builder._model_spec("skin_reaction", "L3") == MODEL


def test_skin_training_successor_keeps_legacy_prompt_immutable() -> None:
    old = SkinV27PromptRenderer()
    new = SkinV27PromptRenderer(training_template=True)
    record = _record(canonical_endpoint_name="skin_sensitization")
    assert old.template_hash == MODEL["prompt_sha256"]
    assert new.template_hash == TRAINING_MODEL["prompt_sha256"]
    assert "Endpoint: skin_sensitization" in new.render(record, "CCN")
    assert "Endpoint: sensitization" in old.render(record, "CCN")
    cli._configure("gold", training_template=True)
    assert builder._model_spec("skin_reaction", "L2") == TRAINING_MODEL
    assert builder.BASE_RANKING_ROOT == cache_profile_root(cli.PROFILES["gold"])
    assert builder.SHARED_LATER_PARENT_UNIVERSE


def test_skin_inference_copy_preserves_training_visible_context() -> None:
    record = _record(
        source_id="sensitization_aop",
        canonical_endpoint_name="skin_sensitization",
        source_fields={
            "assay_type": "HRIPT", "aop_event": "skin_sensitization",
            "experimental_conditions": "CandidateDrug caused reactions in 51 subjects",
            "qualifying_conditions": "CandidateDrug was positive",
            "needs_more_context": "no",
        },
    )
    renderer = SkinV27PromptRenderer(training_template=True)
    known, hidden = renderer.render(record, "CCN").split("Experiment B (query measurement hidden)")
    assert "CandidateDrug" in known
    assert "CandidateDrug" in hidden
    assert "Assay type: HRIPT" in hidden
    assert "Needs more context: no" in hidden


def test_v27_successor_profiles_are_registered_on_active_shelf() -> None:
    from predict.retrieval.assay_reranking.build_safety_v27_ranked_retrieval import profile
    from predict.retrieval.assay_reranking.score_tdc_ranked_retrieval import _profile

    profiles = set(cli.TRAINING_PROFILES.values())
    profiles.update(profile(task, benchmark) for task in ("ames", "dili", "carcinogens")
                    for benchmark in ("gold", "tdc"))
    profiles.update({profile("carcinogens", "gold", width) for width in (40, 50)})
    profiles.update({profile("ames", "tdc", width) for width in (40, 50)})
    profiles.update(_profile(task, "tdc-v1") for task in ("ames", "dili", "carcinogens"))
    for name in profiles:
        assert name in ACTIVE_CACHE_PROFILES
        assert cache_profile_root(name).is_relative_to(CANONICAL_CACHE_ROOT)


def test_opt_in_v27_cache_configs_route_to_successors() -> None:
    from predict.retrieval.assay_reranking.build_safety_v27_ranked_retrieval import profile
    from predict.retrieval.assay_reranking.score_tdc_ranked_retrieval import _profile

    root = Path(cli.__file__).parent
    for benchmark in ("gold", "tdc"):
        path = root / f"ranked_level_retrieval_{benchmark}_v1_v27_successors_v1.yaml"
        caches = yaml.safe_load(path.read_text())["caches"]
        assert set(caches) == {
            "bbb_martins", "bioavailability_ma", "skin_reaction", "ames", "dili", "carcinogens",
        }
        assert (root / caches["skin_reaction"]["later"]).resolve() == (
            cache_profile_root(cli.TRAINING_PROFILES[benchmark]) / "skin_reaction/RELEASE_INDEX.json"
        )
        for task in ("ames", "dili", "carcinogens"):
            capacity = 40 if (benchmark, task) in {("gold", "carcinogens"), ("tdc", "ames")} else 100
            assert (root / caches[task]["later"]).resolve() == (
                cache_profile_root(profile(task, benchmark, capacity)) / task / "RELEASE_INDEX.json"
            )
            if benchmark == "tdc":
                assert (root / caches[task]["L1"]).resolve() == (
                    cache_profile_root(_profile(task, "tdc-v1")) / task / "RELEASE_INDEX.json"
                )


def test_safety_v27_copy_uses_training_visibility_filter() -> None:
    record = {
        "task_id": "dili", "progressive_level": "L2", "canonical_smiles": "CCO",
        "canonical_endpoint_name": "liver_injury",
        "canonical_pair_fields_json": "{}",
        "canonical_measurement_text": "positive", "canonical_unit_text": "binary",
        "source_fields": {
            "assay_format": "cells", "result_status": "positive",
            "molecule_name": "CandidateDrug", "maximum_reported_severity": "severe",
            "qualifying_conditions": "CandidateDrug caused liver injury",
            "support_text": "result quote",
        },
    }
    prompt = SafetyV27PromptRenderer("dili").render(record, "CCN")
    known, hidden = prompt.split("Experiment B (query measurement hidden)")
    assert "Measurement text: positive" in known
    assert "Result status: positive" in known
    assert "Result status" not in hidden
    assert "Assay format: cells" in hidden
    assert "CandidateDrug" in hidden
    assert "Maximum reported severity: severe" in hidden
    assert "Measurement text" not in hidden
    assert "result quote" not in prompt
