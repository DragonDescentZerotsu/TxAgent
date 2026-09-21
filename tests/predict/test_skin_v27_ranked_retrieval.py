from __future__ import annotations

from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder
from predict.retrieval.assay_reranking import build_skin_v27_ranked_retrieval as cli
from predict.retrieval.assay_reranking.runtime import DATA_ACTIVE_CACHE_ROOT, cache_profile_root
from predict.retrieval.assay_reranking.v27_skin import MODEL, SkinV27PromptRenderer


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
