import json
from types import SimpleNamespace

import pytest

from data.processing.evidence_library.shared.v1.clustered_auxiliary_mapping import (
    AuxiliaryExtractionSpec,
    Cluster,
    _terminal_api_error,
    _query_cluster,
    distinct_values,
    validate_response,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.data_processing.build_embedding_bucket_mapping import (
    DEFAULT_MODEL,
    extraction_specs,
)


def test_bbb_mapping_specs_cover_the_reviewed_six_sections():
    specs = extraction_specs()
    assert DEFAULT_MODEL == "gpt-5.4-mini"
    assert {(spec.source_id, spec.output_field) for spec in specs} == {
        (source, output)
        for source in ("direct_bbb", "passive_permeability", "efflux_transport")
        for output in ("global_context", "global_species_context")
    }


def test_cluster_response_requires_complete_exact_ids_and_clean_labels():
    ids = {"item_0001": "Caco-2", "item_0002": "MDCK-MDR1"}
    valid = validate_response(
        json.dumps({"mapping": {"item_0001": "cell assay", "item_0002": "transporter cell assay"}}),
        item_ids=ids,
        null_sentinel=None,
    )
    assert valid == {
        "item_0001": "cell assay",
        "item_0002": "transporter cell assay",
    }
    with pytest.raises(ValueError, match="mapping ID mismatch"):
        validate_response(
            json.dumps({"mapping": {"item_0001": "cell assay"}}),
            item_ids=ids,
            null_sentinel=None,
        )


def test_distinct_values_drop_only_explicit_null_like_values():
    assert distinct_values([None, "unknown", " Caco-2 ", "Caco-2", "in vivo"]) == [
        "Caco-2",
        "in vivo",
    ]


def test_json_mode_request_explicitly_names_json(tmp_path):
    captured = {}

    class Completions:
        def create(self, **request):
            captured.update(request)
            return SimpleNamespace(
                model="gpt-5.4-mini",
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"mapping":{"v0000":"cell assay"}}'))],
                usage=None,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    mapping, _ = _query_cluster(
        client,
        spec=AuxiliaryExtractionSpec(
            source_id="direct_bbb",
            input_path=tmp_path / "unused.parquet",
            input_column="assay_model",
            output_field="global_context",
            prompt="Map the values.",
        ),
        cluster=Cluster("cluster_test", ("MDCK",)),
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        max_retries=1,
        retry_delay=0,
    )
    assert mapping == {"v0000": "cell assay"}
    assert "json" in captured["messages"][0]["content"].casefold()


def test_credit_exhaustion_is_terminal_but_transient_rate_limit_is_not():
    exhausted = RuntimeError("quota")
    exhausted.status_code = 429
    exhausted.body = {"code": "credit_balance_exhausted"}
    transient = RuntimeError("rate")
    transient.status_code = 429
    transient.body = {"code": "rate_limit_exceeded"}
    assert _terminal_api_error(exhausted)
    assert not _terminal_api_error(transient)
