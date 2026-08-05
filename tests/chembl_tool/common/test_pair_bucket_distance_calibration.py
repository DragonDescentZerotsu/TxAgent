from __future__ import annotations

import pandas as pd

from tools.chembl_tool.common.starling.build_pair_bucket_distance_calibration import (
    _build_calibration_entries,
    _bias_corrected_cramers_v_squared,
    _category_gate,
    _score_categorical_candidate,
    _select_categorical_variance_candidate,
)
from tools.chembl_tool.tasks.bbb_martins.build_starling_pair_bucket_transfer_policy import (
    BUILD_SPEC,
)
from tools.chembl_tool.tasks.bbb_martins.starling_schema import (
    RECORD_CONTRACT as BBB_RECORD_CONTRACT,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import (
    RECORD_CONTRACT as BIO_RECORD_CONTRACT,
)
from tools.chembl_tool.tasks.skin_reaction.starling_schema import RECORD_CONTRACT


def _binary_group(*, include_response: bool = True) -> pd.DataFrame:
    categories = ["no_response"] * 25
    ranks = [0] * 25
    if include_response:
        categories = ["no_response"] * 12 + ["response"] * 13
        ranks = [0] * 12 + [1] * 13
    return pd.DataFrame(
        {
            "canonical_category_id": categories,
            "canonical_category_rank": ranks,
        }
    )


def test_binary_gate_requires_both_declared_levels() -> None:
    complete = _category_gate(
        _binary_group(),
        record_contract=RECORD_CONTRACT,
        source_id="direct_skin_reaction",
        kind="binary",
        scale_id="single_subject_logit",
    )
    incomplete = _category_gate(
        _binary_group(include_response=False),
        record_contract=RECORD_CONTRACT,
        source_id="direct_skin_reaction",
        kind="binary",
        scale_id="single_subject_logit",
    )
    assert complete["valid"]
    assert incomplete == {"valid": False, "reason": "incomplete_binary_domain"}


def test_categorical_residual_gate_uses_bias_corrected_cramers_v_squared() -> None:
    group = _binary_group()
    group["assay_batch"] = ["batch_0"] * 12 + ["batch_1"] * 13
    score = _score_categorical_candidate(group, "assay_batch")
    assert score is not None
    assert score["cramers_v_squared"] >= 0.20
    assert score["variance_gate_flagged"]

    independent = pd.DataFrame([[10.0, 10.0], [10.0, 10.0]])
    assert _bias_corrected_cramers_v_squared(independent.to_numpy()) == 0.0


def test_categorical_residual_gate_excludes_the_selected_scale_inputs() -> None:
    group = _binary_group()
    group["substrate_status"] = ["not_substrate"] * 12 + ["substrate"] * 13
    group["transporter_or_enzyme"] = ["ABCB1"] * 25
    group["intestinal_site"] = None
    group["qualifying_conditions"] = None
    result = _select_categorical_variance_candidate(
        group,
        record_contract=BIO_RECORD_CONTRACT,
        source_id="fg",
        scale_id="fg_substrate_status_binary.v1",
    )
    assert result["excluded_controlled_input_fields"] == [
        "substrate_status",
        "transporter_or_enzyme",
    ]
    assert result["candidate_column"] == "__none__"


def test_distance_calibration_worker_count_preserves_exact_entries() -> None:
    rows = []
    for bucket_index in range(120):
        count = 25 if bucket_index == 0 else 2
        for record_index in range(count):
            rows.append(
                {
                    "pair_bucket_key": f"bucket-{bucket_index:03d}",
                    "source_id": "direct_bbb",
                    "measurement_kind": "continuous",
                    "canonical_measurement_scale_id": None,
                    "canonical_category_id": None,
                    "canonical_category_rank": None,
                    "finite_scalar_value": float(record_index + 1),
                    "canonical_record_id": (
                        f"record-{bucket_index:03d}-{record_index:03d}"
                    ),
                    "bbb_transport_label": None,
                    "qualifying_conditions": None,
                }
            )
    frame = pd.DataFrame(rows)
    serial = _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=25,
        workers=1,
    )
    parallel = _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=25,
        workers=2,
    )
    assert parallel == serial
