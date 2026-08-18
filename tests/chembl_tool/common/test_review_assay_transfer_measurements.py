from __future__ import annotations

import math

import pandas as pd

from tools.chembl_tool.common.starling.review_assay_transfer_measurements import (
    _positive_factor,
    _tail_candidates,
)


def test_tail_review_prefilters_buckets_below_minimum_support():
    rows = []
    for key, count in (("small", 24), ("tail", 25), ("flat", 25)):
        for index in range(count):
            value = 10.0 ** (index / 4) if key == "tail" else 1.0
            rows.append({"review_bucket_key": key, "review_value": value})

    candidates = _tail_candidates(pd.DataFrame(rows), minimum_records=25)

    assert [item["pair_bucket_key"] for item in candidates] == ["tail"]


def test_positive_factor_requires_explicit_finite_scientific_notation():
    def factor(text: str):
        return _positive_factor(
            pd.Series(
                {
                    "unit_notation_factor": None,
                    "unit_text": text,
                    "support_text": "",
                }
            )
        )

    assert factor("103.9 mM") is None
    assert factor("10^8159 mg/L") is None
    assert factor("10^6 mg/L") == 1e6
    assert math.isclose(factor("x10^23 mg/L"), 1e23)
