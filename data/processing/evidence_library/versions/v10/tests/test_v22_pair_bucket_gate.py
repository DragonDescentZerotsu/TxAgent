from __future__ import annotations

import pandas as pd

from data.processing.evidence_library.versions.v10.build_pair_bucket_distance_calibration import (
    _build_calibration_entries,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.build_starling_pair_bucket_transfer_policy import (
    BUILD_SPEC,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.starling_schema import (
    RECORD_CONTRACT,
)


def test_v22_gate_requires_sixteen_records_without_a_molecule_minimum() -> None:
    rows = pd.DataFrame({
        "pair_bucket_key": ["bucket"] * 16,
        "source_id": ["direct_bbb"] * 16,
        "measurement_kind": ["continuous"] * 16,
        "canonical_measurement_scale_id": [None] * 16,
        "canonical_category_id": [None] * 16,
        "canonical_category_rank": [None] * 16,
        "finite_scalar_value": list(map(float, range(16))),
        "canonical_record_id": [f"record-{index}" for index in range(16)],
        "canonical_smiles": ["CC"] * 16,
    })

    entries = _build_calibration_entries(
        rows,
        spec=BUILD_SPEC,
        record_contract=RECORD_CONTRACT,
        minimum_samples=16,
        minimum_distinct_molecules=0,
        workers=1,
    )
    assert entries["bucket"]["calibration_valid"] is True
    assert entries["bucket"]["minimum_distinct_molecule_count"] == 0

    rejected = _build_calibration_entries(
        rows.iloc[:-1],
        spec=BUILD_SPEC,
        record_contract=RECORD_CONTRACT,
        minimum_samples=16,
        minimum_distinct_molecules=0,
        workers=1,
    )
    assert rejected["bucket"]["calibration_reason"] == "fewer_than_16_records"
