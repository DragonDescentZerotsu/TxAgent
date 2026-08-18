from __future__ import annotations

import pandas as pd

from tools.chembl_tool.common.starling.rebuild_assay_transfer_stage03 import (
    _restore_stage02_prebase,
)


def test_restore_uses_stage02_prebase_instead_of_final_stage03_tuple(tmp_path):
    stage03 = pd.DataFrame(
        [
            {
                "canonical_record_id": "record-1",
                "canonical_measurement_text": "-6",
                "canonical_unit_text": "log10(M)",
                "finite_scalar_value": -6.0,
                "variation_value": None,
            }
        ]
    )
    stage02 = pd.DataFrame(
        [
            {
                "canonical_record_id": "record-1",
                "assay_transfer_prebase_measurement_text": "1 ± 0.2",
                "assay_transfer_prebase_unit_text": "%/d",
                "assay_transfer_prebase_scalar_value": 1.0,
                "assay_transfer_pretransform_variation_value": 0.002,
                "assay_transfer_scale_factor": 0.01,
            }
        ]
    )
    path = tmp_path / "records.parquet"
    stage02.to_parquet(path, index=False)

    restored = _restore_stage02_prebase(stage03, path).iloc[0]

    assert restored["canonical_measurement_text"] == "1 ± 0.2"
    assert restored["canonical_unit_text"] == "%/d"
    assert restored["finite_scalar_value"] == 1.0
    assert restored["variation_value"] == 0.2
