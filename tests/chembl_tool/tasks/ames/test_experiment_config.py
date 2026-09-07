from pathlib import Path

import pandas as pd

from tools.chembl_tool.paper_experiments.build_assay_family_catalog import (
    TASKS,
    build_catalog,
)
from tools.chembl_tool.tasks.ames import experiment_config as config
from tools.chembl_tool.tasks.ames.source_contract import (
    DAMAGE,
    DIRECT,
    MECHANISM,
    MUTATION,
    NEAR,
    VERSION,
)


def test_ames_direct_view_excludes_nonvoters_and_indirect_evidence():
    groups = {DIRECT, NEAR, MUTATION, DAMAGE, MECHANISM}
    source = config.get_source_config("starling")
    assert [spec.resolve(groups) for spec in source.direct_groups] == [(DIRECT,)]
    contract = config.get_progressive_task_contract()
    assert contract.task == "ames"
    assert contract.label_scope == VERSION
    assert contract.prediction_field == "ames_prediction"
    assert contract.prediction_values == {"positive", "negative"}


def test_ames_catalog_keeps_nonvoters_at_l2_in_a_shared_bacterial_assay(tmp_path):
    spec = TASKS["ames"]
    assert spec["records"] == "data/starling_data/ames/canonical_v1/records.parquet"
    assert Path(spec["output_root"]) / spec["output_name"] == Path(
        "outputs/paper/starling_conditioned_assay_family_curve_v1/family_catalogs/ames"
    )
    path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "group_id": group,
                "canonical_assay_context": "shared bacterial assay"
                if group in {DIRECT, NEAR}
                else group,
                "canonical_endpoint_name": "mutation",
                "retrieval_eligible": True,
            }
            for group in (DIRECT, NEAR, MUTATION, DAMAGE, MECHANISM)
        ]
    ).to_parquet(path, index=False)

    rows, manifest = build_catalog("ames", path, config_name=spec["config_name"])
    assert [level["endpoint_group"] for level in manifest["levels"]] == [
        "direct_ames",
        "nonvoter_ames_outcome",
        "other_genetic_damage",
        "dna_damage_response",
        "genotoxicity_mechanisms",
    ]
    assert [level["source_groups"] for level in manifest["levels"]] == [
        [DIRECT],
        [NEAR],
        [MUTATION],
        [DAMAGE],
        [MECHANISM],
    ]
    assert set(config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS) == {1, 2, 3, 4, 5}
    bacterial = next(
        row for row in rows if row["assay_context"] == "shared bacterial assay"
    )
    assert bacterial["first_level"] == 1
    assert bacterial["family_levels"] == [1, 2]
    assert manifest["family_assignment_unit"] == "source_record"
