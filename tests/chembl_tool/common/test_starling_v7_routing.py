from __future__ import annotations

import json

from tools.chembl_tool.tasks.bbb_martins.starling_policy import (
    POLICY as BBB_POLICY,
    endpoint_inventory as bbb_endpoint_inventory,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_transfer_policy import (
    resolve_output_dir as resolve_bioavailability_output_dir,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import (
    RECORD_CONTRACT as BIOAVAILABILITY_RECORD_CONTRACT,
)
from tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_transfer_policy import (
    resolve_output_dir as resolve_skin_output_dir,
)
from tools.chembl_tool.tasks.skin_reaction.starling_schema import (
    RECORD_CONTRACT as SKIN_RECORD_CONTRACT,
)
from tools.chembl_tool.tasks.skin_reaction.starling_policy import POLICY as SKIN_POLICY


def test_standalone_calibration_defaults_follow_contract_version(tmp_path) -> None:
    for resolver, contract in (
        (resolve_bioavailability_output_dir, BIOAVAILABILITY_RECORD_CONTRACT),
        (resolve_skin_output_dir, SKIN_RECORD_CONTRACT),
    ):
        root = tmp_path / contract.task_id
        metadata = root / "04_pair_buckets/pair_bucket_metadata.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            json.dumps({"contract_version": contract.version}), encoding="utf-8"
        )
        assert resolver(metadata, explicit=None) == root / "05_distance_calibration"

        metadata.write_text(
            json.dumps({"contract_version": "frozen-v6"}), encoding="utf-8"
        )
        assert resolver(metadata, explicit=None) == root / "05_assay_transfer_policy"
        assert resolver(metadata, explicit=root / "custom") == root / "custom"


def test_bbb_endpoint_inventory_points_to_v7_registry() -> None:
    inventory = bbb_endpoint_inventory(
        "passive_permeability", ["Papp"], strict=False
    )
    assert inventory["runtime_decision_registry"] == (
        "02_canonicalized/endpoint_registry.json"
    )


def test_task_policies_declare_complete_scientific_asset_inventories() -> None:
    assert [path.name for path in BBB_POLICY.scientific_assets] == [
        "measurement_semantics.v1.json",
        "reference_semantics_prompts.json",
    ]
    assert [path.name for path in SKIN_POLICY.scientific_assets] == [
        "measurement_semantics.json",
        "reference_semantics_prompts.json",
        "globally_reconciled_auxiliary_value_mapping.json",
        "auxiliary_value_prompts.json",
    ]
