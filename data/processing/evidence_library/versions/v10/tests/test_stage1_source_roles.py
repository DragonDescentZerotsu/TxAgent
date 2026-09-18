"""Executable checks for the declared Stage-1 source-column roles."""

import pytest

from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
    TaskConfig,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    source_role_contract,
)


@pytest.mark.parametrize(
    "task", ["ames", "dili", "carcinogens", "skin_reaction", "bioavailability_ma"]
)
def test_every_new_task_source_declares_one_complete_role_set(task: str) -> None:
    config = TaskConfig(task)
    manifest = config.source_role_manifest()
    assert set(manifest["sources"]) == set(config.SOURCE_IDS)
    for roles in manifest["sources"].values():
        assert roles["endpoint"]["kind"] in {"field", "constant"}
        assert roles["measurement"]["kind"] == "field"
        assert roles["unit"]["kind"] in {"field", "constant", "exception"}


def test_missing_unit_declaration_fails_closed() -> None:
    class Source:
        source_columns = ("endpoint", "measurement")
        endpoint_field = "endpoint"
        endpoint_constant = None
        measurement_field = "measurement"
        unit_field = None
        unit_constant = None

    with pytest.raises(ValueError, match="must declare one unit"):
        source_role_contract({"source": Source()}, {})
