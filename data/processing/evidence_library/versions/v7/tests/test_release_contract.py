from __future__ import annotations

from pathlib import Path

from data.processing.evidence_library.shared.v1.normalization.task_policy import (
    StarlingTaskPolicy,
)
from data.processing.evidence_library.versions.v7.task_registry import import_task_module
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_policy import (
    POLICY as BBB_POLICY,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_policy import (
    POLICY as BIOAVAILABILITY_POLICY,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_policy import (
    POLICY as SKIN_POLICY,
)


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = RELEASE_ROOT.parents[1] / "shared" / "v1"


def test_scientific_assets_are_release_local() -> None:
    for policy in (BBB_POLICY, BIOAVAILABILITY_POLICY, SKIN_POLICY):
        for asset in policy.scientific_assets:
            resolved = Path(asset).resolve()
            assert resolved.is_file()
            if "data_processing" in resolved.parts or "prompts" in resolved.parts:
                assert resolved.is_relative_to(RELEASE_ROOT)


def test_release_source_does_not_import_another_construction_tree() -> None:
    forbidden = (
        "tools.chembl_tool.common.starling",
        "tools.chembl_tool.tasks.bbb_martins",
        "tools.chembl_tool.tasks.bioavailability_ma",
        "tools.chembl_tool.tasks.skin_reaction",
        "data.processing.evidence_library.versions.v8",
    )
    for path in RELEASE_ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in source, f"{path} imports {name}"


def test_task_registry_stays_inside_v7() -> None:
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        policy = import_task_module(task, "starling_policy").POLICY
        assert isinstance(policy, StarlingTaskPolicy)
        resolution = import_task_module(task, "starling_measurement_resolution")
        assert resolution.DEFAULT_MAPPING_PATH.is_file()
        assert Path(resolution.__file__).resolve().is_relative_to(RELEASE_ROOT)
    try:
        import_task_module("clintox", "starling_policy")
    except ValueError:
        pass
    else:
        raise AssertionError("V7 task registry accepted an undeclared task")


def test_shared_v1_has_no_release_or_legacy_task_imports() -> None:
    forbidden = (
        "data.processing.evidence_library.versions.",
        "tools.chembl_tool.common.starling",
        "tools.chembl_tool.tasks.",
    )
    for path in SHARED_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in source, f"{path} imports {name}"


def test_reviewed_regeneration_inputs_are_release_local() -> None:
    required = (
        "tasks/bioavailability_ma/data_processing/oral_study_context_extractions.json",
        "tasks/skin_reaction/data_processing/study_design_reviewed_mapping.json",
    )
    for relative in required:
        assert (RELEASE_ROOT / relative).is_file()
