from __future__ import annotations

from pathlib import Path

from data.processing.evidence_library.shared.v2.normalization.task_policy import (
    StarlingTaskPolicy,
)
from data.processing.evidence_library.versions.v9.task_registry import import_task_module
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_policy import (
    BASE_EXACT_UNIT_MAPPING,
    POLICY as BBB_POLICY,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.starling_policy import (
    POLICY as BIOAVAILABILITY_POLICY,
)
from data.processing.evidence_library.versions.v9.tasks.ames.starling_policy import (
    POLICY as AMES_POLICY,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_policy import (
    POLICY as SKIN_POLICY,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_auxiliary_metadata import (
    AuxiliaryMetadataAttacher as SkinAuxiliaryMetadataAttacher,
)
from data.processing.paths import evidence_library_root


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = RELEASE_ROOT.parents[1] / "shared" / "v2"


def test_scientific_assets_are_release_local() -> None:
    for policy in (BBB_POLICY, BIOAVAILABILITY_POLICY, SKIN_POLICY, AMES_POLICY):
        for asset in policy.scientific_assets:
            resolved = Path(asset).resolve()
            assert resolved.is_file()
            if "data_processing" in resolved.parts or "prompts" in resolved.parts:
                assert resolved.is_relative_to(RELEASE_ROOT)
    assert BASE_EXACT_UNIT_MAPPING.resolve().is_relative_to(RELEASE_ROOT)
    assert BASE_EXACT_UNIT_MAPPING.is_file()


def test_release_source_does_not_import_another_construction_tree() -> None:
    forbidden = (
        "tools.chembl_tool.common.starling",
        "tools.chembl_tool.tasks.ames",
        "tools.chembl_tool.tasks.bbb_martins",
        "tools.chembl_tool.tasks.bioavailability_ma",
        "tools.chembl_tool.tasks.skin_reaction",
        "data.processing.evidence_library.versions.v7",
        "data.processing.evidence_library.versions.v8",
    )
    for path in RELEASE_ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in source, f"{path} imports {name}"


def test_task_registry_stays_inside_v9() -> None:
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        policy = import_task_module(task, "starling_policy").POLICY
        assert isinstance(policy, StarlingTaskPolicy)
        resolution = import_task_module(task, "starling_measurement_resolution")
        assert resolution.DEFAULT_MAPPING_PATH.is_file()
        assert Path(resolution.__file__).resolve().is_relative_to(RELEASE_ROOT)
    assert import_task_module("ames", "starling_policy").POLICY is AMES_POLICY
    assert Path(
        import_task_module("ames", "starling_measurement_resolution").__file__
    ).resolve().is_relative_to(RELEASE_ROOT)
    try:
        import_task_module("ames", "starling_reference_semantics")
    except ValueError:
        pass
    else:
        raise AssertionError("V9 Ames registry accepted an unimplemented module")
    try:
        import_task_module("clintox", "starling_policy")
    except ValueError:
        pass
    else:
        raise AssertionError("V9 task registry accepted an undeclared task")


def test_shared_v2_has_no_release_or_legacy_task_imports() -> None:
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


def test_v9_policies_write_only_to_v9_release_roots() -> None:
    for task, policy in (
        ("bbb_martins", BBB_POLICY),
        ("bioavailability_ma", BIOAVAILABILITY_POLICY),
        ("skin_reaction", SKIN_POLICY),
    ):
        assert Path(policy.default_out_dir) == evidence_library_root(task, "v9")


def test_skin_reviewed_auxiliary_mapping_is_self_contained() -> None:
    assert SkinAuxiliaryMetadataAttacher().manifest()["mapping_version"]
