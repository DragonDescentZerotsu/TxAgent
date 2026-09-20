from pathlib import Path

import pytest

from data.processing.evidence_library.versions.v8.tasks.bbb_martins.artifact_store import (
    PROFILE,
)
from data.processing.paths import (
    LEGACY_GOLD_LABELS_ROOT,
    assert_canonical_library_output,
    evidence_library_root,
)
from data.processing.gold_labels.conditioned_benchmark import split_path


def test_active_release_paths_are_explicit_and_fail_closed() -> None:
    assert evidence_library_root("bbb_martins").name == "v10"
    assert evidence_library_root("bioavailability_ma").name == "v10"
    assert evidence_library_root("skin_reaction").name == "v10_main_universe_v5"
    assert (
        evidence_library_root("skin_reaction", "v10_main_universe_v5").name
        == "v10_main_universe_v5"
    )
    assert evidence_library_root("bbb_martins", "v7").name == "v7"
    assert evidence_library_root("ames", "v8").name == "v8"
    assert split_path("bbb_martins", "train").is_file()
    with pytest.raises(ValueError, match="unsupported active data task"):
        evidence_library_root("clintox")
    assert evidence_library_root("ames").name == "v9"


def test_data_root_contains_only_canonical_categories() -> None:
    root = Path("data")
    names = {
        path.name
        for path in root.iterdir()
        if not path.name.startswith(".") and path.name != "__pycache__"
    }
    assert names == {
        "__init__.py",
        "README.md",
        "artifacts",
        "caches",
        "evidence_libraries",
        "gold_labels",
        "legacy",
        "processing",
        "raw",
    }
    assert not any(path.is_symlink() for path in root.iterdir())
    assert (LEGACY_GOLD_LABELS_ROOT / "processed").is_dir()
    assert (LEGACY_GOLD_LABELS_ROOT / "processed_starling").is_dir()


def test_evidence_library_has_no_compatibility_links_or_imports() -> None:
    construction_root = Path("data/processing/evidence_library")
    assert not any(path.is_symlink() for path in construction_root.rglob("*"))
    assert not (construction_root / "starling").exists()
    assert not (construction_root / "prompts").exists()
    assert not (construction_root / "tasks").exists()

    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        task_root = Path("tools/chembl_tool/tasks") / task
        assert not any(path.is_symlink() for path in task_root.iterdir())
    assert not Path("tools/chembl_tool/common/starling").exists()
    assert not Path("tools/chembl_tool/tasks/ames").exists()

    forbidden = (
        "tools.chembl_tool.common." + "starling",
        "data.processing.evidence_library." + "starling",
        "data.processing.evidence_library." + "tasks",
        "shared." + "llm_prompts",
    )
    for root in (construction_root, Path("predict"), Path("tools"), Path("tests")):
        for path in root.rglob("*.py"):
            if (
                path.resolve() == Path(__file__).resolve()
                or path.name == "test_release_contract.py"
                or "__pycache__" in path.parts
            ):
                continue
            source = path.read_text(encoding="utf-8")
            for name in forbidden:
                assert name not in source, f"{path} imports removed namespace {name}"


def test_historical_library_aliases_are_read_only() -> None:
    old = Path(
        "outputs/chembl_tool/tasks/bbb_martins/evidence_library/"
        "starling_normalized_v8"
    )
    assert (old / "manifest.json").resolve() == (
        evidence_library_root("bbb_martins", "v8") / "manifest.json"
    ).resolve()
    with pytest.raises(ValueError, match="read-only aliases"):
        assert_canonical_library_output(old)


def test_bbb_v8_release_profile_targets_the_published_release() -> None:
    assert PROFILE.local_root == evidence_library_root("bbb_martins", "v8")
    assert PROFILE.stages == (
        "00_source",
        "01_cleaned",
        "02_canonicalized",
        "03_pair_buckets",
        "audits",
    )
    assert not (PROFILE.tracked_root / "manifest.json").exists()
    assert (PROFILE.local_root / "manifest.json").is_file()
    assert not (PROFILE.local_root / ".build-incomplete.json").exists()
    assert not list(PROFILE.tracked_root.parent.parent.rglob("release_receipt.json"))
    assert not (PROFILE.tracked_root.parent / "v7").exists()
