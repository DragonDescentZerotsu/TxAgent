"""Keep the canonical inference package independent of legacy tool modules."""

from __future__ import annotations

import ast
from pathlib import Path


PREDICT_ROOT = Path(__file__).resolve().parents[2] / "predict"


def _imported_modules(root: Path) -> list[tuple[str, str]]:
    imports: list[tuple[str, str]] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            imports.extend(
                (str(path.relative_to(PREDICT_ROOT)), module)
                for module in modules
                if module
            )
    return imports


def test_predict_does_not_import_legacy_chembl_tool_modules() -> None:
    violations = [
        path
        for path, module in _imported_modules(PREDICT_ROOT)
        if module.startswith("tools.chembl_tool")
    ]
    assert not violations, f"legacy inference imports: {sorted(set(violations))}"


def test_harness_universes_do_not_import_each_other() -> None:
    branch_imports = _imported_modules(PREDICT_ROOT / "harnesses" / "branches")
    progressive_imports = _imported_modules(PREDICT_ROOT / "harnesses" / "progressive")
    assert not [
        path
        for path, module in branch_imports
        if module.startswith("predict.harnesses.progressive")
    ]
    assert not [
        path
        for path, module in progressive_imports
        if module.startswith("predict.harnesses.branches")
    ]
