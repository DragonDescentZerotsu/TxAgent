import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_MODULES = (
    "artifacts.py",
    "publication.py",
    "publish_weighted_policy_v2.py",
    "audit_policy_v2.py",
    "sliding_weight_assignment.py",
    "calibrate_weights_v6.py",
)


def test_maintained_functions_are_at_most_60_lines() -> None:
    violations = []
    for name in MAINTAINED_MODULES:
        path = ROOT / name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                if length > 60:
                    violations.append(f"{name}:{node.lineno} {node.name} ({length})")
    assert violations == []
