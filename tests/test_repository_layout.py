"""Keep documentation and derived analyses in their canonical roots."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_human_documentation_is_centralized() -> None:
    remaining = []
    for source_root in ("baselines", "data/processing", "predict", "tools"):
        for path in (ROOT / source_root).rglob("*.md"):
            relative = path.relative_to(ROOT)
            if path.name == "AGENTS.md":
                continue
            if any(part in {"prompt_instructions", "prompt_examples", "reports"} for part in relative.parts):
                continue
            if "rejected_candidate_inventory" in str(relative):
                continue
            remaining.append(str(relative))
    assert remaining == []


def test_recent_analysis_defaults_use_their_canonical_roots() -> None:
    from semantic_buckets import bioavailability_semantic_readout_v1
    from analysis.prediction import progressive_pair_buckets

    semantic_root = ROOT / "semantic_buckets/releases/bioavailability_ma/v10"
    assert bioavailability_semantic_readout_v1.DEFAULT_OUTPUT.is_relative_to(semantic_root)
    assert progressive_pair_buckets.DEFAULT_OUTPUT.is_relative_to(
        ROOT / "outputs" / "analysis"
    )


def test_retired_analysis_and_matrix_entrypoints_are_absent() -> None:
    retired = (
        "data/processing/evidence_library/relevance_bucket_diagnostic.py",
        "data/processing/evidence_library/bioavailability_semantic_readout_v1.py",
        "tools/chembl_tool/paper_experiments/analyze_progressive_pair_buckets.py",
        "tools/chembl_tool/paper_experiments/run_flat_matrix.py",
        "tools/chembl_tool/paper_experiments/run_progressive_matrix.py",
    )
    assert not [path for path in retired if (ROOT / path).exists()]
