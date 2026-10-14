"""Keep documentation and derived analyses in their canonical roots."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_human_documentation_is_centralized() -> None:
    remaining = []
    for source_root in ('baselines', 'data/processing', 'predict', 'tools'):
        for path in (ROOT / source_root).rglob('*.md'):
            relative = path.relative_to(ROOT)
            if path.name == 'AGENTS.md':
                continue
            if any((part in {'prompt_instructions', 'prompt_examples', 'reports'} for part in relative.parts)):
                continue
            if 'rejected_candidate_inventory' in str(relative):
                continue
            remaining.append(str(relative))
    assert remaining == []

def test_recent_analysis_defaults_use_their_canonical_roots() -> None:
    from optimization.semantic_buckets import inputs as artifacts, build as construction
    from analysis.prediction import progressive_pair_buckets
    assert construction.PROMPT_ROOT.is_relative_to(ROOT / 'optimization/semantic_buckets/prompts')
    assert artifacts.RELEASE_ROOT == ROOT / 'optimization/semantic_buckets/releases'
    assert progressive_pair_buckets.DEFAULT_OUTPUT.is_relative_to(ROOT / 'outputs' / 'analysis')

def test_retired_analysis_and_matrix_entrypoints_are_absent() -> None:
    retired = ('data/processing/evidence_library/relevance_bucket_diagnostic.py', 'data/processing/evidence_library/bioavailability_semantic_readout_v1.py', 'tools/chembl_tool/paper_experiments/analyze_progressive_pair_buckets.py', 'tools/chembl_tool/paper_experiments/run_flat_matrix.py', 'tools/chembl_tool/paper_experiments/run_progressive_matrix.py')
    assert not [path for path in retired if (ROOT / path).exists()]

def test_semantic_owner_has_no_root_alias():
    alias = ROOT / 'semantic_buckets'
    assert not alias.exists() and (not alias.is_symlink())
    from optimization.semantic_buckets import inputs as artifacts
    assert artifacts.RELEASE_ROOT == ROOT / 'optimization/semantic_buckets/releases'
