import numpy as np

from tools.chembl_tool.paper_experiments.analyze_starling_direct_significance import (
    _add_holm,
    _macro_f1,
    _paired_comparison,
)


def test_macro_f1_matches_balanced_perfect_and_all_positive_cases():
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    assert _macro_f1(y, y) == 1.0
    assert _macro_f1(y, np.ones(4, dtype=np.int8)) == 1 / 3


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    rows = [{"p": value} for value in (0.04, 0.01, 0.03)]
    _add_holm(rows, "p", "holm")
    assert [row["holm"] for row in rows] == [0.06, 0.03, 0.06]


def test_paired_comparison_reports_direction_and_complete_correctness_table():
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    result = _paired_comparison(
        y,
        y,
        np.asarray([1, 1, 1, 1], dtype=np.int8),
        permutation_replicates=100,
        bootstrap_replicates=100,
        permutation_seed=3,
        bootstrap_seed=5,
    )
    assert result["delta_macro_f1_agent_minus_baseline"] > 0
    assert result["agent_only_correct"] == 2
    assert result["baseline_only_correct"] == 0
    assert result["both_correct"] + result["both_wrong"] + 2 == len(y)
