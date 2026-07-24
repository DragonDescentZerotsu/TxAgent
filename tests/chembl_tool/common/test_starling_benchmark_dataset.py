import json

import pytest

from tools.chembl_tool.common.starling.benchmark_dataset import (
    LabeledSourceRecord,
    accepted,
    bemis_murcko_scaffold,
    build_benchmark_dataset,
    calculate_test_size,
    classify_interval,
    has_reported_text,
    parse_numeric_interval,
    scaffold_group_split,
    stratified_hash_split,
)
from tools.chembl_tool.tasks.bbb_martins.starling_benchmark import label_record as label_bbb
from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
    is_human_context,
    label_bioavailability_value,
)
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import (
    label_record as label_skin,
)


@pytest.mark.parametrize(
    ("text", "threshold", "expected"),
    [
        ("11 ± 3%", 20.0, 0),
        ("20%", 20.0, 1),
        ("10-22%", 20.0, None),
        ("≤ 19%", 20.0, 0),
        ("at least 20 percent", 20.0, 1),
        ("0.61 ± 0.16", 20.0, 1),
        ("0.20 ± 0.05", 20.0, None),
        ("-1.2", -1.0, 0),
        ("-1.0", -1.0, 1),
    ],
)
def test_numeric_interval_thresholds(text, threshold, expected):
    interval = parse_numeric_interval(text, fraction_to_percent=threshold == 20.0)
    assert interval is not None
    assert classify_interval(interval, threshold=threshold) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("low", 0),
        ("very poor", 0),
        ("good", 1),
        ("almost complete", 1),
        ("moderate", None),
        ("orally bioavailable", None),
        ("3-fold higher", None),
        ("increased by 25%", None),
        ("low to moderate (≤ 21%)", None),
        ("19%", 0),
        ("20%", 1),
        ("10-22%", None),
    ],
)
def test_bioavailability_value_policy(value, expected):
    label, _ = label_bioavailability_value(value)
    assert label == expected


def test_human_context_is_conservative():
    assert is_human_context("healthy human volunteers")
    assert is_human_context("adult patients (n=8)")
    assert not is_human_context("humanized mice")
    assert not is_human_context("beagle dogs")
    assert not is_human_context("species not stated")


def test_nullable_text_detection():
    assert has_reported_text("fed state")
    assert not has_reported_text(None)
    assert not has_reported_text(float("nan"))
    assert not has_reported_text("null")


def test_bbb_uses_explicit_labels_and_logbb_only():
    assert label_bbb({"bbb_permeability_label": "permeable"})[0] == 1
    assert label_bbb({"bbb_permeability_label": "good_penetration"})[0] == 1
    assert label_bbb({"bbb_permeability_label": "poor_penetration"})[0] == 0
    assert label_bbb({"bbb_transport_label": "efflux_substrate"})[0] is None
    assert label_bbb({"quant_metric": "log BB", "quant_value": "-0.9"})[0] == 1
    assert label_bbb({"quant_metric": "Papp", "quant_value": "10"})[0] is None
    assert (
        label_bbb(
            {
                "bbb_permeability_label": "permeable",
                "quant_metric": "logBB",
                "quant_value": "-1.2",
            }
        )[0]
        is None
    )


def test_skin_reaction_matches_sensitization_not_irritation():
    assert label_skin({"reaction_type": "sensitization", "outcome_label": "positive"})[0] == 1
    assert (
        label_skin(
            {
                "reaction_type": "allergic_contact_dermatitis_contact_allergy",
                "outcome_label": "negative",
            }
        )[0]
        == 0
    )
    assert label_skin({"reaction_type": "irritation", "outcome_label": "positive"})[0] is None
    assert label_skin({"reaction_type": "sensitization", "outcome_label": "inconclusive"})[0] is None


def test_parent_conflicts_are_dropped_and_split_is_exact(tmp_path):
    decisions = []
    smiles_values = (
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "c1ccoc1",
        "c1ccsc1",
        "C1CCNCC1",
    )
    for index, smiles in enumerate(smiles_values):
        decisions.append(
            accepted(
                LabeledSourceRecord(
                    smiles=smiles,
                    label=index % 2,
                    source_id="test",
                    source_record_id=str(index),
                )
            )
        )
    decisions.extend(
        [
            accepted(LabeledSourceRecord(smiles="CCN", label=0, source_id="a")),
            accepted(LabeledSourceRecord(smiles="CC[NH3+].[Cl-]", label=1, source_id="b")),
        ]
    )
    summary = build_benchmark_dataset(
        task_name="test",
        decisions=decisions,
        source_metadata={},
        output_dir=tmp_path,
        max_test_size=2,
        test_fraction=0.5,
        seed=7,
    )
    assert summary["n_conflicting_parent_groups"] == 1
    assert summary["test_size_policy"]["target_test_size"] == 2
    assert summary["splits"]["random"]["n_test"] == 2
    assert summary["splits"]["scaffold"]["n_test"] == 2
    assert sum(1 for _ in (tmp_path / "random" / "test.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "scaffold" / "test.jsonl").open()) == 2
    split_rows = [
        json.loads(line)
        for line in (tmp_path / "molecule_labels.jsonl").read_text().splitlines()
    ]
    assert all(set(row["split_assignments"]) == {"random", "scaffold"} for row in split_rows)
    assert sum(1 for _ in (tmp_path / "random" / "test_molecule_labels.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "random" / "train_molecule_labels.jsonl").open()) == 4
    conflict = json.loads((tmp_path / "conflicting_molecules.jsonl").read_text().splitlines()[0])
    assert conflict["drop_reason"] == "conflicting_parent_level_labels"


@pytest.mark.parametrize(
    ("n_molecules", "expected"),
    [
        (100, 20),
        (1_862, 372),
        (1_900, 380),
        (17_893, 500),
    ],
)
def test_test_size_policy(n_molecules, expected):
    assert calculate_test_size(n_molecules) == expected


def test_hash_split_is_reproducible():
    rows = [
        {"molecule_identity_key": f"k{index}", "Y": index % 2}
        for index in range(20)
    ]
    first = stratified_hash_split(rows, test_size=6, seed=11)
    second = stratified_hash_split(rows, test_size=6, seed=11)
    assert first == second
    assert {row["Y"] for row in first[1]} == {0, 1}


def test_scaffold_split_is_reproducible_and_disjoint():
    smiles_values = [
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "C1CCCCC1",
        "OC1CCCCC1",
        "c1ccoc1",
        "c1ccsc1",
    ]
    rows = [
        {
            "drug": smiles,
            "molecule_identity_key": f"k{index}",
            "Y": index % 2,
            "bemis_murcko_scaffold": bemis_murcko_scaffold(smiles),
        }
        for index, smiles in enumerate(smiles_values)
    ]
    first = scaffold_group_split(rows, test_size=4, seed=11)
    second = scaffold_group_split(rows, test_size=4, seed=11)
    assert first == second
    assert len(first[1]) == 4
    train_scaffolds = {row["bemis_murcko_scaffold"] for row in first[0]}
    test_scaffolds = {row["bemis_murcko_scaffold"] for row in first[1]}
    assert not (train_scaffolds & test_scaffolds)
