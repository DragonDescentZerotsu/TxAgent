import json

import pytest

from tools.chembl_tool.common.starling.benchmark_dataset import (
    LabeledSourceRecord,
    accepted,
    bemis_murcko_scaffold,
    build_benchmark_dataset,
    calculate_eval_size,
    calculate_test_size,
    classify_interval,
    has_reported_text,
    parse_numeric_interval,
    scaffold_group_split,
    scaffold_group_three_way_split,
    stratified_hash_split,
    stratified_hash_three_way_split,
)
from tools.chembl_tool.common.starling.heldout_index import (
    filter_heldout_evidence_rows,
    identity_key,
    load_heldout_identity_keys,
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
        ("64% increase", None),
        ("reduced by 22%", None),
        ("decreased from 100 to 50%", None),
        ("30 to 50% less than that of the conventional formulation", None),
        ("23% decreased systemic drug exposure", None),
        ("about 25% greater", None),
        ("low to moderate (≤ 21%)", None),
        ("19%", 0),
        ("20%", 1),
        ("greater than 80%", 1),
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


def test_record_majority_and_three_way_splits_are_exact(tmp_path):
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
        max_eval_size=2,
        valid_fraction=0.34,
        test_fraction=0.34,
        agreement_threshold=0.70,
        seed=7,
    )
    assert summary["n_parent_groups_with_label_conflict"] == 1
    assert summary["n_rejected_parent_groups"] == 1
    assert summary["split_size_policy"]["target_valid_size"] == 2
    assert summary["split_size_policy"]["target_test_size"] == 2
    assert summary["splits"]["random"]["n_valid"] == 2
    assert summary["splits"]["random"]["n_test"] == 2
    assert summary["splits"]["scaffold"]["n_valid"] == 2
    assert summary["splits"]["scaffold"]["n_test"] == 2
    assert sum(1 for _ in (tmp_path / "random" / "valid.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "random" / "test.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "scaffold" / "valid.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "scaffold" / "test.jsonl").open()) == 2
    split_rows = [
        json.loads(line)
        for line in (tmp_path / "molecule_labels.jsonl").read_text().splitlines()
    ]
    assert all(set(row["split_assignments"]) == {"random", "scaffold"} for row in split_rows)
    assert sum(1 for _ in (tmp_path / "random" / "test_molecule_labels.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "random" / "valid_molecule_labels.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "random" / "train_molecule_labels.jsonl").open()) == 2
    assert sum(1 for _ in (tmp_path / "random" / "heldout_molecule_labels.jsonl").open()) == 4
    conflict = json.loads((tmp_path / "conflicting_molecules.jsonl").read_text().splitlines()[0])
    assert conflict["drop_reason"] == "parent_record_label_tie"


def test_seventy_percent_record_majority_recovers_three_to_one_parent(tmp_path):
    decisions = [
        accepted(LabeledSourceRecord(smiles="CCN", label=1, source_id="source")),
        accepted(LabeledSourceRecord(smiles="CC[NH3+].[Cl-]", label=1, source_id="source")),
        accepted(LabeledSourceRecord(smiles="CCN", label=1, source_id="source")),
        accepted(LabeledSourceRecord(smiles="CCN", label=0, source_id="source")),
    ]
    for index, smiles in enumerate(("c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccoc1") * 3):
        decisions.append(
            accepted(
                LabeledSourceRecord(
                    smiles=smiles + ("" if index < 4 else "C"),
                    label=index % 2,
                    source_id="source",
                )
            )
        )

    summary = build_benchmark_dataset(
        task_name="test",
        decisions=decisions,
        source_metadata={},
        output_dir=tmp_path,
        max_eval_size=2,
        valid_fraction=0.25,
        test_fraction=0.25,
        agreement_threshold=0.70,
    )

    recovered = [
        json.loads(line)
        for line in (tmp_path / "molecule_labels.jsonl").read_text().splitlines()
        if json.loads(line)["molecule_identity_key"] == "QUSNBJAOOMFDIB-UHFFFAOYSA-N"
    ]
    assert len(recovered) == 1
    assert recovered[0]["Y"] == 1
    assert recovered[0]["agreement_fraction"] == 0.75
    assert recovered[0]["label_decision"] == "accepted_record_majority"
    assert summary["n_parent_groups_recovered_by_majority"] >= 1


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


@pytest.mark.parametrize(
    ("n_molecules", "expected"),
    [(100, 10), (2_140, 214), (2_456, 245), (19_425, 500)],
)
def test_eval_size_policy(n_molecules, expected):
    assert calculate_eval_size(n_molecules) == expected


def test_hash_split_is_reproducible():
    rows = [
        {"molecule_identity_key": f"k{index}", "Y": index % 2}
        for index in range(20)
    ]
    first = stratified_hash_split(rows, test_size=6, seed=11)
    second = stratified_hash_split(rows, test_size=6, seed=11)
    assert first == second
    assert {row["Y"] for row in first[1]} == {0, 1}


def test_hash_three_way_split_is_reproducible_and_disjoint():
    rows = [
        {"molecule_identity_key": f"k{index}", "Y": index % 2}
        for index in range(30)
    ]
    first = stratified_hash_three_way_split(
        rows, valid_size=5, test_size=5, seed=11
    )
    second = stratified_hash_three_way_split(
        rows, valid_size=5, test_size=5, seed=11
    )
    assert first == second
    assert [len(part) for part in first] == [20, 5, 5]
    key_sets = [{row["molecule_identity_key"] for row in part} for part in first]
    assert not (key_sets[0] & key_sets[1] | key_sets[0] & key_sets[2] | key_sets[1] & key_sets[2])


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


def test_scaffold_three_way_split_has_pairwise_scaffold_disjointness():
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
    train, valid, test = scaffold_group_three_way_split(
        rows, valid_size=2, test_size=2, seed=11
    )
    scaffold_sets = [
        {row["bemis_murcko_scaffold"] for row in part}
        for part in (train, valid, test)
    ]
    assert not (
        scaffold_sets[0] & scaffold_sets[1]
        or scaffold_sets[0] & scaffold_sets[2]
        or scaffold_sets[1] & scaffold_sets[2]
    )


def test_heldout_filter_removes_parent_equivalent_evidence(tmp_path):
    heldout_path = tmp_path / "test_molecule_labels.jsonl"
    heldout = {
        "drug": "CCO",
        "molecule_identity_key": identity_key({"canonical_smiles": "CCO"}),
        "molecule_identity": {
            "normalizer_version": "rdkit_fragment_parent.v1",
        },
        "Y": 1,
    }
    heldout_path.write_text(json.dumps(heldout) + "\n", encoding="utf-8")
    heldout_keys = load_heldout_identity_keys(heldout_path)

    kept, stats = filter_heldout_evidence_rows(
        [
            {"canonical_smiles": "CCO"},
            {"canonical_smiles": "CCO.[Na+]"},
            {"canonical_smiles": "CCN"},
            {"canonical_smiles": ""},
        ],
        heldout_keys,
    )

    assert [row["canonical_smiles"] for row in kept] == ["CCN"]
    assert stats["n_excluded_evidence_rows"] == 3
    assert stats["n_excluded_unresolved_evidence_rows"] == 1
    assert stats["zero_parent_overlap"] is True

def test_heldout_identity_loader_recomputes_and_validates_stored_key(tmp_path):
    heldout_path = tmp_path / "test_molecule_labels.jsonl"
    heldout_path.write_text(
        json.dumps(
            {
                "drug": "CCO",
                "molecule_identity_key": "incorrect-key",
                "molecule_identity": {
                    "normalizer_version": "rdkit_fragment_parent.v1",
                },
                "Y": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stores parent identity"):
        load_heldout_identity_keys(heldout_path)
