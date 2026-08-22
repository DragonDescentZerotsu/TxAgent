import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.minimol import starling_table2_data
from baselines.minimol.head_runtime import (
    EmbeddingDataset,
    evaluate_loss,
    make_model,
    predict_scores,
    train_one_epoch,
)
from baselines.minimol.merge_starling_table2_shards import merge_shards
from baselines.minimol.run_starling_table2 import (
    _parse_repetition_indices,
    cantor_pairing,
    metric_value,
    scaffold_split_indices,
    selected_repetition_indices,
)
from baselines.minimol.starling_table2_data import (
    TASKS,
    filter_rdkit_invalid,
    extraction_weights,
    load_task_tables,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_oral_loader_preserves_soft_train_targets_and_validates_hard_test(tmp_path):
    _write_csv(
        tmp_path / "oral_ba_tdc_plus_lit_train.csv",
        ["canonical_smiles", "label", "source", "n_extractions"],
        [
            {"canonical_smiles": "CCO", "label": 0, "source": "tdc", "n_extractions": ""},
            {"canonical_smiles": "CCN", "label": 1, "source": "tdc", "n_extractions": ""},
            {"canonical_smiles": "CCC", "label": 0.49, "source": "lit", "n_extractions": 2},
        ],
    )
    _write_csv(
        tmp_path / "oral_ba_tdc_test.csv",
        ["canonical_smiles", "label"],
        [
            {"canonical_smiles": "C1CC1", "label": 0},
            {"canonical_smiles": "c1ccccc1", "label": 1},
        ],
    )
    _write_csv(
        tmp_path / "oral_ba_lit_test.csv",
        ["canonical_smiles", "label", "n_extractions", "label_hard"],
        [
            {"canonical_smiles": "CO", "label": 0.19, "n_extractions": 1, "label_hard": 0},
            {"canonical_smiles": "CN", "label": 0.20, "n_extractions": 1, "label_hard": 1},
        ],
    )

    soft = load_task_tables(tmp_path, TASKS["oral_bioavailability"])
    hard = load_task_tables(
        tmp_path,
        TASKS["oral_bioavailability"],
        train_label_policy="hard",
    )

    assert soft.tdc_train.targets == [0.0, 1.0]
    assert soft.augmented_train.targets == [0.0, 1.0, 0.49]
    assert hard.augmented_train.targets == [0.0, 1.0, 1.0]
    assert soft.literature_test.targets == [0.0, 1.0]
    assert soft.literature_test.raw_labels == [0.19, 0.20]
    assert extraction_weights(soft.augmented_train, "sqrt") == pytest.approx(
        [1.0, 1.0, np.sqrt(2)]
    )


def test_weighted_logit_loss_uses_sample_weights():
    class FirstFeature(nn.Module):
        def forward(self, inputs):
            return inputs[:, :1]

    embeddings = torch.zeros(2, 512)
    embeddings[1, 0] = 2.0
    loader = DataLoader(
        EmbeddingDataset(embeddings, [0.0, 1.0], [1.0, 9.0]),
        batch_size=1,
    )
    observed = evaluate_loss(
        FirstFeature(),
        loader,
        nn.BCEWithLogitsLoss(reduction="none"),
        torch.device("cpu"),
    )
    expected = (
        nn.functional.binary_cross_entropy_with_logits(torch.tensor(0.0), torch.tensor(0.0))
        + 9
        * nn.functional.binary_cross_entropy_with_logits(torch.tensor(2.0), torch.tensor(1.0))
    ) / 10
    assert observed == pytest.approx(float(expected))


def test_loader_rejects_contradictory_label_hard(tmp_path):
    _write_csv(
        tmp_path / "bbb_tdc_plus_lit_train.csv",
        ["canonical_smiles", "label", "source", "n_extractions"],
        [
            {"canonical_smiles": "CCO", "label": 0, "source": "tdc", "n_extractions": ""},
            {"canonical_smiles": "CCN", "label": 1, "source": "lit", "n_extractions": 1},
        ],
    )
    _write_csv(
        tmp_path / "bbb_tdc_test.csv",
        ["canonical_smiles", "label"],
        [
            {"canonical_smiles": "C1CC1", "label": 0},
            {"canonical_smiles": "c1ccccc1", "label": 1},
        ],
    )
    _write_csv(
        tmp_path / "bbb_lit_test.csv",
        ["canonical_smiles", "label", "n_extractions", "label_hard"],
        [
            {"canonical_smiles": "CO", "label": 0.5, "n_extractions": 2, "label_hard": 0},
            {"canonical_smiles": "CN", "label": 1, "n_extractions": 1, "label_hard": 1},
        ],
    )

    with pytest.raises(ValueError, match="label_hard contradicts"):
        load_task_tables(tmp_path, TASKS["bbb"])


def test_rdkit_invalid_filter_is_audited_without_changing_test_rows(tmp_path, monkeypatch):
    invalid = "C[CH3+]OC(=O)c1ccc(N)cc1"
    _write_csv(
        tmp_path / "bbb_tdc_plus_lit_train.csv",
        ["canonical_smiles", "label", "source", "n_extractions"],
        [
            {"canonical_smiles": "CCO", "label": 0, "source": "tdc", "n_extractions": ""},
            {"canonical_smiles": invalid, "label": 1, "source": "lit", "n_extractions": 1},
        ],
    )
    _write_csv(
        tmp_path / "bbb_tdc_test.csv",
        ["canonical_smiles", "label"],
        [
            {"canonical_smiles": "C1CC1", "label": 0},
            {"canonical_smiles": "c1ccccc1", "label": 1},
        ],
    )
    _write_csv(
        tmp_path / "bbb_lit_test.csv",
        ["canonical_smiles", "label", "n_extractions", "label_hard"],
        [
            {"canonical_smiles": "CO", "label": 0, "n_extractions": 1, "label_hard": 0},
            {"canonical_smiles": "CN", "label": 1, "n_extractions": 1, "label_hard": 1},
        ],
    )

    raw = load_task_tables(tmp_path, TASKS["bbb"])
    parse_smiles = starling_table2_data.Chem.MolFromSmiles
    monkeypatch.setattr(
        starling_table2_data.Chem,
        "MolFromSmiles",
        lambda smiles: None if smiles == invalid else parse_smiles(smiles),
    )
    filtered, exclusions = filter_rdkit_invalid(raw)

    assert len(raw.augmented_train.smiles) == 2
    assert filtered.augmented_train.smiles == ["CCO"]
    assert filtered.tdc_train.smiles == ["CCO"]
    assert filtered.tdc_test.smiles == raw.tdc_test.smiles
    assert filtered.literature_test.smiles == raw.literature_test.smiles
    assert exclusions == [
        {
            "canonical_smiles": invalid,
            "reason": "rdkit_sanitized_parse_failed",
            "occurrences": [
                {
                    "surface": "augmented_train",
                    "row_index": 1,
                    "raw_label": 1.0,
                    "source": "lit",
                }
            ],
        }
    ]


def test_scaffold_split_is_deterministic_disjoint_and_complete():
    smiles = ["c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CC1CCCCC1", "CCO", "CCN"]
    first = scaffold_split_indices(smiles, seed=17)
    second = scaffold_split_indices(smiles, seed=17)

    assert first == second
    train, valid, counts = first
    assert sorted(train + valid) == list(range(len(smiles)))
    assert not set(train) & set(valid)
    assert counts["n_train_scaffolds"] + counts["n_valid_scaffolds"] == 3


def test_cantor_seeds_match_upstream_pairing_contract():
    assert cantor_pairing(1, 6) == 34
    assert len({cantor_pairing(rep, member) for rep in range(1, 6) for member in range(6, 11)}) == 25


def test_repetition_shards_preserve_global_indices():
    assert _parse_repetition_indices("5, 2") == [5, 2]
    args = SimpleNamespace(repetition_indices=[5, 2], repetitions=5)
    assert selected_repetition_indices(args) == [2, 5]
    with pytest.raises(ValueError, match="cannot exceed"):
        selected_repetition_indices(SimpleNamespace(repetition_indices=[6], repetitions=5))


def test_shared_head_runtime_supports_regression():
    args = SimpleNamespace(
        hidden_dim=8,
        depth=3,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        warmup=0,
        epochs=2,
    )
    device = torch.device("cpu")
    embeddings = torch.randn(8, 512)
    labels = np.linspace(-1.0, 1.0, 8).tolist()
    loader = DataLoader(EmbeddingDataset(embeddings, labels), batch_size=4)
    model, optimizer, scheduler, loss_fn = make_model(args, device, task_type="regression")

    before = evaluate_loss(model, loader, loss_fn, device)
    train_one_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        loss_fn,
        device,
    )
    scores = predict_scores(model, loader, device, task_type="regression")

    assert np.isfinite(before)
    assert scores.shape == (8,)
    assert metric_value(TASKS["ld50"], labels, scores) >= 0


def test_weighted_auroc_uses_evaluation_sample_weights():
    task = TASKS["oral_bioavailability"]
    targets = [0.0, 0.0, 1.0, 1.0]
    scores = np.asarray([0.1, 0.9, 0.8, 0.7])

    unweighted = metric_value(task, targets, scores)
    weighted = metric_value(task, targets, scores, [9.0, 1.0, 1.0, 9.0])

    assert unweighted == pytest.approx(0.5)
    assert weighted == pytest.approx(0.9)


def test_merge_repetition_shards_requires_and_combines_exact_coverage(tmp_path):
    metrics_paths = []
    for repetition, base_score in ((1, 0.6), (2, 0.8)):
        shard = tmp_path / f"shard_{repetition}"
        shard.mkdir()
        conditions = {}
        for condition, offset in (("tdc_only", 0.0), ("augmented", 0.1)):
            conditions[condition] = {
                "summary": {},
                "repetitions": [
                    {
                        "repetition": repetition,
                        "members": [],
                        "metrics": {
                            "tdc_test": base_score + offset,
                            "literature_test": base_score + offset - 0.05,
                        },
                    }
                ],
            }
            for evaluation in ("tdc_test", "literature_test"):
                rows = [
                    {
                        "canonical_smiles": "CCO",
                        "target": 1.0,
                        "raw_label": 1.0,
                        "score": base_score + offset,
                    }
                ]
                (shard / f"{condition}_{evaluation}_predictions.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
        payload = {
            "type": "starling_table2_minimol_reproduction.v1",
            "task": {"slug": "oral_bioavailability"},
            "args": {
                "train_label_policy": "paper_soft",
                "extraction_weight": "sqrt",
                "evaluation_extraction_weight": "sqrt",
                "epochs": 25,
                "repetitions": 2,
                "ensemble_size": 5,
                "dropout": 0.1,
                "weight_decay": 1e-4,
                "warmup": 5,
                "train_batch_size": 32,
                "eval_batch_size": 128,
            },
            "data": {"files": {}},
            "minimol": {"model_version": "minimol_v1"},
            "embeddings": {},
            "conditions": conditions,
            "paper_comparison": None,
        }
        metrics_path = shard / "metrics.json"
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")
        metrics_paths.append(metrics_path)

    merged = merge_shards(metrics_paths, tmp_path / "merged")

    assert merged["repetition_indices"] == [1, 2]
    assert merged["conditions"]["tdc_only"]["summary"]["tdc_test"]["mean"] == pytest.approx(0.7)
    prediction = json.loads(
        (tmp_path / "merged/tdc_only_tdc_test_predictions.jsonl").read_text(encoding="utf-8")
    )
    assert prediction["score"] == pytest.approx(0.7)
