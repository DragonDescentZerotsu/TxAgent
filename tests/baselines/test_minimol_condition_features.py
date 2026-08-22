import json
from types import SimpleNamespace

import pytest
import torch

from baselines.minimol.condition_features import (
    CONDITION_ENCODING_VERSION,
    append_condition_one_hot,
    condition_feature_contract,
    condition_one_hot,
    condition_vocabulary,
)
from baselines.minimol.head_runtime import make_model
from baselines.minimol.run_bioavailability_ma import load_split


def test_train_derived_condition_one_hot_is_stable() -> None:
    vocabulary = condition_vocabulary(["fed", "null", "fed", "fasted"])
    encoded = condition_one_hot(["null", "fasted"], vocabulary)

    assert vocabulary == ["fasted", "fed", "null"]
    assert encoded.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="absent from train"):
        condition_one_hot(["cirrhosis"], vocabulary)


def test_append_condition_features_preserves_disabled_behavior() -> None:
    embeddings = torch.ones(2, 4)

    assert append_condition_one_hot(embeddings, None, None) is embeddings
    conditioned = append_condition_one_hot(
        embeddings, ["null", "fed"], ["fed", "null"]
    )
    assert conditioned.shape == (2, 6)
    assert torch.equal(conditioned[:, :4], embeddings)


def test_condition_feature_contract_records_input_shape() -> None:
    contract = condition_feature_contract(
        field="condition_group",
        vocabulary=["fed", "null"],
        molecule_embedding_dim=512,
    )

    assert contract["encoding"] == CONDITION_ENCODING_VERSION
    assert contract["condition_feature_dim"] == 2
    assert contract["model_input_dim"] == 514


def test_split_loader_reads_optional_condition_field(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"drug": "CC", "Y": 1, "condition_group": "fed"}) + "\n"
    )

    assert load_split(path).conditions is None
    assert load_split(path, condition_field="condition_group").conditions == ["fed"]


def test_head_runtime_accepts_condition_augmented_input() -> None:
    args = SimpleNamespace(
        hidden_dim=4,
        depth=3,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        warmup=0,
        epochs=2,
    )

    model, _, _, _ = make_model(args, torch.device("cpu"), input_dim=515)

    assert model.dense1.in_features == 515
    assert model.final_dense.in_features == 519
