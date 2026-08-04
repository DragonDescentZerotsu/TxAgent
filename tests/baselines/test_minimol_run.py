import torch

from pathlib import Path

from baselines.minimol.run_bioavailability_ma import _json_safe, _load_reusable_embeddings


def test_reusable_embeddings_are_indexed_by_exact_smiles(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        {
            "smiles": ["CCO", "CCC"],
            "labels": [0, 1],
            "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        cache_dir / "train.pt",
    )

    registry, files = _load_reusable_embeddings([cache_dir])

    assert files == [cache_dir / "train.pt"]
    assert set(registry) == {"CCO", "CCC"}
    assert torch.equal(registry["CCO"], torch.tensor([1.0, 0.0]))


def test_json_safe_serializes_repeated_cache_paths():
    assert _json_safe([Path("first"), Path("second")]) == ["first", "second"]
