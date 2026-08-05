import torch
from types import SimpleNamespace

from tools.chembl_tool.paper_experiments.build_minimol_retrieval_features import (
    _embed_with_audited_fallback,
    _load_unique_indices,
    _minimol_fallback_smiles,
)


PROBLEMATIC_MIXTURE = (
    "F[PH](F)(F)(F)(F)F."
    "N=c1ccc2c(-c3ccccc3)c3ccc(N)cc3sc-2c1"
)


class _FakeMiniMol:
    def __call__(self, smiles):
        if PROBLEMATIC_MIXTURE in smiles:
            raise AttributeError("'str' object has no attribute 'stores'")
        return [torch.tensor([float(len(item)), 1.0]) for item in smiles]


def test_invalid_whole_record_falls_back_to_largest_parseable_fragment():
    fallback, reason = _minimol_fallback_smiles(PROBLEMATIC_MIXTURE)

    assert fallback == "N=c1ccc2c(-c3ccccc3)c3ccc(N)cc3sc-2c1"
    assert reason == "largest_parseable_fragment"


def test_embedding_fallback_preserves_original_row_order_and_audit():
    embeddings, records = _embed_with_audited_fallback(
        _FakeMiniMol(),
        ["CCO", PROBLEMATIC_MIXTURE, "CCN"],
    )

    assert embeddings.shape == (3, 2)
    assert records == [
        {
            "chunk_row": 1,
            "original_smiles": PROBLEMATIC_MIXTURE,
            "embedded_smiles": "N=c1ccc2c(-c3ccccc3)c3ccc(N)cc3sc-2c1",
            "reason": "largest_parseable_fragment",
        }
    ]


def test_unique_index_loader_accepts_compact_directory_indices(monkeypatch, tmp_path):
    index_dir = tmp_path / "08_neighbor_index"
    index_dir.mkdir()
    loaded = {"molecules": [], "group_to_molecule_indices": {}}
    calls = []

    def fake_load_index(path):
        calls.append(path)
        return loaded

    monkeypatch.setattr(
        "tools.chembl_tool.paper_experiments.build_minimol_retrieval_features.load_index",
        fake_load_index,
    )
    experiments = [
        SimpleNamespace(index=str(index_dir)),
        SimpleNamespace(index=str(index_dir)),
    ]

    assert _load_unique_indices(experiments) == {str(index_dir): loaded}
    assert calls == [index_dir]
