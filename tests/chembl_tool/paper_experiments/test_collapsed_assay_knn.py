import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.run import (
    build_feature_catalog,
    encode_assays,
    load_numeric_assays,
    mean_neighbor_matrix,
    morgan_fingerprints,
    neighbor_label_scores,
    retrieve_neighbors,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.model_ablation import (
    build_model,
    load_neighbor_label_scores,
    remove_presence,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_k_cv import (
    choose_k,
    make_folds,
)


def test_sparse_assay_encoding_and_neighbor_averaging() -> None:
    train = [
        {"parent_key": "a", "Y": 0},
        {"parent_key": "b", "Y": 1},
        {"parent_key": "c", "Y": 1},
    ]
    assays = {
        "a": {"assay-1": 1.0},
        "b": {"assay-1": 3.0, "assay-2": 5.0},
        "c": {},
        "valid": {"assay-1": 2.0, "unseen": 9.0},
    }
    catalog, feature_index = build_feature_catalog(train, assays)
    train_matrix, train_coverage = encode_assays(train, assays, catalog, feature_index)
    valid_matrix, valid_coverage = encode_assays(
        [{"parent_key": "valid"}], assays, catalog, feature_index
    )

    assert [row["pair_bucket_key"] for row in catalog] == ["assay-1", "assay-2"]
    assert train_coverage[2]["observed_train_catalog_assays"] == 0
    assert valid_coverage == [
        {"observed_train_catalog_assays": 1, "ignored_unseen_assays": 1}
    ]
    np.testing.assert_allclose(valid_matrix.toarray(), [[0.0, 0.0, 1.0, 0.0]])

    neighbors = [[1, 2], [0, 2], [0, 1]]
    means = mean_neighbor_matrix(train_matrix, neighbors, 2)
    expected = (train_matrix[1] + train_matrix[2]) / 2
    np.testing.assert_allclose(means[0].toarray(), expected.toarray())
    np.testing.assert_allclose(
        neighbor_label_scores(np.asarray([0, 1, 1]), neighbors, 2),
        [1.0, 0.5, 0.5],
    )
    assert sparse.isspmatrix_csr(means)


def test_stage06_filter_and_parent_median(tmp_path) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _stage06_row("CCO", 1.0),
                _stage06_row("CCO", 3.0),
                _stage06_row("CCO", 100.0, assay_transfer_eligible=False),
                _stage06_row("CCN", 7.0, aggregation_method="categorical_mode"),
                _stage06_row("CCC", 9.0, retrieval_source_id="direct_vote"),
            ]
        ),
        path,
    )
    identity = normalize_molecule_identity("CCO")
    parent_key = identity.parent_inchi_key or identity.parent_smiles
    assays, stats = load_numeric_assays(path, {parent_key})

    assert assays == {parent_key: {"assay": 2.0}}
    assert stats["eligible_rows_read"] == 2
    assert stats["parent_assay_collisions_collapsed_by_median"] == 1


def test_train_neighbor_retrieval_is_leave_one_out() -> None:
    rows = [
        {"drug": "CC", "parent_key": "a", "Y": 0},
        {"drug": "CCC", "parent_key": "b", "Y": 1},
        {"drug": "CCCC", "parent_key": "c", "Y": 1},
    ]
    fingerprints = morgan_fingerprints(rows)
    neighbors, _ = retrieve_neighbors(
        rows,
        fingerprints,
        rows,
        fingerprints,
        top_k=2,
        leave_one_out=True,
    )

    assert all(index not in selected for index, selected in enumerate(neighbors))
    assert all(len(selected) == 2 for selected in neighbors)


def test_model_ablation_helpers(tmp_path) -> None:
    matrix = sparse.csr_matrix([[1.0, 2.0, 1.0, 0.0]])
    np.testing.assert_allclose(remove_presence(matrix, 2).toarray(), [[1.0, 2.0]])
    path = tmp_path / "neighbors.jsonl"
    path.write_text(
        '\n'.join(
            [
                '{"split":"train","query_index":1,"neighbors":[{"train_index":0},{"train_index":2}]}',
                '{"split":"train","query_index":0,"neighbors":[{"train_index":1},{"train_index":2}]}',
            ]
        )
        + '\n'
    )
    scores = load_neighbor_label_scores(path, "train", np.asarray([0, 1, 1]), 2)
    np.testing.assert_allclose(scores, [1.0, 0.5])
    assert build_model("logistic_l1").l1_ratio == 1.0
    assert build_model("logistic_l2").l1_ratio == 0.0
    assert build_model("random_forest").n_estimators == 100


def test_scaffold_folds_and_smaller_k_tie_break() -> None:
    rows = [
        {"drug": smiles, "parent_key": f"p{index}", "Y": index % 2}
        for index, smiles in enumerate(
            [
                "c1ccccc1",
                "Cc1ccccc1",
                "C1CCCCC1",
                "CC1CCCCC1",
                "c1ccncc1",
                "Cc1ccncc1",
                "C1CCNCC1",
                "CC1CCNCC1",
                "CC",
                "CCC",
                "CCO",
                "CCN",
            ]
        )
    ]
    folds, groups = make_folds(rows, n_folds=2, seed=0)
    heldout = np.concatenate([query for _, query in folds])
    assert sorted(heldout.tolist()) == list(range(len(rows)))
    for reference, query in folds:
        assert not set(np.asarray(groups)[reference]) & set(np.asarray(groups)[query])
    assert choose_k([{"k": 10, "macro_f1": 0.6}, {"k": 5, "macro_f1": 0.6}])["k"] == 5


def _stage06_row(smiles: str, value: float, **updates):
    row = {
        "canonical_smiles": smiles,
        "pair_bucket_key": "assay",
        "finite_scalar_value": value,
        "aggregation_method": "continuous_median",
        "aggregation_status": "valid",
        "assay_transfer_eligible": True,
        "retrieval_source_id": "indirect",
    }
    row.update(updates)
    return row
