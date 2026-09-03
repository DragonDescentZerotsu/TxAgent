import pandas as pd

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.tasks.clintox.build_flat_assay_support_evidence import (
    _direct_rows,
)


def test_direct_rows_remove_heldout_parents_and_keep_retrieval_fields(tmp_path):
    direct_path = tmp_path / "direct.parquet"
    pd.DataFrame(
        [
            {
                "source_row_number": 0,
                "toxicity_outcome": "failure",
                "support_text": "held-out direct evidence",
                "confidence": 0.9,
                "SMILES": "CCO",
            },
            {
                "source_row_number": 1,
                "toxicity_outcome": "failure",
                "support_text": "retained direct evidence",
                "confidence": 0.8,
                "SMILES": "CCN",
            },
        ]
    ).to_parquet(direct_path, index=False)
    membership = pd.DataFrame(
        [
            {
                "source_id": "clinical_trial_failure",
                "source_row_number": 0,
                "assay_id": "a1",
                "molecule_id": "heldout",
            },
            {
                "source_id": "clinical_trial_failure",
                "source_row_number": 1,
                "assay_id": "a1",
                "molecule_id": "retained",
            },
        ]
    )
    heldout_key = normalize_molecule_identity("CCO").parent_connectivity_key

    rows, stats = _direct_rows(
        direct_path=direct_path,
        membership=membership,
        assay_context_by_id={"a1": "clinical failure"},
        heldout_parent_keys={heldout_key},
    )

    assert [row["molecule_chembl_id"] for row in rows] == ["retained"]
    assert rows[0]["canonical_smiles"] == "CCN"
    assert rows[0]["group_id"] == "Assay.a1"
    assert stats["n_direct_heldout_records_excluded"] == 1
    assert stats["n_direct_heldout_records_after_filter"] == 0
