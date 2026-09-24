# TDC DILI: parent-disjoint rerun

**Scope: rerun only TDC DILI Direct + Indirect on the existing 96 test queries.**
Reuse the existing Direct reference. Leave all other tasks, the 402-row Gold-v1
benchmark, splits, labels, L1 evidence and other experiments unchanged.

## What we found

The completed archive has Macro-F1 **0.7556 Direct vs 0.7430 Mixed**: six queries
improve and seven regress. This is one fewer correct prediction, not a large drop.
31/96 Mixed queries retrieve indirect records sharing the query's connectivity.

The trace-associated code (`5c3a6ae2`,
`predict/retrieval/assay_reranking/build_ranked_uid_retrieval.py`, lines 849–859)
only checks `parent_id != query_parent`. Different full InChIKeys can share the
same first block: for finasteride, `DBEPLOCGEIEOCV-UHFFFAOYSA-N` versus
`DBEPLOCGEIEOCV-WSBQPABSSA-N`. Here the query omits stereochemistry and the donor
specifies it; these should not count as independent donors under our policy.

Our three-case replay used native DILI labels and removed these records without
backfill. All three gold labels are negative; indices are zero-based:

| Query | Keep records | Remove matching records | Removed |
|---|---|---|---:|
| #7 Finasteride | Wrong | Correct | 48/50 |
| #86 Prednisone | Wrong | Wrong | 46/50 |
| #93 Benzphetamine | Correct | Correct | 4/50 |

Prednisone still relied on other steroid labels and an uncertain human case
association. Identity filtering therefore will not fix every transfer error.
These selected cases do not establish a full-test gain.

## Reuse our implementation

- [molecule_identity.py](../../common/molecule_identity.py): `normalize_molecule_identity`.
- [retrieval_policy.py](../../common/retrieval_policy.py): `decide_candidate`.
- [Existing integration](../../common/task_workflows/retrieve_neighbors.py): `_retrieve_group_neighbors`.

`parent_disjoint` excludes exact identity, same-connectivity variants and same
parent; it still permits different parents sharing a scaffold. Use the shared
functions, not another identity implementation. Normalize the query once, then
apply this inside the existing **TDC DILI L2+** candidate loop:

```python
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import decide_candidate

query_identity = normalize_molecule_identity(query_smiles)  # once per query
# Inside the existing donor loop:
decision = decide_candidate(
    query_identity, {"canonical_smiles": donor_smiles}, "parent_disjoint"
)
if decision.excluded:
    continue
```

Filter before truncating the candidate list. Keep the current ranking/selector
and fill the original **50 indirect records** from eligible donors; report any
shortfall. Rebuild only this TDC DILI retrieval cache under a new identity/version
so the old cache cannot bypass the filter. Keep L1 exactly as before.

Use the [corrected DILI labels](README.md#task-prompts). If the reused Direct run
used the old prompt, label it a historical reference, not a retrieval-only control.
Keep other inference settings fixed; use the existing runner and retry handling.
Do not add donor caps, increase the budget, move L2 records or rerun other tasks.

Before running, check that #7/#86/#93 donors above are excluded and L1 is unchanged.
Save to a fresh output directory; report Macro-F1, accuracy, and paired corrections
versus regressions. This is a follow-up on an already inspected test set.
