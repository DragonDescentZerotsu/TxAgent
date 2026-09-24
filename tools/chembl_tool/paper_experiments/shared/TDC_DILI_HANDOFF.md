# TDC DILI: parent-disjoint rerun

**Continue with your latest optimizations and configuration. The only additional
change requested here is to fix parent-disjoint filtering for TDC DILI indirect
evidence, then rerun TDC DILI Direct + Indirect.** Keep your current prompt,
selector, budgets and inference settings; do not revert to the archived setup.
Leave other tasks and Gold-v1 experiments untouched.

## What we found

In the inspected archive, 31/96 Mixed queries retrieved indirect records sharing
the query's connectivity. The associated code (`5c3a6ae2`,
`predict/retrieval/assay_reranking/build_ranked_uid_retrieval.py`, lines 849–859)
only compared full parent IDs. For example, finasteride's query and donor keys
start with `DBEPLOCGEIEOCV` but differ in stereochemical annotation. Our policy
excludes this match. A small replay corrected finasteride after removal, but did
not fix every case; full-test improvement is not yet established. This diagnosis
describes the archived implementation, not an audit of your latest optimizations.

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

Apply the filter before candidate truncation, then let your latest selector fill
its configured budget from eligible donors. Refresh only the TDC DILI retrieval
cache so old cached matches cannot bypass the fix. Leave current L1 evidence alone.

Use the existing runner, save the rerun separately, and compare with your current
TDC DILI results, noting any configuration differences. No other experiment needs
to be rerun for this request.
