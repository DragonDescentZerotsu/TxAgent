# Distance-expansion same-molecule relevance

## Scope

This feature applies only to the experimental H1/H2 distance-expansion workflow. It does not apply to
ordinary source normalization, pair buckets, or direct/mechanism-family retrieval.

The implementation is in:

```text
tools/chembl_tool/common/evidence_distance.py
tools/chembl_tool/common/task_workflows/distance_assay_manifest.py
```

## Problem it prevents

A biological path may be correct while changing which molecule the conclusion is about:

```text
molecule A activates NRF2
  -> barrier P-gp abundance increases
  -> probe substrate B has lower brain accumulation
```

This evidence does not show that molecule A is a P-gp substrate. It therefore cannot predict A's own BBB
disposition without separate evidence establishing the missing role. Graph distance, assay quality, and
scope checks do not detect this change of causal subject.

## Audit contract

Every distance-expansion measurement family must have one `FamilySelfRelevanceAudit` with a rationale and
checkable citations. Its status is one of:

```text
pass_same_molecule
  The inference continues to describe the assayed molecule.

requires_query_role
  The inference requires an additional role for that molecule, such as transporter-substrate status.

context_only
  The evidence changes a system or describes the outcome of another molecule.

unresolved
  The causal subject cannot yet be established.
```

The publication gate is:

```python
validate_self_relevance_audit(config, audits, require_publishable=True)
```

It requires complete family coverage and accepts only `pass_same_molecule`. Other statuses may remain in
candidate and audit artifacts but cannot enter the published H1/H2 retrieval graph. A prompt warning does
not replace missing evidence about the query molecule's role.

## Current use

`distance_assay_manifest.py` runs this validation before scanning source assays. BBB currently provides the
audit in `tools/chembl_tool/tasks/bbb_martins/distance_self_relevance.py`. Its NRF2, KEAP1-NRF2, HIF-1, and
PHD2 expansion families require missing query-molecule roles, so the current BBB distance manifest is
intentionally blocked from publication.
