# BBB meaningful-CNS-access source review

## Frozen decision

The v2 gold target is experimentally supported **meaningful or adequate CNS access after systemic administration** versus experimentally supported **restricted or poor access**. It is mechanism-agnostic: passive diffusion, active influx, and efflux can explain an outcome, but a mechanism-only proxy cannot vote by itself. Low but nonzero exposure may remain negative, and mere CNS detectability is not the positive-label rule.

Gold records are not resampled or relabeled to balance agent mechanism groups. Direct, passive, efflux, and influx coverage must instead be audited on the held-out-parent-filtered train-only evidence index.

## Review coverage

- Frozen Starling source revision: `f50c638621fcc2dedfc9e00f7074f867640efc1b`
- Review model: three independent `gpt-5.6-sol` reviewers
- Deterministic sampling: up to 25 records per endpoint-family and label bucket, seed `20260809`
- Unique source records read across iterative replacement rounds: **366**
- Final deterministic sample: **294/294 passed**
- Audit status and fingerprint are stored in `summary.json`; reviewed rows are stored in `source_review_sample.jsonl`.

The review rounds were cumulative. The first v2 sample had 299 records: 247 passed, 33 failed, and 19 were uncertain. After revision-bound exclusions and generic rule tightening, 49, 14, 2, 1, and 1 new replacement records were reviewed. Every record remaining in the final sample passed the frozen contract.

## Main failure causes found

1. Computational, calculated-logBB, in-vitro, or ex-vivo-only evidence presented as an in-vivo outcome.
2. Query/analyte identity mismatch, including abbreviation errors, parent/metabolite ambiguity, and a compound measured only after a different precursor was dosed.
3. PMID, route, model, or primary-experiment mismatch.
4. Altered barriers, intracranial/non-systemic delivery, special formulations, disease lesions, or mixed-extract attribution that could not support a molecule-only systemic-access label.
5. Indirect inference from efficacy, pharmacodynamic response, target engagement without an access measurement, or a mechanism-only efflux ratio.
6. Qualitative positive labels that only established low-level detectability rather than meaningful access.
7. Total-radioactivity records that could not distinguish unchanged query molecule from metabolites, and metal complexes/elements that collapse under the current fragment-parent/tool identity contract.

## Reproducibility boundary

All row-level manual removals are revision-bound in:

```text
tools/chembl_tool/tasks/bbb_martins/experimental_meaningful_cns_access_exclusions.json
```

Generic rejection logic and unit tests live beside the task adapter. A rebuild resets the manual-review status to `pending`; it can be marked `passed` only when the regenerated sample belongs to the same build fingerprint and has been read under this contract.

This is a high-precision source audit, not a substitute for independent double annotation of every accepted source record for a final paper release.
