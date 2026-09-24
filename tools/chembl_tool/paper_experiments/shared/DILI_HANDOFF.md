# DILI: small changes to try

For the retained **TDC DILI-only** prompt and completed replay results, use
[the scoped handoff](TDC_DILI_HANDOFF.md). This override does not apply to Gold-v1.
The Gold-v1 suggestions below are background, not instructions to launch more runs.

Use the existing harness and the same 402-row Gold-v1 scaffold test. This is a
follow-up on an inspected test set, not a new held-out evaluation.

## Fix the output labels first

The archived prompt says `pass = dili_risk`, but the model sometimes reads `pass` as
"safe". In the completed-v2 recovery, 13 new Mixed regressions and 4 new Direct
errors had a final label inconsistent with their own reasoning. For example,
Mixed query #97 says `fail (meaning it does cause DILI)`; #113 says
`pass (no DILI risk)`. The parser follows the documented mapping correctly.
These indices are zero-based. This is a model encoding error, not a reversed
Python mapping, and it affects both arms.

The [current prompt handoff](README.md#task-prompts) supplies the fix and its
integration steps for all four risk tasks. DILI now outputs `dili_risk` or
`no_dili_risk` in the existing `final_prediction` field, with an identity mapping.
The task's exposure/population qualifications remain unchanged. The existing
validator and mapper support these values; no new parser or runner is needed.

Use the same service configuration for Direct and Mixed when rerunning;
recovery used different endpoints. Do not globally swap the historical mapping
or infer scored labels from reasoning with regex. The supplied fix has passed
offline checks but has not been rerun on the full Gold-v1 test.

## Three separate follow-ups

- **L2 membership:** some L2 records have explicit source labels. The
  [32 flagged records](dili_l2_former_voters_20260924.csv) are L1 voters in our
  current source but were explicitly demoted to L2 in Joseph's level mapping v2
  because they are outside his current published-card voter membership. Review
  this membership difference before changing levels; do not delete them or treat
  every L2 record as uncertain. A donor label is not a query label, and repeated
  reports of one study are not independent votes. The CSV covers the 32 records
  flagged in 19 negative-query regressions, not every L2 record in the library.
- **Donor concentration:** some queries get 46/50 or 49/50 indirect records from
  one donor. Try a simple per-donor cap using the existing selector, filling from
  other eligible donors and reporting any shortfall. Concentration occurs in
  both helpful and harmful cases; improvement is not established.
- **Indirect budget:** compare 50 with 100 while keeping Direct and the selector
  fixed. Test the donor cap separately. More records may add useful coverage or
  more repetition; 100 is a hypothesis, not a known better setting. Our previous
  median 87 records was the total Direct + Indirect context, not indirect alone.

## Where the label mismatch came from

Our old shared exporter introduced the portable pass/fail labels. Our local
progressive and aligned full-flat DILI runs use `dili_risk / no_dili_risk`.
The DILI task configuration in the shared bundle at main `dbbede84` is identical
after YAML parsing to Joseph's upstream-v2 configuration at `5c3a6ae2`; his
system/user templates have other changes. This was not a byte-identical export
of our local runtime prompt. Original assets and the current bundle's provenance
are documented in the [bundle README](joseph_full_flat_v5/README.md).
The [other-task trace audit](LABEL_ENCODING_AUDIT.md) records the related findings.

Evidence: [completed trace archive](https://github.com/DragonDescentZerotsu/TxAgent/tree/4b4489c1598ba0fc428308b930d0ca71a90f2c84/outputs/paper/assay_transfer_harness/joseph/trace_archives/final_test_20260923_completed_v2).
Both arms have 402 successful predictions; Macro-F1 is 0.6145 Direct versus
0.5705 Mixed. The label fix has not yet been rerun on Gold-v1. L2 citations occur in 47
improvements and 38 regressions; citation is not proof of L2-only causation.
