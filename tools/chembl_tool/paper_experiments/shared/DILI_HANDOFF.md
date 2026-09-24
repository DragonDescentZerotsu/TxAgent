# DILI: small changes to try

Use the existing harness and the same 402-row Gold-v1 scaffold test. This is a
follow-up on an inspected test set, not a new held-out evaluation.

## Fix the output labels first

The prompt says `pass = dili_risk`, but the model sometimes reads `pass` as
"safe". In the completed-v2 recovery, 13 new Mixed regressions and 4 new Direct
errors had a final label inconsistent with their own reasoning. For example,
Mixed query #97 says `fail (meaning it does cause DILI)`; #113 says
`pass (no DILI risk)`. The parser follows the documented mapping correctly.
These indices are zero-based. This is a model encoding error, not a reversed
Python mapping, and it affects both arms.

The corrected [task configuration](joseph_full_flat_v5/tasks.yaml) and matching
[provenance mapping](joseph_full_flat_v5/provenance.json) are now supplied as
`joseph_full_flat_v5_dili_native_labels.v1`. Only DILI label instructions, allowed
values and mapping change; `system.jinja`, `user.jinja`, the two-field JSON shape
and all other task definitions are unchanged. Both native labels have passed
offline rendering and the existing parser-function checks; no LLM rerun yet.

Keep `claims` + `final_prediction`, but use native DILI labels:

```text
Use exactly one final_prediction:
- dili_risk: predict clinically meaningful human DILI under the query condition.
- no_dili_risk: predict no clinically meaningful human DILI under that condition.
This does not imply safety under every exposure.
Ensure final_prediction agrees with your overall evidence assessment.
```

In Joseph's existing prompt configuration:

- Set DILI `positive_prediction: dili_risk` and
  `negative_prediction: no_dili_risk`; remove the pass/fail instruction.
- Let the existing schema/validator derive allowed values from that contract.
  For this new version, reject `pass`/`fail` through the existing validation retry.
- Change DILI's `provenance.json` label mapping to
  `{"dili_risk": "dili_risk", "no_dili_risk": "no_dili_risk"}`.
  Existing `flat.py::native_flat_prediction()` can use this identity mapping;
  the final scorer already maps native labels to 1/0.

Keep the other scientific instructions and retrieval unchanged for this check.
Check both labels through rendering, validation and scoring. Use a fresh prompt
version/output root and the same service configuration for both arms; recovery
used different endpoints. Do not globally swap the old mapping or infer labels
from reasoning with regex. No new runner, parser or inference stage is needed.

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
of our local runtime prompt. Do not regenerate a new DILI bundle with the old
portable exporter unchanged. The original shared assets remain available at commit
`945380eb`; the current bundle contains the explicit DILI-only revision.

The [other-task trace check](LABEL_ENCODING_AUDIT.md) also confirms final label
inversions in Ames and Carcinogens. Skin has a self-corrected intermediate
inversion, but no final inversion was confirmed in this screening. The current
patch changes DILI only; the other tasks still use the old pass/fail contract.

Evidence: [completed trace archive](https://github.com/DragonDescentZerotsu/TxAgent/tree/4b4489c1598ba0fc428308b930d0ca71a90f2c84/outputs/paper/assay_transfer_harness/joseph/trace_archives/final_test_20260923_completed_v2).
Both arms have 402 successful predictions; Macro-F1 is 0.6145 Direct versus
0.5705 Mixed. The label fix has not yet been rerun. L2 citations occur in 47
improvements and 38 regressions; citation is not proof of L2-only causation.
