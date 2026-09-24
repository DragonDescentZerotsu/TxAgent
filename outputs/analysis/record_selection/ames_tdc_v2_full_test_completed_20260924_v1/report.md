# TDC-v2 AMES full-test result

All 1,457 test queries have a valid final response across the original run and targeted recoveries. Direct-plus-indirect macro-F1 is 0.799350, versus 0.782289 for direct-only. The direct-plus-indirect confusion matrix is TN=409, FP=178, FN=96, TP=774; accuracy is 0.811942. Valid-response source precedence is original, OpenRouter recovery, first local recovery, 18082 tail, then the isolated query-1158 retry. No valid overlapping predictions conflicted.

This is complete **response coverage across runs**, not a claim that each historical matrix has a terminal complete receipt. The last missing query, index 1158, has a schema-valid response in a fresh one-query run. Its rendered request is byte-identical to the 65,536-token attempt, but its maximum completion allowance is 16,384 tokens; this request-configuration difference is part of the result provenance. All attempts used the upstream-v3 prompt and high reasoning effort. Redundant long-tail launchers were stopped after full response coverage was verified.

The machine-readable scores are in `results.tsv`; `manifest.json` pins the label file, source matrices, and final recovery response.
