# Gold-v1 AMES full-test result

All 274 direct-plus-indirect test queries have a valid final response across the preserved original and recovery runs. Macro-F1 is 0.774211, versus 0.708578 for the 274-query direct-only run. The confusion matrix is TN=52, FP=26, FN=24, TP=172. The final missing query, index 151, was correctly predicted positive in the completed OpenRouter recovery.

This is complete **response coverage across runs**, not a claim that each historical matrix has a complete terminal receipt. The original and earlier recovery matrices remain preserved as they were. The machine-readable result is in `results.tsv`; `manifest.json` pins the benchmark labels and the final recovery response.
