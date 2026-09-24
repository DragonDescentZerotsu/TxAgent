# TDC-v2 DILI: direct-only versus indirect validation

A 47-query direct-only control used the frozen ten L1 cards, exact L1 scores, DILI mixed-v3 system prompt, cached query prior, DeepSeek V4 Flash 0731, and OpenRouter auto routing used by the indirect validation winner. All 47 responses and traces completed. Each paired query had byte-identical system instructions and cached query-prior output.

| Selection | Macro F1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|
| Direct only | 0.831541 | 0.851064 | 12 | 5 | 2 | 28 |
| Direct + indirect | 0.825465 | 0.851064 | 11 | 6 | 1 | 29 |

The indirect arm changed two calls: index 21 changed from false negative to true positive, and index 44 changed from true negative to false positive. Direct-only already had five of the indirect winner’s six false positives. Thus the validation risk bias was largely present before adding L2–L7 records. The two flipped calls used different OpenRouter upstream providers between arms, so this comparison does not isolate their causal source.

[results.tsv](results.tsv) exports the metrics, [paired_validation.tsv](paired_validation.tsv) lists every query and route, and [provenance.json](provenance.json) pins the source artifacts.
