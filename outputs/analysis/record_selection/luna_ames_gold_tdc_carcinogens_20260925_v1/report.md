# Luna Gold AMES and TDC-v2 Carcinogens optimization

Five DeepSeek-seeded direct profiles were evaluated on full validation for each panel. The highest validation macro-F1, then accuracy, then profile name fixed the direct panel. Five indirect profiles were then evaluated with that direct panel held fixed. The winning weights were reselected on the test cache without test tuning.

| Benchmark | Task | Validation direct | Validation mixed | Test direct | Test mixed |
|---|---|---:|---:|---:|---:|
| Gold-v1 | AMES | 0.776817 | 0.787095 | 0.667730 | 0.706003 |
| TDC-v2 | Carcinogens | 0.755102 | 0.681300 | 0.841539 | 0.737500 |

All 24 validation and test arms completed with zero failed queries. The four test arms cover 274 Gold AMES and 56 TDC-v2 Carcinogens queries each. [results.tsv](results.tsv) records every profile, and [comparisons.tsv](comparisons.tsv) records the frozen winners and test differences. The four freeze records and [provenance.json](provenance.json) pin the decisions and source matrices.

All requests used `final_full_flat`, `openai/gpt-6-luna` through OpenRouter Flex with OpenAI upstream, high reasoning effort, and one 512-request launcher pool. TDC-v2 Carcinogens used the reviewed original-split cached-prior overlay and exact-score selection; Gold AMES used the V27 successor cache and `available_scores.v1`. The Carcinogens validation split has only 28 queries, so its profile ranking is uncertain. The mixed and direct test results describe these frozen configurations; they do not isolate a causal effect of indirect records.
