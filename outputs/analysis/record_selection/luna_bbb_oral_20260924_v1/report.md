# Luna BBB and Oral direct and indirect optimization

Five DeepSeek-seeded direct settings were evaluated per panel on the full validation split. The highest validation macro F1 (then accuracy, then profile name) fixed the direct selection. Five indirect settings were then evaluated with that direct selection unchanged. The frozen direct and mixed weights were reselected on each benchmark's test cache and evaluated without test tuning.

| Benchmark | Task | Validation direct | Validation mixed | Test direct | Test mixed |
|---|---|---:|---:|---:|---:|
| gold_v1 | bbb_martins | 0.764957 | 0.766554 | 0.666929 | 0.72261 |
| gold_v1 | bioavailability_ma | 0.750945 | 0.818124 | 0.760093 | 0.856127 |
| tdc_v2 | bbb_martins | 0.876349 | 0.864937 | 0.828823 | 0.84633 |
| tdc_v2 | bioavailability_ma | 0.74359 | 0.814493 | 0.718372 | 0.774023 |

All four direct and all four mixed test arms completed every query with zero failed runs. `results.tsv` gives every validation and test profile metric; `comparisons.tsv` gives counts, accuracies, profiles, and direct-to-mixed differences. `provenance.json` pins the source matrices, completion receipts, prompt provenance, provider pool, and freeze records. Results are descriptive comparisons of these fixed configurations, not estimates of the causal effect of indirect records.

Gold-v1 BBB and Oral used the reviewed Gold cache bundle. TDC-v2 used the query-pure cache and exact-score overlay; BBB removed prediction-only records before joint selection. The overlay has no reusable scores for six TDC-v2 BBB validation queries and five test queries, and its L5 rows have no assay scores. Their prompts show Morgan similarity, while scored records show both Morgan similarity and transfer likelihood. No score was inferred for a missing item or gated by a whole-query coverage percentage.

All model requests used `openai/gpt-6-luna` through OpenRouter Flex with OpenAI upstream, high reasoning effort, and one 256-request launcher pool per batch. The prompt was `final_full_flat` throughout. Failed attempts were eligible for immediate transport retry and three stage requeues; completed matrices have no unresolved failures.

DeepSeek weights were candidate priors only. Gold direct priors came from complete `valid_small` task results; TDC-v2 direct priors transferred weights from TDC-v1 screens but rebuilt selections on v2. TDC-v2 BBB indirect priors came from its 100-query filtered DeepSeek screen; the other indirect panels used a documented DeepSeek seed and four neighbors fixed before Luna scoring. See `indirect_candidates_frozen_before_luna.json`.
