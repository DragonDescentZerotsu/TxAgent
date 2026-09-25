# Luna Gold-v1 Skin direct FINAL arm

The frozen direct profile `ga100_mc025_label010` scored 0.716178 macro F1 on 100 `valid_small` queries and 0.559252 macro F1 on all 241 Gold-v1 test queries. Test accuracy was 0.672199; the confusion matrix was TN 20, FP 48, FN 31, TP 142. The profile was chosen on validation before test.

The new arm at `final/luna/test/gold_v1/skin_reaction/direct/` preserves all per-query requests, responses, traces, the exact upstream-v2 prompt snapshot, the frozen direct selection, the OpenRouter Azure route, and source hashes. Every response requested and served `openai/gpt-6-luna` through Azure. The existing Gold-v1 Skin mixed arm remains unchanged.

`results.tsv` contains every numeric result above; `provenance.json` pins the source study, FINAL arm, parameter snapshot, and collection entries. The compressed archive was left untouched.
