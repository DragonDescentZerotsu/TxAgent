# Measurement-resolution quality evaluation

This directory is the single home for reviewed measurement-resolution gold data
and its quality checks. Gold labels are edited only through manual scientific
review; they are never regenerated from model output.

## Contents

- `gold/bbb_martins.v8.jsonl`: the active 500-case BBB V8 quality corpus. Each
  case includes the source input, gold status, gold measurement coefficient,
  and gold unit.
- `gold/bbb_martins.v7.jsonl`: the preserved pre-V8 BBB corpus. The V8 manifest
  records every unit-policy and single-measurement migration from this file.
- `gold/bioavailability_ma.v4.jsonl`: 300 reviewed Bioavailability cases.
  Each case includes the
  source input, gold status, gold measurement coefficient, and gold unit.
- `gold/skin_reaction.v1.jsonl`: 300 reviewed Skin cases.
- `test_gold_corpora.py`: structural and scientific invariants for the labels.
- `evaluate_mapping.py`: offline scoring of a frozen extraction mapping. It
  reports status, measurement, unit, joint pair, whole-record, and per-source
  accuracy, and writes a row-level CSV for review.

## BBB V8 quality run

The corpus contains 500 cases, but the V8 replay intentionally calls the model
only for cases that the current deterministic router sends to extraction. Exact
copies and deterministic non-scalar rejections should not consume LLM tokens.
Consequently, the report records both the 500-case corpus size and the number of
model-evaluated cases.

Run the frozen replay with a dedicated cache, then score it:

```bash
python -m data.processing.evidence_library.versions.v8.build_measurement_resolution_mapping \
  --task bbb_martins \
  --gold-replay \
  --mapping-path data/artifacts/evidence_library_assets/bbb_martins/construction_assets/measurement_resolution_v15_gpt54mini/quality/measurement_resolution.parquet \
  --cache-dir data/artifacts/evidence_library_assets/bbb_martins/construction_assets/measurement_resolution_v15_gpt54mini/quality/cache \
  --token-ledger data/artifacts/evidence_library_assets/bbb_martins/construction_assets/measurement_resolution_gpt54mini/ledgers/openai_key_1.json \
  --base-url https://api.openai.com/v1 \
  --provider openai \
  --api-key-env OPENAI_API_KEY \
  --env-file /vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/distillation/.env \
  --model gpt-5.4-mini \
  --workers 64 \
  --budget-epoch bbb-v8-quality-gpt54mini-key1 \
  --budget-max-tokens 10000000

python -m tests.chembl_tool.common.measurement_resolution_quality.evaluate_mapping \
  data/artifacts/evidence_library_assets/bbb_martins/construction_assets/measurement_resolution_v15_gpt54mini/quality/measurement_resolution.parquet
```

The evaluator writes `quality_metrics.json` and `quality_rows.csv` beside the
mapping. A live API call is never part of pytest. Full generation resumes the
same submission cache with `OPENAI_API_KEY_TWO` and its own 10,000,000-token
ledger only if the first ledger is exhausted; cached rows are never submitted
twice.
