# Oral Bio V9 measurement extraction

Full extraction is paused pending prompt review. The active prompt component is
`bioavailability_measurement_resolution_prompt.v4`, with at most one selected
self-contained outcome per row. It does not render endpoint-profile statistics.
The evaluated v3 template `bioavailability_v1.jinja` remains baseline provenance;
it is not selected by the task configuration.

The 500-case quality replay uses
`tests/chembl_tool/common/measurement_resolution_quality/gold/bioavailability_ma.v9.2.jsonl`
and the isolated `quality_baidu_v4/` directory under the run root below. Previous
v3 caches must not be resumed with v4; their input contracts intentionally reject
the prompt mismatch. The v9.2 change ledger preserves the earlier gold answers.

This component consumes the fresh V9 Stage-01 records and endpoint profile.
Only rows with the persisted `extract` route reach a model. Older measurement
maps are not reused. Existing maps for other canonical columns remain available
for the later canonicalization stage.

The run uses `gpt-5.4-mini` with `OPENAI_API_KEY`, then `OPENAI_API_KEY_TWO`.
Each key has an independent 10,000,000-token ledger; input and output tokens,
reasoning, retries and conservative unknown usage count toward the cap.
`OPENAI_API_KEY_THREE` is excluded. Restarting resumes the same ledgers.

Unprocessed and technically failed rows then use
`deepseek/deepseek-v4-flash-0731` through OpenRouter, restricted to `baidu/fp8`
with provider fallback disabled. Valid uncertain GPT answers are retained.
DeepSeek usage and reported cost are recorded without an additional token cap.
Three fallback passes are allowed before reporting unresolved failures.

Historical scale recipe (do not resume until a separate scaling decision):

```bash
python -u -m data.processing.evidence_library.versions.v9.build_measurement_resolution_mapping \
  --task bioavailability_ma --two-key-baidu-run \
  --cache-dir data/artifacts/evidence_library_assets/bioavailability_ma/v9_measurement_extraction_20260904 \
  --budget-epoch oral_bio_v9_20260904
```

The cache retains authentic API responses, requested/returned model, actual
provider, request ID and usage. Its input contract pins the source records,
endpoint profile and prompt; incompatible inputs cannot resume that cache.
Credential values are loaded only through `data/processing/llm_api.py`.

The mapping is published only after every extraction candidate has a valid
terminal answer. A transport or schema failure is never a scientific
`unavailable` answer. Stage 02 canonicalization and Stage 03 pair buckets are
outside this extraction run.
