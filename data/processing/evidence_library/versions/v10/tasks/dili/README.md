# DILI V10 Stage 1

DILI V10 currently implements source ingestion, cleaning, measurement routing,
and measurement resolution only. Stage 2 canonicalization, pair-bucket policy,
assay-transfer eligibility, a completed evidence library, and a `CURRENT`
publication are outside this stage and have not been created.

The frozen source snapshot contains six schemas. Measurement resolution reads
the schema-owned outcome fields directly:

| source | measurement | unit | semantic context |
|---|---|---|---|
| `dili_base` | `causal_status` | none | human agent-DILI relationship |
| `dili_v1` | `result_value` | `result_unit` | `assay_and_readout` |
| `dili_v2` | `reported_result` | inline | `assay_detail` |
| `dili_v3` | `result_value` | `result_unit` | `endpoint_metric` and related assay fields |
| `dili_v4` | `result_value` | `result_unit` | `endpoint_and_comparator` and platform fields |
| `dili_v5` | `quantitative_value` | `quantitative_unit` | `specific_endpoint_name` and measure type |

`canonical_endpoint_name` is only an identity batching key: the cleaned raw
endpoint literal, the fixed `human_dili_relation` key for `dili_base`, or the
`missing_endpoint` sentinel. This stage has no scientific endpoint aliases and
does not create canonical endpoint mappings. In V3, the outcome column can
still contain a delta; an `endpoint_metric` beginning `change in` explicitly
marks the selected value as relative.

Stage 1 retains 1,627,666 of 1,645,109 source rows after 17,443 structure
rejections. Routing is a complete partition: 140 narrow deterministic V5 potency
accepts, 478,469 model-extraction rows, and 1,149,057 deterministic rejections.
The deterministic accepts require an exact `AC50`, `EC50`, `IC50`, or `MEC`
same-clause value/unit pair; the realized set contains 92 EC50, 47 IC50, and one
MEC row. A routing rejection means that no numeric mapping is attempted; it does
not delete the source record.

Model extraction uses the fixed DILI wrapper:

```bash
python -m data.processing.evidence_library.versions.v10.tasks.dili.build_measurement_resolution_mapping
```

The wrapper fixes `http://dgx027:50001/v1`,
`deepseek-ai/DeepSeek-V4-Flash-0731`, provider `local`, temperature `0`, 8,192
completion tokens, no credential, and one 512-worker pool within the single
launcher. It rejects provider, model, endpoint, credential, worker, and
token-limit overrides. Cached responses remain under
`outputs/chembl_tool/measurement_resolution_generation/`.
Temperature zero freezes the request setting; independent replays show that the
distributed server is not bitwise deterministic.

The completed full mapping is
`data_processing/measurement_resolution_v1/measurement_resolution.parquet`
(478,469 rows; SHA-256
`d8d979c6415b4385a64e3346da00a64b91b7e0a001d779782dd1e43e27d3c1af`).
It contains 55,887 `ok`, 123,085 `relative`, 121,408 `unsure`, and 178,089
`unavailable` rows. Assignment methods are 376,404 single-pass, 1,453 structural
retry, and 100,612 guard-demoted `unsure` rows. One transient connection failure
for 20 rows was retried after rechecking all frozen input hashes; the final
mapping has complete ID/UID/source coverage and no terminal failures. The
adjacent `provider_endpoint_receipt.json` records mapping, cache, gold, and live
endpoint verification.

The reviewed regression corpus is
`tests/chembl_tool/common/measurement_resolution_quality/gold/dili.v10.jsonl`.
It has 1,192 source-only cases: all 1,052 reviewed extraction cases plus the
complete 140-row deterministic-accept universe. The selected guard-v13 replay
covered all model-target rows and returned the exact reviewed measurement-unit
pair for 195 of 197 `ok` predictions (98.98%). Its two strict misses were
source-supported descriptive unit expansions reserved for the separate
canonicalization process. The precision-first policy abstains often: it recovered
197 of 335 reviewed usable measurements (58.81%), with 804 of 1,052 statuses
correct (76.43%) and 802 whole records correct (76.24%). Stage 1 preserves these
abstentions as `unsure` or `unavailable`; the quality gate has no recovery
minimum.
