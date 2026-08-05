# BBB v7 canonicalization policy

BBB v7 separates source cleaning, endpoint/value/unit canonicalization, pair
bucketing, and distance calibration into independently auditable stages. It
does not change the frozen v6 artifacts, approved Direct endpoint v1 mapping,
or frozen auxiliary reconciliation.

- Direct uses `assay_model` and `species` for separate canonical contexts.
- Passive permeability keeps `biological_system` as the cleaned source field and may derive both assay and
  species canonical contexts from it.
- Efflux keeps `assay_system` as the cleaned source field and derives both contexts independently; transporter
  and evidence type also receive explicit canonical fields before pair bucketing.
- Influx has no invented measurement or unit source value. Its transport mechanism is canonicalized directly.
- Binary categorical scales are complete with both declared levels. Ordinal scales require at least three.
- `measurement_semantics.v1.json` is the only BBB-specific registry that can
  approve an endpoint's quantity kind, numeric domain, compatible unit family,
  or a missing-unit default. Unknown combinations remain evidence-only.
- `canonical_endpoint_producer_id` and `canonical_pair_producer_id` identify
  the actual scalar or controlled-categorical producer used for every record;
  an atomic endpoint/measurement/unit projection cannot mix producers.
- The common measurement parser folds only explicit scientific notation and
  supports independent `value ± variation` factors. Ambiguous compressed OCR
  notation remains quarantined.
- An explicit BBB source denominator such as `mL/100 g/min` is converted to
  the per-gram basis by scaling both the value and its variation by `0.01`;
  numerator `×100` expressions are not matched by this rule.

Materialize the endpoint/unit decision inventory after a candidate build:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.audit_starling_measurement_semantics \
  --records <candidate>/02_canonicalized/records.parquet \
  --output <candidate>/09_audits/measurement_semantics.json
```

Known ambiguities fail closed. Direct endpoint corrections use the separate
human-gated v2 errata workflow in `../direct_endpoint_normalization_v2/`.
