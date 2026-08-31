# Bioavailability v7 canonicalization audit

This directory records the local, no-API review behind `starling_schema.py`. It does not replace or publish
the frozen auxiliary mapping.

- Oral exposure cleans `exposure_measure`, `parameter_value`, `parameter_units`, and source `smiles` into the
  four universal roles. Canonical structure still comes from the reviewed `global_identifier` mapping.
- Fa uses `assay_system` for assay context and `biological_context` plus `assay_system` for species context.
- Fg independently derives assay and species context from `assay_system`; no cleaned species alias is created.
- Fh uses `assay_system` for assay context and `species` plus `assay_system` for species context.
- Direct-HF supplies working constants for endpoint and unit; those constants are not marked source-visible.
- Measurement and unit rules remain atomic. The existing Fg scalar rule module remains the focused scientific
  helper rather than being copied into the schema.
- `endpoint_concepts/*.json` exhaustively maps every frozen source `endpoint_name` to a hand-reviewed,
  source-specific `canonical_endpoint_concept`. The source endpoint and existing fine-grained
  `canonical_endpoint_name` remain unchanged; Stage 3 uses the concept only in the endpoint slot of the
  pair-bucket key. The mapping loader verifies the
  frozen endpoint inventory count and SHA-256 and fails closed on any new or missing endpoint.
- `pair_bucket_semantic_eligibility_v1/` preserves the retired review of
  non-absolute buckets. The active core accepts only absolute and
  endpoint-defined-ratio numeric calibration candidates.

The v6 artifacts and their hashes remain historical lineage. A v7 build must write to
`starling_normalized_v7` and pass source-field, measurement/unit, and pair-bucket fixture tests before use.
