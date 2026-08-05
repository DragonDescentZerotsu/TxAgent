# Skin Reaction v7 canonicalization audit

This is a local, no-API schema review over the globally reconciled mapping. It changes field binding, not the
reviewed mapping values.

- Direct records use canonical assay/test, species/population, and severity dimensions.
- Sensitization derives canonical assay type and species context independently from `assay_type`; its globally
  reconciled endpoint is the final `canonical_endpoint_name`, so v7 needs no endpoint-key override.
- Phototoxicity uses canonical assay method and evidence system. Evidence system is not mislabeled as species.
- Skin exposure uses canonical study design and derives species context from `skin_source`.
- `single_subject_logit` is a binary scale and requires both levels. Ordinal and signed scales retain the
  three-level minimum.

The published global mapping remains immutable. V7 writes a new canonical view under
`starling_normalized_v7`; it does not republish or overwrite v6.
