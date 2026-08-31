# Skin Reaction v7 canonicalization audit

This is a local, no-API schema review over the globally reconciled mapping. It changes field binding, not the
reviewed mapping values.

- Direct records use canonical assay/test, species/population, and severity dimensions.
- Sensitization derives canonical assay type and species context independently from `assay_type`; the existing
  globally reconciled endpoint remains provenance for its fine-grained `canonical_endpoint_name`.
- Phototoxicity uses canonical assay method and evidence system. Evidence system is not mislabeled as species.
- Skin exposure uses canonical study design and derives species context from `skin_source`.
- Group incidence and resolved incidence percentages use raw fractions. Single-subject outcomes remain on the
  separate binary `single_subject_fraction.v1` scale and require both levels. Ordinal and signed scales retain
  the three-level minimum.
- Explicit `positive_count` and `total_tested` columns are source-exact. Fractions or percentages found only in
  text must first pass the frozen measurement resolver; the categorical policy does not parse their numeric
  value directly.
- `endpoint_concepts/*.json` exhaustively maps every frozen raw/canonical endpoint pair to a hand-reviewed,
  source-specific concept. Stage 3 uses this concept only for the endpoint slot of the pair-bucket key; source
  endpoints, fine-grained canonical endpoints, AOP events, assay context, species, and reference semantics stay
  independently preserved.

The published global mapping remains immutable. V7 writes a new canonical view under
`starling_normalized_v7`; it does not republish or overwrite v6.
