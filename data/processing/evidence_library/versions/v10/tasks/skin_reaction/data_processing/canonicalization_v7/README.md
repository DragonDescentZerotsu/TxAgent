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

## V10 Stage 1c unit successor

`exact_measurement_unit_map.v3.json` is the Skin V10 successor to the reviewed
v2 unit map. It preserves every v2 decision, then reconciles the units exposed
by the V10 source snapshot, deterministic router, and frozen v8 measurement
extraction. `exact_measurement_unit_map.v3.manifest.json` pins all three inputs
and records coverage and decision counts. Rebuild it with:

```bash
python -m data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.build_unit_reconciliation --overwrite
```

The builder reuses an existing v2 decision only when its action, canonical
unit, scale, and domain are unambiguous. New percentage and explicit-denominator
forms receive reviewed exact transforms; safely named scales retain their
identity, and prose that is not a unit is excluded explicitly.

## V10 Stage 1d exact deduplication

`../stage1_exact_deduplication.v1.json` classifies every raw source column as
scientific, canonicalized base input, provenance, prose, or processing metadata.
Exact candidates require repaired SMILES, PMID, canonical measurement, and
canonical unit. Within-source rows additionally match every configured
scientific field. Cross-source rows are preserved when the sources have no
shared scientific field, preventing equal percentages from collapsing distinct
populations, concentrations, or study arms. The published Stage 1 removes 295
within-source duplicates and retains 66,667 records.
