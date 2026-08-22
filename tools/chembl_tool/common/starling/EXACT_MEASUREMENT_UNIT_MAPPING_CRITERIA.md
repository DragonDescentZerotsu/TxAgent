# Exact measurement unit mapping criteria

This document defines how to build and review
`exact_measurement_unit_map.v1.json`. The production map is intentionally only
an exact lookup table; this document is the reproducible human-review rubric.

## Scope

Stage 02 applies one rule to each successful numeric tuple using the exact key:

```text
(task, canonical_endpoint, input_unit)
```

The input may be a finite positive source value that bypassed extraction or a
quantity returned by the frozen LLM extraction. Every successful multi-quantity
extraction is expanded before lookup. The rule either:

- maps the coefficient to one `canonical_unit` with one positive fixed-point
  `scale` and numeric `domain`; or
- excludes the tuple from scalar use while preserving it as source evidence.

There are no runtime regexes, source-specific overrides, or per-row repairs.
One rule must therefore be safe for every active row with the same exact key.

## Primary decision rule

Map a key whenever its unit meaning is self-contained and one interpretation is
safe across all affected rows. Preserve an unusual literal unit if a broader
normalization is not proven. A rare unit or a new unit-specific pair bucket is a
valid result.

Exclude a key only when the measurement cannot be assigned one scientifically
honest unit rule across all affected rows.

### Never exclude for these reasons

- The unit creates or fragments a pair bucket.
- The unit is rare, a singleton, unfamiliar, or absent from an existing unit
  vocabulary.
- The unit is unexpected for its canonical endpoint.
- The endpoint is usually categorical.
- The key lowers a downstream bucket count or falls below a minimum bucket size.
- The quantity came from a multi-quantity extraction.
- Keeping the literal unit is less convenient than merging it with another unit.

These are organizational consequences, not evidence that the unit is invalid.

### Valid exclusion categories

| Category | Criterion |
|---|---|
| `ambiguous_meaning` | The quantity, unit, reference, direction, log basis, or physical meaning cannot be determined. |
| `mixed_exact_key` | Rows sharing the exact key require incompatible rules. |
| `missing_dimension` | The input supplies only a multiplier, index, or log label without the underlying measurement axis. |
| `parse_corruption` | Some affected rows were extracted inconsistently and no single rule repairs all of them. |
| `unresolved_scale_convention` | A displayed power of ten may be a literal multiplier or an inverse table-header convention, and the full source context does not resolve it. |
| `malformed_quantity` | The source and extraction together do not form a self-contained numeric quantity. |

An exclusion audit must name one of these categories and identify the evidence
that prevents a safe mapping. “Endpoint incompatible,” “rare,” and “bucket
fragmentation” are not valid categories.

## Mapping rules

1. Preserve dimensions and reference bases. Numerator, denominator, time, area,
   mass, tissue, dose, species, and reference population are part of the unit.
2. Normalize aliases or convert units only when the conversion is exact and
   valid for every affected row. Do not assume molecular weight, density,
   geometry, exposure duration, or another missing quantity.
3. Apply `scale` exactly once to the extracted coefficient. Store it as a finite,
   positive fixed-point decimal, not scientific notation.
4. Treat powers of ten as part of the unit interpretation. A table heading such
   as a reported coefficient multiplied by `10^6` may require the inverse scale;
   notation alone is insufficient. Review all source contexts first.
5. Retain the log base, sign convention, and underlying linear unit. Changing a
   logarithmic unit can require an additive offset, so it is not represented as
   a multiplicative `scale`. A bare `log Kp` is ambiguous if its underlying axis
   cannot be established.
6. Convert percent, fraction, ratio, or fold only when their reference and
   direction are equivalent across the entire exact key. A conversion is not
   required merely to merge buckets.
7. Choose the numeric domain from measurement semantics. Use `any` for valid
   signed values, log values, or changes; `nonnegative` or `positive` only when
   zero or negative values are physically invalid for that quantity.
8. If a useful literal unit cannot be reconciled with other units, map it to its
   own canonical spelling. Do not discard it.

## Required review procedure

For each task:

1. Freeze and record hashes for Stage 01, the extraction mapping, the current
   exact unit map, and the immutable source files or manifests.
2. Enumerate the complete active key set from both source-exact tuples and
   successful LLM-extracted tuples.
3. For every new, changed, or excluded key, collect every affected Stage-02 row.
   Join each row back to the immutable full source record using stable source
   identifiers and row numbers.
4. Inspect all source columns that can establish meaning, including endpoint,
   measurement, unit, assay description, table/header context, qualifiers,
   support text, and provenance. Do not decide from the normalized row alone.
5. Search the complete affected set for counterexamples. Repeated homogeneous
   rows may be reviewed as groups, but a sample is not evidence that one rule is
   safe for all rows.
6. Apply the primary decision rule. If affected rows require different rules,
   exclude the exact key as `mixed_exact_key`; do not add source-specific or
   per-record exceptions.
7. Remove task entries with no active source-exact or successful extracted
   tuples. An inactive legacy rule has no current evidence and a future unseen
   key should fail closed for review.
8. Sort the complete production entries deterministically by task, endpoint,
   and input unit.

The reviewer should retain a task-specific audit artifact containing the full
affected rows, source joins, decision, exclusion category when applicable, and
input hashes. This audit is evidence for the JSON decision; it is not another
runtime mapping.

## Validation gates

A mapping is ready only when all of the following hold:

- The JSON version is `starling_exact_measurement_units.v1`, it loads through
  `load_exact_unit_mapping`, and it has no empty or duplicate exact keys.
- The task's map keys equal its complete active key set after inactive entries
  are pruned. Missing active keys are build errors.
- Every `map` rule has a nonempty canonical unit, finite positive fixed-point
  scale, and one allowed domain: `any`, `nonnegative`, or `positive`.
- Every `exclude` rule has null output unit and scale and a full-row audit tied
  to one valid exclusion category.
- Every changed decision has complete effect rows and immutable source evidence,
  not only a count or sampled example.
- Reapplying the current map is deterministic. Candidate comparisons report
  scalar gains, losses, value changes, unit changes, and row explosions.
- Retrieval-visible source text and retrieval membership do not change.
  Unit-specific Stage-04 bucket counts may change and are reported as outcomes,
  never used as mapping objectives.
- Any downstream label or threshold policy remains separate. Mapping a scalar
  does not invent reference semantics or make it eligible for a gold label.

## Minimal examples

- **Map:** an explicit but rare `% dose/cm^2` unit. Its rarity and new bucket do
  not make it ambiguous.
- **Map separately:** two dimensionally honest units under one endpoint when no
  exact conversion is justified.
- **Exclude:** `10^-6` without a physical numerator, denominator, or time basis.
- **Exclude:** rows sharing one exact `ratio` key where the full records show
  incompatible reference directions.
- **Exclude:** a nominal `log(cm/s)` key when the complete affected set contains
  inconsistent coefficient extraction that one exact rule cannot repair.

The governing principle is scientific interpretability per exact key. Bucket
compactness is downstream organization and must not influence map or exclude
decisions.
