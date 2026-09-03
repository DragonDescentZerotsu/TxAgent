# BBB V8 measurement-resolution rule review

V8 Stage 1 has been built, but no full V8 extraction or downstream library build
has been run. The rules below are the complete routing policy. Their definitions live in
`measurement_resolution_rules.py`; `ROUTING_RULES` is the sole precedence list.

`accept` means that the source coefficient and unit can be copied without an
LLM. It does not make the row absolute or transfer-eligible. `reject` means only
that scalar extraction is skipped; the source row remains available for the
independent categorical encoder or as free text.

Counts below come from all 498,640 BBB V8 Stage-1 rows.
They are precedence-adjusted, so every row is counted under exactly one rule.

## Exact-copy rules

| order | decision | function | rows | review | rationale |
|---:|---|---|---:|---|---|
| 1 | accept | `exact_finite_number_with_separate_unit` | 16,500 | approved 2026-09-01 | The complete field parses as a finite number and its separate unit occurs in the shared observed-unit vocabulary. Zero and negative values are copied; domain validity is decided later. |
| 2 | accept | `exact_direct_logbb` | 2,981 | approved 2026-09-01 | A bare direct `logBB` coefficient is assigned the explicit canonical unit `log10_ratio`. |
| 3 | accept | `exact_number_percent` | 5,349 | approved 2026-09-01 | The complete field is a number plus `%`; a separate unit must be absent or percent-equivalent. |
| 4 | accept | `exact_point_with_uncertainty_and_unit` | 16,251 | approved 2026-09-01 | Full-field match only: `(number) ± (number)` or `(number) +/- (number)`, with an observed separate unit. Prefixes, suffixes, and other punctuation fall through. |
| 5 | accept | `exact_percent_with_uncertainty` | 553 | approved 2026-09-01 | For `45 ± 5%`, copy the point and `%` when no conflicting source unit exists. |
| 6 | accept | `exact_scaled_number_with_separate_unit` | 1,165 | approved 2026-09-01 | Full-field match only: for `3.1 × 10^-6` plus an observed `cm/s`, preserve the coefficient and printed factor without rescaling. |
| 9 | accept | `exact_embedded_number_and_unit` | 483 | approved 2026-09-01 | Full-field value-plus-unit match only, with no separate unit. The complete cleaned unit phrase must occur in one of the seven declared source unit columns. Labels and added prose fall through. |

## Deterministic scalar-rejection rules

| order | decision | function | rows | review | rationale |
|---:|---|---|---:|---|---|
| 7 | reject | `reject_hard_bound` | 2,874 | pending | Full-field bound only: `<5`, `≥2`, and similar exact forms do not state a point scalar. Added prose or an inline physical unit falls through. |
| 8 | reject | `reject_explicit_range` | 6,084 | pending | Full-field interval only: an exact form such as `7-12%` has no unique point to copy. |
| 10 | reject | `reject_no_digit` | 343,185 | approved 2026-09-01 | Digit-free measurement text contains no numeric coefficient. |
| 11 | reject | `reject_no_standalone_number` | 11,018 | approved 2026-09-01 | Any digits belong only to identifiers or explicit notation, such as `GLUT-1/SLC2A1`. |
| 12 | reject | `reject_multiple_independent_numbers` | 22,755 | approved 2026-09-01 | After removing strictly written uncertainty and scale exponents, more than one coefficient remains and the router will not choose among them. |

## Rows left for V8 extraction

The strict policy leaves 69,442 rows for the LLM. Endpoint/source grouping makes
3,643 requests at a maximum batch size of 20. V8 sent 94,820 rows.

| source | reject | exact | LLM |
|---|---:|---:|---:|
| direct_bbb | 226,783 | 39,945 | 35,820 |
| efflux_transport | 120,564 | 0 | 27,736 |
| influx_transport | 37,433 | 47 | 5,575 |
| passive_permeability | 1,136 | 3,290 | 311 |

## Removed V8 behavior

- **Categorical routing:** Stage 1 no longer imports or runs the categorical
  encoder. Stage 2 still applies it when no resolved scalar is present, so no
  categorical evidence is deleted and a scalar row does not create a sibling.
- **Positive-only exact gate:** V8 sent zero and negative finite values to the
  LLM. V8 copies them and leaves endpoint-domain validation downstream.
- **Broad influx input rejection:** V8 rejected 96 rows under this rule. Eighty-
  four were `Km`, eleven were `Kt`, and one was a generic concentration. `Km` and
  `Kt` are fitted measurements, so this rule is absent from V8.
- **Condition-only rejection:** the first V8 draft contained a broad textual
  condition rule, but it matched no current rows and was removed as imprecise.
- **Ambiguous power-of-ten regex:** V8 could interpret ordinary integers such as
  `100`, `109`, and `1052` as powers of ten. V8 requires an explicit scientific-
  notation marker.
- **Loose partial matches:** bounds, ranges, uncertainty, scale notation, and
  atomic inline values now require complete-field syntax. Ambiguous cases go to
  the LLM even when that reduces deterministic coverage.
- **Generic unit parsing and label allowlist:** an embedded pair no longer passes
  because a parser recognizes its tokens or because it starts with Km/Kt/Ki/
  Vmax/Tmax/Papp. It passes only when the complete unit phrase is present in the
  shared 12,721-value observed-source vocabulary.

## Deliberately not added

- No generic endpoint-defined-ratio shortcut: a nominal brain/plasma endpoint
  can hide an outer knockout/control comparison.
- No generic condition-word search in support text: typed scalar source columns
  should already isolate the measurement, while influx `reported_result` is
  intentionally prose and should use the LLM when mixed.
- No approximate-value shortcut yet: `about 5` and `~5` remain in the LLM pool
  until explicitly approved.
