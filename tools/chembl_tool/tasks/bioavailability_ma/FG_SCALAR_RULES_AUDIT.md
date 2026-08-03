# FG single-outcome scalar rules: implementation audit

This audit documents the scalar extraction rules integrated for the `fg`
source in `starling_normalized_v5`. The source-facing measurement and unit
fields remain unchanged; the normalized, organized, indexed, pair-bucket, and
endpoint-policy v2 artifacts were rebuilt after the rules passed manual
review.

## Acceptance rules

A measurement is eligible only when all of the following hold:

1. The outcome has an explicit semantic label, or the text is an atomic value
   with an embedded physical unit.
2. Absolute `Fg` and combined `Fa·Fg` values remain distinct semantic outcomes.
3. A physical measurement carries its unit in the measurement text.
4. The complete measurement text describes exactly one outcome.
5. Compound measurements are rejected and are never split.
6. Bounds and ranges remain non-scalar.
7. The proposed value passes the semantic domain checks for its outcome and
   unit.

The source `measurement_text` and `unit_text` remain unchanged. A proposal adds
only a derived canonical measurement, canonical unit, and finite scalar value.
Dimensionless ratios use canonical unit `ratio`; fold changes use `fold`; and
bounded fractions such as absolute `Fg` use `fraction`.

## Full-source result

The current v2 rules produced syntactically valid proposals for 2,966 of 27,713
FG rows (10.70%). Of these, 2,465 have a resolved canonical molecule and 501
have an unresolved structure. The organization stage already requires both
`canonical_smiles` and `group_id` for retrieval eligibility, so those 501
unresolved rows cannot become molecule-level retrieval evidence.

| Rule | Accepted rows |
|---|---:|
| Explicit ratio | 1,066 |
| Embedded physical unit | 806 |
| Single percentage | 682 |
| Explicit fold | 288 |
| Explicit `Fg` or `Fa·Fg` fraction | 124 |

The machine-readable full audit is under
`outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v5/analysis/fg_scalar_rules_v2/`.

## Post-review v2 corrections

The v2 review fixed two persistence/safety defects found after the initial rebuild:

- All 505 accepted FG physical measurements with explicit scientific notation now retain both
  `unit_notation_status=unambiguous_scientific_notation` and their non-null notation factor. The canonical
  numeric values were already folded correctly; this restores the missing audit provenance.
- Eight coordinated or compound outcomes involving paired directions/sites/populations/conditions, or newly
  recognized `K_a`, `P_app`, and flux metric names, now remain non-scalar. Their original source text remains
  retrieval evidence and no outcome is split or discarded.

This changes the normalized FG scalar count from 2,974 to 2,966 and the organized count from 2,972 to 2,964.
It does not change the total retained-record, retrieval-eligible, evidence-row, or molecule counts.

## Second manual validation sample

A second, non-overlapping sample of 100 proposed accepted rows was reviewed,
stratified across ratio (25), physical-unit (25), percentage (20), fold (15),
and `Fg`/`Fa·Fg` fraction (15) rules. The sample seed was `20260801`.

The initial review found 95 clean proposals, four rule-level failures, and one
source-level inconsistency.

The four rule-level failures were:

| Source row | Measurement | Problem | Revised disposition |
|---:|---|---|---|
| 13142 | `13.4 ± 1.6 µmol/100 cm·h` | Numeric denominator `100` was lost | Reject: unsupported numeric unit denominator |
| 24885 | `2.0 ± 0.04 nmol/min per 100 mg tissue` | Numeric denominator `100` was lost | Reject: unsupported numeric unit denominator |
| 12905 | `0.28 ± 0.18 nmol min^-1 mg^-1 protein` | Ambiguous protein-normalization notation | Reject: ambiguous protein normalization |
| 20198 | `4.18 µmol min-1 mg protein-1` | Ambiguous protein-normalization notation | Reject: ambiguous protein normalization |

The revised rules reject these four cases and also conservatively reject 42
other rows with the same unsafe forms: 15 rows have a numeric `/100` or
`per 100` denominator, and 31 have ambiguous inverse-protein notation.

Source row 11010 remains an explicit scalar proposal because its measurement
text states `efflux ratio 5.8`. Its support text reports Papp values of 39 and
43 nm/s, which imply a ratio near 1.1, while also explicitly repeating 5.8.
This is an internal source-data conflict. The extractor must not silently
replace the reported outcome with a value recalculated from a compound support
sentence; the row requires a separate source-QA flag if these proposals are
integrated.

After revision, 99 of the 100 reviewed rows have an appropriate automatic
disposition. The remaining row is not resolvable by scalar parsing alone
because the source record is internally inconsistent.

## Third manual validation sample

A third sample of 100 proposed accepted rows was reviewed with 20 rows from
each rule family. The deterministic selection seed was `20260802`; the
persisted first sample and the five specifically adjudicated rows from the
second review were excluded.

Reviewed source rows:

- Ratio: `12844, 3849, 4458, 2923, 12487, 10805, 4581, 7093, 2965, 15639, 26502, 2910, 11054, 5137, 13207, 12814, 1588, 17503, 10167, 1983`
- Physical unit: `13385, 24782, 17705, 16264, 17340, 2742, 11417, 10104, 25236, 717, 24786, 26494, 8392, 10042, 20820, 11408, 27231, 24833, 17889, 21424`
- Percentage: `7381, 2274, 9518, 17816, 11431, 7302, 26211, 4624, 3498, 2240, 22759, 5606, 18189, 2275, 24459, 24648, 6041, 26064, 19521, 3883`
- Fold: `18700, 19185, 16110, 121, 13651, 22781, 25223, 12978, 3661, 6839, 18666, 22780, 19048, 6838, 134, 11478, 4743, 25938, 24383, 26447`
- `Fg`/`Fa·Fg`: `770, 23365, 10866, 16881, 16218, 3114, 10861, 23698, 20578, 3112, 10865, 3768, 16220, 10855, 19491, 772, 1967, 771, 23702, 16880`

The initial review found 98 clean proposals and two compound-outcome failures:

| Source row | Measurement | Problem | Revised disposition |
|---:|---|---|---|
| 9518 | `~60% reduction in peak plasma concentration and bioavailability` | One percentage is attached to two outcomes | Reject: compound or multi-outcome |
| 4743 | `AUC0^-∞ and Cmax increased ~2.5-fold` | One fold value is attached to two outcomes | Reject: compound or multi-outcome |

The compound detector now recognizes indexed AUC notation such as `AUC0` and
the explicit `peak plasma concentration` outcome. These two rows are rejected
without splitting or choosing one of the outcomes. After revision, all 100
reviewed rows have an appropriate automatic disposition.

## Fourth manual validation sample

A fourth balanced sample was reviewed with 20 rows from each rule family. The
deterministic selection seed was `20260803`, and all previously recorded
reviewed source rows were excluded.

Reviewed source rows:

- Ratio: `12135, 946, 18175, 27394, 16771, 26485, 17611, 16440, 7108, 18041, 26495, 13833, 5211, 23358, 2937, 19292, 14623, 15624, 25213, 19189`
- Physical unit: `17342, 10402, 23098, 5191, 20657, 14970, 4016, 7158, 9245, 18425, 17707, 8736, 6879, 18432, 16766, 5058, 7628, 5445, 18929, 18424`
- Percentage: `17726, 17440, 27425, 6032, 8600, 22055, 4687, 744, 5608, 17033, 10218, 4793, 22763, 14723, 16580, 12052, 27599, 26733, 14268, 15771`
- Fold: `6843, 17772, 182, 24380, 6801, 23073, 6555, 9944, 2430, 16920, 6554, 25741, 10813, 21185, 18913, 2429, 22943, 1357, 16428, 17332`
- `Fg`/`Fa·Fg`: `8910, 16906, 16381, 3111, 16886, 15729, 10604, 13980, 23704, 22026, 20579, 10867, 23703, 10854, 6557, 3769, 18862, 16219, 16073, 8480`

The review found 99 clean rows and no new scalar-parser failure. Source row
18175 is an internal source-text contradiction: `measurement_text` explicitly
reports `efflux ratio 44.27`, and the support interprets this as strong efflux,
but the same support sentence says apical-to-basolateral permeability was
higher than basolateral-to-apical permeability. That direction statement is
incompatible with a conventional large efflux ratio.

The scalar proposal remains appropriate under the measurement-only extraction
contract. The extractor must not silently invert or recalculate it from prose;
the row requires a separate source-QA conflict flag if the proposals are
integrated. Thus all 100 rows have the appropriate automatic parser
disposition, with one row requiring source-level review.

## Fifth manual validation sample

A fifth balanced sample was reviewed with 20 rows from each rule family. The
deterministic selection seed was `20260804`, and all previously recorded
reviewed source rows were excluded.

Reviewed source rows:

- Ratio: `7334, 4598, 5102, 17504, 6017, 1582, 5561, 27393, 19290, 13219, 25469, 15977, 26497, 7100, 13067, 2723, 14913, 11422, 21272, 25214`
- Physical unit: `13594, 4009, 9242, 6288, 25175, 8009, 7686, 10300, 16767, 26708, 16736, 21575, 13301, 8809, 24041, 10103, 5265, 13303, 18928, 8093`
- Percentage: `3819, 707, 2501, 7379, 20917, 17810, 6572, 2287, 6964, 18188, 2507, 6771, 11816, 10837, 20443, 11837, 19210, 25295, 19839, 21991`
- Fold: `23628, 7567, 14936, 20314, 20458, 3526, 14423, 6657, 21566, 22745, 16754, 17470, 11308, 21438, 18609, 16669, 24384, 3578, 19107, 24379`
- `Fg`/`Fa·Fg`: `9991, 23700, 16503, 10856, 21154, 23688, 10859, 16222, 10864, 3767, 23706, 16221, 3985, 16876, 8965, 16902, 16378, 24618, 16905, 18851`

All 100 measurement strings received an appropriate scalar-parser disposition,
and no new syntax rule was required. One row exposed an integration-boundary
case: source row 24618 reports `average f_g 97%` for the aggregate category
`ECCS class 3A` across 368 compounds. It has no canonical SMILES and its
normalization status is `unresolved_structure`.

The `97%` is a valid aggregate scalar but is not a molecule-specific
measurement. It must remain retrieval-ineligible under the existing
organization gate; it should not be counted among the 2,465 resolved-molecule
proposals that could supply continuous FG evidence.

## Verification

The complete Bioavailability task and shared normalization/contract checks pass with 399 tests. The rebuilt
normalized artifact contains 2,966 FG finite scalars; organization removes two unresolved duplicate scalar
rows, leaving 2,964 organized FG scalar rows. All 2,465 resolved-molecule scalar rows remain
retrieval-eligible. The v2 audit also verifies persisted rule parity and scientific-notation provenance for
every FG row.
