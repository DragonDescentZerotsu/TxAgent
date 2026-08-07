# Starling normalization manual audit

Audit date: 2026-08-05

## Scope

This audit reviews 1,500 real Stage-01 records: 500 each from BBB_Martins,
Bioavailability_Ma, and Skin_Reaction. The permanent executable corpus is
`tests/chembl_tool/common/fixtures/normalization_manual_audit.jsonl`; it contains
the exact Stage-01 source view, resolved structure state, reviewed canonical
output, pair-bucket decision, and rationale for every case. There is no fixture
writer or expectation-update mode in the test suite.

Sampling is deterministic and stratified by source, accepted/rejected status,
measurement kind, categorical encoder, and reviewed semantic rule. It includes:

- BBB: direct binary/continuous records, passive permeability, efflux substrate
  and inhibitor outcomes, influx abstentions, and all nine newly identified
  directional-context exclusions.
- Bioavailability: all 121 accepted direct-HF qualitative phrases, direct and
  nondirect HF numeric/rejected forms, every Fg scalar-rule family, Fa/Fh/oral
  exposure boundaries, and all 16 newly identified directional-context
  exclusions.
- Skin: every categorical scale, all signed phototoxicity directions, declined
  labels and parser edges, plus sampled coverage of every reviewed
  sensitization and exposure semantics rule.

The audit also reviewed the complete 219-value Bioavailability endpoint
inventory and the complete current-v7 census of 73 Skin Stage-01 value changes.

## Findings

Of the 1,500 reviewed cases:

- 642 are accepted atomic or controlled measurements.
- 462 are precision-preserving abstentions.
- 121 use reviewed complete-phrase Bioavailability qualitative encodings.
- 250 exercise reviewed Skin measurement-semantics rules.
- 25 expose one confirmed false-promotion class.

The confirmed defect was a numeric prefix followed by directional or comparator
text that was not represented in the canonical unit. Examples include
`62 ± 3% decrease` and `2.3 ± 0.8-fold higher`. Treating those prefixes as
ordinary scalars erases polarity or comparator identity and incorrectly permits
pair bucketing. The affected transfer-eligible census is 9 BBB records, 16
Bioavailability records, and 0 Skin records.

The shared normalizer now retains the display measurement but sets the scalar to
null and marks `measurement_unit_status=non_atomic_directional_context`. The
gate runs again after task-specific enrichment so a contextual-unit resolver
cannot restore the scalar. Explicit controlled directional units remain valid:
for example, a source `% increase` or `% reduction` unit keeps direction in the
canonical unit and pair identity.

No additional false promotions were confirmed in the reviewed qualitative,
categorical, decimal-comma, endpoint, or measurement-semantics strata. The
audit deliberately leaves ambiguous and unrecognized forms non-scalar rather
than adding aggressive parsing rules.

## Regression vault

The audit is enforced at three levels:

1. `normalization_regressions.jsonl` contains small, hand-explained real-record
   cases for the defect and its controlled-unit boundaries.
2. `source_value_cleaning_audit_corpus.jsonl` now contains 5,073 cases,
   including all 73 current Skin decimal-comma changes.
3. `normalization_manual_audit.jsonl` replays the complete 1,500-case Stage-02
   canonicalization and Stage-04 pair-bucket contract.

## Final single rebuild

One final Stage-02-through-index rebuild was run for each task after the audit
fixture passed. All three manifests list Stages 01-09 complete, and all 1,500
fixture cases match the persisted Stage-02 and Stage-04 rows exactly.

| task | eligible records before | after | delta | buckets before | after | delta |
|---|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 370,409 | 370,400 | -9 | 55,668 | 55,663 | -5 |
| Bioavailability_Ma | 265,908 | 255,177 | -10,731 | 8,098 | 8,092 | -6 |
| Skin_Reaction | 347,463 | 347,463 | 0 | 23,630 | 23,630 | 0 |

The BBB delta is exactly the nine newly rejected direct directional-context
records. Bioavailability combines the 16 directional-context exclusions
(Fa -2, Fh -14) with the previously implemented but not yet rebuilt strict HF
source parser (HF -10,715); the latter rejects ambiguous/unitless direct and
nondirect values rather than recovering units or polarity from context. Skin
does not change. No directional-context record is eligible in any final
sidecar.

Neighbor-index molecule counts are unchanged for BBB (36,717 random; 36,695
scaffold) and Skin (18,752; 18,744). Bioavailability becomes 27,065 random and
27,061 scaffold; its membership increase reflects the already implemented v7
direct/nondirect evidence-group wiring that this rebuild finally materialized,
not promotion of the newly rejected scalar records.
