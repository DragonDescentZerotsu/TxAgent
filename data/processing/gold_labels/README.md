# Gold-label processing

This package owns benchmark labels, conditioned split contracts, condition-review
logic, and benchmark publication. It consumes evidence-library outputs but is not
part of evidence-library normalization or pair-bucket construction.

The active versioned scaffold releases are resolved through
`data/gold_labels/<Task>/CURRENT`. BBB_Martins and Bioavailability_Ma resolve to
v1; Ames, DILI, Carcinogens, and Skin_Reaction v1 are byte-exact imports from upstream commit
`26d44f121141f3738764c0b00dc09e185445341b`; Skin v1 replaces the earlier local
cohort. Reproduce that import with:

```bash
python -m data.processing.gold_labels.import_upstream_conditioned_benchmark
```

The current general entry point is `build_conditioned_benchmark.py`. Historical
record-supported publication remains isolated in
`build_record_supported_benchmark.py` and does not participate in V7 or V8 builds.

## Small validation views

Each active Gold release exposes `valid_small.jsonl` and its aligned
`valid_small_molecule_condition_labels.jsonl` as an additive, 100-row view of
`valid`. Complete scaffold groups are selected so the view has no scaffold
overlap with the omitted validation rows; size is optimized first and validation
label-ratio fidelity second. For Ames, Carcinogens, and DILI, distinct-parent
count is maximized before the label-ratio tie-break to increase molecular
coverage. The original `valid` split and held-out union remain unchanged. Rebuild
and verify the six views with:

```bash
python -m data.processing.gold_labels.build_valid_small
```

The adjacent `valid_small_manifest.json` pins the source and output hashes,
selection seed, counts, and disjointness result.

Gold-version level mappings are evidence-library outputs published under
`data/gold_labels/<Task>/level_mappings/<version>/`. This package may expose the
release's voter-membership contract for validation, but it must not construct,
promote, demote, or otherwise assign evidence levels.

## BBB and Oral conditioned v2 rebuild

`build_v1_voter_lineage.py` first recovers the closed v1 voter universe. BBB v1
already names physical rows. Oral v1 names canonical claims, so the builder
replays the hash-pinned historical claim artifact and expands each selected claim
to its distinct HF/local physical members. Stage-1 exact deduplication prefers a
v1 physical member over a non-gold duplicate, then uses the smallest UID.

`build_repaired_v2.py` admits only surviving v1-lineage members, applies the
current row label adapter as a hard eligibility gate, and drops failures with an
uncapped `provenance/v1_lineage_label_rejections.jsonl` audit. It requires one
payload-bound condition decision for every remaining physical voter, aggregates
by normalized parent and exact condition, preserves surviving v1 split/row
identities, applies the existing 70% null-condition and 60% external-condition
vote thresholds and group gates, and writes uncapped `voter_membership.parquet`
provenance. It also writes `gold_label_record_index.parquet`, one row per
published card with its sorted physical `source_row_uids`, vote counts, and
unchanged voter mean; consumers can denest those UIDs through the edge table or
join them directly to Stage 3. The membership `vote_label` is authoritative for
gold reconstruction; consumers must not infer it from a Stage-3 display or
calibration field. A v1 null-condition context whose majority label is unchanged may
use a two-thirds migration-preservation floor; exact ties remain rejected. This
changes only card publication, never the physical voters or their vote mean.
Other label-eligible Stage-1 rows never enter gold v2.

```bash
python -m data.processing.gold_labels.build_repaired_v2 \
  --stage1-root /local/joseph/txagent-v10-stage1-gold-priority-v3 \
  --condition-ledger-root <reviewed-ledger-root> \
  --voter-lineage-root \
    data/artifacts/gold_labels/conditioned_benchmark/v2_voter_lineage \
  --v1-root data/gold_labels \
  --output-root <staged-output-root>
```

The builder preflights both tasks before writing either v2 directory. Missing
Stage 1 deduplication receipts, hash or UID drift, an inherited voter missing
without an inherited duplicate survivor, stale condition payloads, cross-source
condition inheritance, or incomplete eligible-voter coverage produce only
`v2_build_blockers.json` and exit 2. The reviewed releases passed publication
validation on 2026-09-16 but remain inactive while both tasks use `CURRENT=v1`.
The aggregation gate runs before split assignment, so it applies to train,
valid, and test: 70% for null-condition cards, 60% for external-condition cards,
plus the documented two-thirds migration-preservation exception above.
