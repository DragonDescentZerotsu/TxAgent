# AMES V10 measurement-candidate blind audit v3 protocol

## Purpose and immutable boundary

This is a new generalization audit for the candidate generator created after the
frozen v2 audit failed. V2 is development evidence only; its sample, reviews,
labels, and score cannot be changed or rescored. V3 uses a new generator,
candidate artifact, audit version, seed, and source-only sample that is disjoint
from the canonical 500-case gold fixture and both prior blind samples.

The v3 seed is
`ames-v10-candidate-blind-audit.v3|20260908|post-v2-general-repair`. The audit
version is `ames_candidate_generalization_blind_sample.v3`. Any change to the
seed, frozen inputs, selection logic, sample, label policy, or scoring policy
requires another audit version.

## Freeze and access sequence

1. Accept one complete candidate artifact only after its generator passes an
   independent full-corpus semantic review. Pin the artifact, generator, writer,
   compiler contract, source-field contract, and Stage 1 record SHA-256 values.
2. The builder verifies all hashes and recursively rebuilds the candidate
   inventory before sampling. It refuses an existing sample, template, or
   manifest.
3. Selection uses only Stage 1 identities, source fields, canonical endpoints,
   and source-text complexity strata. Candidate values, units, rules, counts,
   dispositions, model outputs, v2 labels, and v2 scores cannot affect selection.
4. Before adjudicated labels are frozen, generator authors, root agents, code
   reviewers, and automated reviewers cannot open the sample, review template,
   reviewer outputs, adjudication, or final labels. Designated scientific
   labelers see source-only cases and never see candidates, gold labels, model
   outputs, or another review.
5. Freeze two independent reviews, source-only adjudication, final labels, and a
   label manifest before candidate scoring. Only then may an analyst join cases,
   labels, and the already-frozen candidate artifact.
6. Score exactly once. Never modify or rescore this generator after inspecting
   v3 cases or labels. A failed gate requires a new generator and a new disjoint
   audit version.

## Population and exclusions

Select exactly 30 Stage 1 extraction rows from each of `fixed_mutation`,
`mutagenicity_mechanism`, and `premutagenic_damage`, for 90 total. Exclude every
cleaned record ID, source row UID, and source-field payload hash represented in:

- the canonical 500-case AMES gold fixture;
- the 90-case blind v1 sample; and
- the frozen 90-case blind v2 sample, SHA-256
  `bec6e2d0004b58114214ec4f44300eaccda3c2e62688df272fa4bad1e60e3bb8`.

Map every exclusion identity back to the pinned Stage 1 records and require
one-to-one coverage. Deduplicate the remaining population globally by canonical
source-field payload hash. Within each source, cover every available canonical
endpoint and primary source-text complexity stratum, then balance endpoint and
stratum counts using the frozen seed. Candidate identity fields may be used only
to verify that every selected row exists in the frozen artifact.

## Source-only label contract

Each reviewer preserves `audit_case_id` and `review_index`, sets a unique
`reviewer_id` and `review_status`, and fills `reason`, `status`, `measurements`,
`measurement_evidence`, `unit_evidence`, and `notes`. Measurement entries are
objects with string fields `measurement` and `unit`.

| reason | status | measurements |
|---|---|---|
| `absolute` | `ok` | exactly one source-supported pair |
| `experiment_relative` | `relative` | `[]` |
| `missing_unit` | `unsure` | `[]` |
| `bound_or_range` | `unsure` | `[]` |
| `multiple_or_conflict` | `unsure` | `[]` |
| `no_eligible_numeric` | `unavailable` | `[]` |

Apply this precedence: no eligible endpoint number; bound or range; unresolved
multiple or conflict; incomplete unit or scale; experiment-specific external
comparator; otherwise one absolute complete named readout. Do not calculate,
convert, average, round, or borrow wording from another row. Evidence fields are
exact same-row source substrings or null.

For an absolute label, preserve the printed numeric coefficient as a string.
Scoring treats standard thousands separators as formatting: for example,
`60,000` and `60000` are the same coefficient. All other numeric content must be
equal as a finite decimal. The unit/readout must match exactly after trimming
outer whitespace; no unit conversion, inferred scale, synonym, or alias is
accepted in this candidate-recall audit. This scoring rule is frozen before v3
sampling and labeling.

Adjudication requires initial exact agreement on reason, status, measurement,
and unit. Disagreements are resolved from source-only evidence. The final label
manifest pins the protocol, sample, sample manifest, both reviews,
adjudication, and final-label hashes and records row counts, source/status/reason
counts, reviewer identities, agreement rate, validations, and blindness claims.

## One-time score gates

The scorer first revalidates and rebuilds the complete frozen candidate inventory.
For each adjudicated `absolute` row, a hit requires a numerically equivalent
coefficient under the frozen thousands-separator rule and the exact unit/readout.
Require all of the following:

- at least 95% exact absolute-pair recall overall;
- at least 90% exact absolute-pair recall within every source having an
  adjudicated absolute case;
- zero evidence-grounding failures across every candidate in the 90 sampled
  rows; and
- the recursively validated inventory's no-truncation, candidate-cap, byte-cap,
  identity, hash, and complete Stage 1 coverage checks all true.

The score writes immutable aggregate and row-level receipts. It cannot be rerun
against v3 after labels are visible.

