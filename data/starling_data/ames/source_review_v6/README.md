# Ames record family/level repair

This continues the interrupted v5 placement audit. It judges the contents of
existing source records, not whether the original papers or extractions are
authentic. No external source or name-resolution requests were made.

The two audits cover **394 distinct records**: the previous 200 plus 194 here.
The new sample contains 140 unread records from the v5 candidate pool, in frozen
pool order within level/source strata, and 54 checks of the draft rule's proposed
transitions (source IDs ranked by SHA256 within each transition). These are
deliberately enriched for family boundaries, not a population error-rate sample.
The transition checks include proposed changes rejected during development;
their annotation describes the intended final family, not the draft output.

## Authored decisions

| New decisions | Records |
|---|---:|
| Retain | 144 |
| Relocate | 46 |
| Unresolved family boundary, retain prior placement | 4 |
| Total individually read | 194 |

Together with the 49 v5 relocations, **95 explicit relocations** are applied.
`placement_decisions.jsonl` has 100 entries: these relocations plus five
case-specific retention guards that preserve individually reviewed semantics. Every entry pins the complete raw payload and
canonical scientific card surface. It cannot grant retrieval eligibility, alter
an identity exclusion, create a vote, or alter a gold candidate.

The production change also assigns indirect evidence by source-independent
endpoint classes and assay readouts. The automatically changed population is
reported separately in `family_changes.parquet` and `summary.json`; those rows
are not counted as individually reviewed. The ledger includes newly recoverable
endpoint records, if any, separately from movements of existing records.

## Semantic boundaries

- Ames author citations, Ames dwarf mice, and statements that an assay was not
  performed are distinct from an Ames-negative outcome.
- Plant, yeast, animal and mammalian-cell reversion are not bacterial reverse
  mutation. A bacterial transgene in an animal does not change the host organism.
- Bacterial repair reporters and repair-dependent survival are distinct from
  revertant outcomes. An actual accompanying bacterial result keeps the complete
  record at L2, including when it concerns another molecule.
- DNA lesions, repair synthesis, repair kinetics and DDR biomarkers belong at L4;
  comet called "clastogenicity" and PAR intensity inside a micronucleus do not
  become chromosome-aberration or micronucleus-frequency assays.
- Other fixed genetic and chromosome outcomes belong at L3. Metabolic enzymes,
  protein/GSH adducts, redox and topoisomerase machinery belong at L5 unless a
  more direct readout is actually supplied. Prediction status and modifier roles
  are preserved and do not independently lower or exclude evidence.

The independent validator retains its broad bacterial-content screen for every
unreviewed record. It verifies each authored exception against original Parquet
row content and the unchanged complete canonical/index card surface before
allowing an apparent lexical hit outside L1/L2. Manual group edits alone cannot
pass. This deliberately leaves other ambiguous bacterial mentions conservative;
the repair is not a claim that all remaining placements have been read.

## Preserved and unresolved

Gold source votes, all published benchmark files and the 3,333 actual L1 voters
are preserved. No split allocator or reasoning model is run. The two previous
v5 gold scope/condition boundaries and eight gold-recall candidates remain open;
this placement-only revision does not grant them L1 membership. Four v5 and four
v6 indirect-family boundary decisions retain their prior placements with their
uncertainty recorded. Previous v4 source/identity decisions remain active.

`summary.json` records the final source/index lineage, applied movement counts,
raw and canonical review checks, and preservation/validation results.
`previous_*_manifest.json` pins the inputs at the start of this revision.
`sample.jsonl` retains all semantic fields from each original reviewed payload;
`authored_judgments.tsv` and `audit_annotations.jsonl` contain individual reasons.
The source-file stem plus zero-based Parquet ordinal is the source identity;
`extraction_id` alone is not a unique key.

## Rebuild and verify

From the repository root, with existing project dependencies:

```sh
conda run -n vllm python -m \
  tools.chembl_tool.tasks.ames.build_dataset --phase refresh-retrieval --workers 128
conda run -n vllm python -m \
  tools.chembl_tool.tasks.ames.validate_retrieval
conda run -n vllm python -m pytest \
  tests/chembl_tool/tasks/ames
```

The refresh requires byte-identical votes and benchmark files. Both heldout-filtered
indices are rebuilt or reused after dependency and output hash checks. The independent
validator checks full source/index card correspondence and cumulative/progressive
retrieval for scaffold and random. The placement regression suite compares the
builder policy with authored judgments and rejects changed-content exceptions.

`retrieval_build_performance.json` separates the 23.95x aggregation microbenchmark
from whole-build and cache-hit timings. Source processing, complete assay-molecule
groups, standardization and ordered JSONL serialization run in bounded process pools.
The two split builds share a 128-worker budget (64 each); native libraries use one
thread per worker. Verified inputs and temporary outputs use node-local NVMe;
identical outputs skip NFS publication. Final artifacts retain canonical paths.

Stage reuse checks input, policy, review, runtime dependency and output hashes.
The digest cache also checks complete file metadata and host boot identity;
the independent audit initially hashes published files without producer digest reuse.
Explicit `source`/`all` phases reconstruct the source; a no-op refresh preserves
the dataset manifest and its validation lineage. Cache location and controls are
documented in `tools/chembl_tool/tasks/ames/README.md`.
