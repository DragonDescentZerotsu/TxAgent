# Ames conditioned data

The target is **an experimental bacterial reverse-mutation result under the
reported strain panel and metabolic activation regime**. It is not general
genotoxicity, chromosome damage, carcinogenicity, or a regulatory classification
across an unreported complete test battery.

The adapter uses the existing conditioned publisher, scaffold/random allocators,
record-family catalog and assay retrieval implementation. It does not launch a
reasoning model. Ames evaluation uses the shared progressive/full-flat runners and
level plotter; current results are registered in `paper_experiments/current_conditioned_results.json`.

For frozen retrieval restoration, verification, and ordinary-Parquet sharing, use
the same [current Starling entrypoints](../../paper_experiments/CURRENT_STARLING_RETRIEVAL.md)
as BBB, Bioavailability, and Skin. The common manifest includes Ames; its canonical
snapshot already contains the reviewed family assignment. Source-policy changes
still use the adapter below and require updating the adopted frozen snapshot.

## Inputs and reproducibility

The 2026-09-13 trace follow-up is recorded in
`data/starling_data/ames/trace_review_20260913/verification.json`. Two payload-pinned
decisions extend the existing placement ledger: `ames_v2:425939` moves L2→L4
(SOS/umu response), and `ames_base:186648` moves L2→L3 (plant chromosome damage).
The accepted cinnoline record `ames_base:80868` also receives a qualifying-context
correction: quinoline itself is an exception to the second-ring-nitrogen structural
pattern. Original raw/support text is preserved. The original PMID7022455 paper
confirms its quinoline TA98/+S9 positive result; later conflicting findings do not
justify replacing that observed outcome or the current gold label.

The full rebuild preserves all 3,333 study-unit labels and all six split rowsets.
Scaffold train/valid/test input JSONL bytes are unchanged from immediately before
this follow-up; random split membership and labels are unchanged but row order
changes. Cinnoline vote metadata/fingerprint and both retrieval indices change.
Do not reuse predictions by directory name: validate selected model inputs and
query indices, or replay. Published Stage-03 and shared Parquet include this repair.

The registered `ames_trace_source_replay_20260913` suite completes that audit and
targeted v5 progressive replay for scaffold valid/test: 23 affected queries,
97 downstream levels, 2,643 verified reused prefix levels and zero final failures.
Final Macro-F1 is unchanged (valid 0.7152; test 0.7545), with two beneficial and
two harmful final changes in valid. Use the registry's report and reuse receipts;
this correction is not an independent replicate or a replay of historical full-flat.

`data/starling_data/ames/raw_v1/manifest.json` pins four unchanged Parquet files
from commit `03e4c7c694b45bcfdf7776ac1045bc3e69936f0a` of
`DragonDescentZerotsu/TxAgent`, originally on `codex/ames-compressed-parquets`.
Missing or hash-mismatched raw files are restored from that pinned commit into
a temporary file; only a successful SHA-256 check publishes the replacement.
An interrupted restore can be retried without manually deleting partial files.

The files contain 1,510,447 source records in total. A record's stable source ID
is `<source-file-stem>:<zero-based-row>`; `extraction_id` alone is not unique.

Frozen PubChem name responses live under
`data/starling_data/ames/identity_review_v1/`. Rebuilding is offline and uses
these responses, rather than silently refreshing a time-dependent identity
lookup. PubChem supplies identity evidence, never Ames outcome labels.
Benchmark labels use an identity-verified source subset. Candidate names still
failing after at least four recorded requests are excluded from both votes and
retrieval and listed in `canonical_v1/pending_identity.jsonl`. A service failure
is distinct from a completed name lookup returning no match. The frozen response
file retains every attempt, including successful case-only name retries.
The indirect v1–v3 files lack a molecule-name field: their retained structures
are source-supplied and normalized, without independent name-based identity
verification. The record audit distinguishes that status from a verified match.

For routine refreshes of the current initialized dataset, from the repository
root on node002:

```sh
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.ames.build_dataset --phase refresh-retrieval --workers 128
```

For an initial build or an intentional gold/benchmark reconstruction use
`--phase all`. The same entrypoint accepts `source`, `benchmark`, and `retrieval` phases.
For a retrieval-policy revision use `--phase refresh-retrieval`: each stage is
reused only when its inputs, policy/review code, runtime dependencies and output
hashes match. Source reconstruction is forced by `source` and `all`. A refresh
requires byte-identical votes and preserves every benchmark file without running
the split allocator. An unchanged refresh also preserves the dataset manifest
bytes, so it does not invalidate an existing validation receipt.

On node002 the default is 128 workers, matching its 128 physical cores / 256
logical CPUs. Source processing uses that budget; scaffold and random index
builds run concurrently with 64 workers each. Aggregation partitions complete
assay-molecule groups after one stable sort, avoiding per-group DataFrames.
Standardization and JSONL serialization are also parallel (serialization uses
at most 16 workers per index). Ordered merging preserves first-record dedup,
card selection and exact evidence JSONL bytes. Set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` to avoid nested native thread pools.

Verified content-addressed inputs and temporary outputs use node-local NVMe at
`/local/tmp/txagent-build-cache` (`TXAGENT_BUILD_CACHE` overrides this location).
Canonical artifacts remain at their published repository paths. Identical
outputs skip publication; changed outputs publish atomically. The cache is
regenerable and is not a source of provenance. Digest reuse checks full file
metadata and host boot identity; the independent validator initially hashes the
published files independently. Downstream phases verify their actual inputs;
source-only rebuilding invalidates prior downstream references.

The v6 review's `retrieval_build_performance.json` records the measured aggregation
speedup, full build and cache-hit timings, and byte-equivalence checks separately.
A changed build and independent full audit still require processing the data.

## Source and label policy

`source_contract.py` is a deterministic source-annotation policy. Passing it is
not a claim of exhaustive human review. A stratified 48-record agent review of
supplied source text is retained under `source_review_v1/`; it is neither a
full-paper audit nor an estimated error rate for the complete source.

**Gold recall remains incomplete.** The deterministic v1 gate examines native
`ames_base` bacterial-reverse-mutation rows. The builder now also consumes
`source_review_v4/applied_decisions.jsonl`: individually authored decisions for
any source/family, pinned to source ordinal, raw payload hash, PMID and parent.
Accepted reviews still require frozen PubChem name-parent verification and
study-unit collapse; reviewed eligibility alone never grants L1 membership.
Withheld direct claims stay near-direct, and confirmed subject/structure errors
are quarantined across the reviewed paper/parent pair. Original extractions are
immutable. A factual correction to a card must be explicitly documented in the
review decision; gold votes and derived review reasons are not prompt fields.

The preceding `source_review_v3/` nonvoter audit covered 144 records and changed
no active data. The subsequent `source_review_v4/` audit individually reads 96
additional actual voters and 48 additional L2 records, revisits six prior
candidates, examines selected primary abstracts and four full-text methods/result
sections, and applies bounded corrections. Its `audit_annotations.jsonl` and
`README.md` distinguish retained records, unresolved claims, verified errors,
new candidates, duplicate study descriptions and actual post-collapse votes.
This is **Codex agent review, not human-expert annotation, an exhaustive full-paper
audit or a population error-rate estimate**. An unresolved record is not evidence
that the experiment is invalid. Most source records remain unreviewed.

L1 purity and L1 recall are separate checks: actual-voter membership prevents
nonvoters being displayed at L1, but does not establish that every suitable
experiment was considered for gold. A gold revision must review candidates
across all four sources, normalize endpoint-specific outcomes and conditions,
and distinguish measured positive controls from control-name mentions,
weak potency from equivocal results, mechanism uncertainty from outcome
uncertainty, and concordant methods from conflicting treatment arms. Source
`needs_more_context`, a prediction elsewhere in the passage, a nonempty qualifier,
or a nonstandard field spelling is a review signal rather than proof of an
invalid experiment. Citation recovery must use the original study identity;
the paepalantine review record points to original PMID 9261922, not an additional
experiment under review PMID 15720259. Merge repeated study descriptions before
voting. Only integrated study-unit representatives enter L1; unresolved direct
content stays at L2 and never moves to L3-L5. Any gold revision must rebuild and
validate dependent splits and heldout-filtered indices before model use.

Gold requires a defined chemical, a primary in-vitro bacterial reverse-mutation
experiment, recoverable endpoint-specific strain/activation conditions and an
unambiguous measured outcome. The deterministic baseline rejects weak enums,
unresolved qualifiers and unusual protocols pending review. Individually reviewed
records can resolve those gates: measured positive controls are eligible; a
low-potency positive is not automatically equivocal; concordant assay methods,
mechanism speculation or an unfamiliar but explicit Salmonella genotype do not
invalidate a measured result. Predictions do not supply gold. Unknown treatment
arms, inferred outcomes, unverified study attribution and unresolved identity or
material remain nonvoters. A PubMed `Review` tag alone is not a rejection rule:
some such articles report original experiments.

Gold additionally requires one unambiguous PubChem name resolution whose parent
identity exactly matches the source structure. Failed requests remain unresolved;
they must not be interpreted as absent names. Ambiguous names, stereochemical
mismatches, known misassignments, unsupported inorganic/metal materials and
multiple organic components do not vote. Structures are never automatically
repaired. Name resolution does not establish the validity of a cited experiment or that
the extracted name denotes the original paper's test article. The v4 audit found
OA/okadaic-acid, COL/collagenase and NIDA abbreviation-resolution errors despite
name-to-structure matches; these wrong subjects cannot vote.

The condition signature is the exact normalized **strain panel × activation**.
`both_reported` remains one pooled source-reported unit: it is never duplicated
into positive or negative votes for each activation regime. For a panel, a
positive result means a reported positive response within that panel, not that
every strain is positive. A limited-panel negative does not imply a global Ames
negative. Missing or unresolved panels/activation are nonvoters and are never
silently converted into `no_reported_external_condition`.

Exact duplicate records retain source pointers in the audit. Repeated descriptions
within the same **PMID × parent × exact condition** contribute at most one vote.
Conflicting calls within that unit contribute none. Across accepted study units,
the shared builder requires at least 60% agreement for external-condition rows
and 70% for rows without a reported external condition, and rejects ties. Ames
currently contains only external-condition rows. Support counts
therefore represent distinct accepted PMID units, not repeated paragraphs.

Condition groups need at least three accepted parents and three nonempty scaffolds,
and must occur in train, valid and test. Complete scaffold groups are assigned by
the shared lexicographic quality allocator. First, the fresh builder finds the
smallest feasible equal valid/test size at or above the nominal 10% target while
preserving every eligible condition. It records any expansion and freezes that
size before optimizing record support and label balance; explicit requested
sizes remain strict. Random splits use the exact same
rows/labels, assign whole parents, and enforce exact rounded 80/10/10 sizes and
three-way condition coverage. Both schemes have zero parent overlap; only the
scaffold scheme requires zero nonempty-scaffold overlap.

## Evidence families and retrieval acceptance

The baseline remains `ames_bacterial_reverse_mutation.v1`, supplemented by the
payload-pinned semantic review ledger. Retrieval uses
`ames_inclusive_retrieval.v3`; gold rejection is not a retrieval rejection.

| Level | Family | Membership |
|---|---|---|
| L1 | `direct_ames` | Actual study-unit representative voters only |
| L2 | `nonvoter_ames_outcome` | Nonvoter bacterial outcomes/predictions, unspecified mutagenicity and mixed passages containing direct-related content |
| L3 | `other_genetic_damage` | Other gene mutation, chromosome damage, micronucleus and cytogenetic outcomes |
| L4 | `dna_damage_response` | DNA damage, repair, damage responses and their modifying effects |
| L5 | `genotoxicity_mechanisms` | Bioactivation, detoxification, DNA interaction, redox/antioxidant defense, replication/topoisomerase and chromosome-segregation mechanisms |

Predictions of the direct outcome are retained at L2. Predictions of an indirect
endpoint remain at that endpoint's family. `needs_more_context`, weak/mixed/
equivocal results, secondary references and modifier/protectant roles do not
alone disqualify retrieval. Complete source text and explicit roles, uncertainty,
conditions and target/endpoint qualifiers remain on the cards; their admission
never supplies a label or proves transferability to the query's own outcome.
Measured lesions, probes and co-treatment effects must not be misread as the
record molecule causing its own mutagenicity.

The full source-supplied semantic record is checked before indirect assignment.
Any Ames/bacterial reverse-mutation context (including predicted, mixed or
incidental outcomes, WP2, TA suffix strains and source-native reverse-mutation
enums) is conservatively confined to L2 unless it is an actual L1 voter. Generic
unspecified mutagenicity is also near-direct. This may route a background mention
to L2; it avoids carrying a target answer into L3-L5 without truncating its passage.
Bacterial DNA-damage reporters without reverse-mutation context remain L4.

The v3 placement revision removes source-run-based indirect assignment. Concrete
endpoint classes and assay readouts determine L3–L5 across all four sources;
damage-specific classes prevent a comet assay described as "clastogenicity" or
PAR measured inside micronuclei from being mistaken for chromosome scoring.
Mixed passages and false bacterial hits additionally use individually authored
`source_review_v6/placement_decisions.jsonl`, pinned to the complete raw payload
and unchanged canonical card surface. The broad independent direct-content guard
remains active for every unreviewed record; a group edit alone cannot bypass it.
Reviewed author names, Ames dwarf mice, explicit absence of testing, nonbacterial
reversion and repair-only reporters can therefore be placed at their actual
endpoint. A negative Ames result or a real accompanying bacterial outcome stays
at L2. These decisions cannot create votes or override identity exclusions.

`source_review_v5/` remains the historical 200-record proposal audit.
`source_review_v6/` adds 194 individually read records (140 unread candidates
from that audit's pool and 54 transition checks), with 46 new relocation
decisions. Together the two placement audits cover 394 distinct records and
95 relocation decisions. Four v6 and four v5 indirect-family boundary cases
retain their prior placement pending clearer endpoint interpretation; two v5
gold-scope/condition boundaries and eight gold-recall candidates are also not
resolved by a placement-only revision. This is a risk-enriched source-content
audit, not external source verification or an estimate of the whole library's
error rate. See the v6 summary for applied counts, preserved hashes and validation.

No standalone source enum allowlist or model/uncertainty exclusion limits the
indirect pool. Substantive endpoint-family descriptors are required; blank or
unresolvable unrelated endpoints, missing support/structures, invalid parents,
unsupported material/multicomponent identities, known identity mismatches and
unresolved name identities remain audit-only. Prediction status is not an
identity repair. Simple, single-component non-carbon molecules with at least two
heavy atoms from N/O/F/P/S/Cl/Br/I (for example H2O2) remain retrievable after valid
parent normalization; their original gold exclusion remains unchanged. Metals,
materials and unresolved multicomponent identities still require separate review.
L3 and L5 were renamed to reflect the broader endpoint coverage.

The current source contains **1,299,584 retrievable records (86.04% of the raw
source)**: 3,333 / 138,571 / 236,309 / 496,880 / 424,491 in L1–L5 before heldout
removal. The v3 retrieval revision changes 86,093 existing record placements,
retains every previously retrievable record with unchanged evidence fingerprints
and parents, and recovers 111 previously unresolved endpoint records. These
rule-based movements are separate from the 95 individually reviewed relocations.
The earlier v4 gold/source review removed 36 wrong-subject/structure records; unresolved direct claims retain L2 membership. It adds 12 study votes,
withdraws 25 (7 confirmed identity-attribution errors and 18 pending
source/condition/material resolution), and supplements two existing votes with
cross-source supporting descriptions. A third reviewed duplicate was already
represented and adds no vote. All current representative voters still originate
in `ames_base`; this is the observed collapse result, not a restriction on
reviewed source eligibility.

The payload-pinned exclusion in `identity_repair_3acaba/receipt.json` removes
`ames_base:76701` (PMID 14981162): the paper's 3-Ac-ABA is
3-acetylaminobenzanthrone, whereas the source supplied an anthranilic-acid
derivative structure. The raw extraction remains immutable; no replacement
structure is inferred. The same applied-review ledger drives offline rebuilds.
This removes one L3 nonvoter, preserving all gold votes and benchmark files;
the experiment registry records selected-surface audits and targeted replay.

The payload-pinned `correct_structure` reviews in
`identity_repair_nitroso/receipt.json` correct seven nonvoter records whose text
identifies N-nitroso-N-methylcyclohexylamine but whose source structure lacks the
N–N bond. The methyl-d3 record retains its isotope labels. Raw structures and
evidence text remain unchanged; canonical identities, similarities and derived
indices are rebuilt. This review action cannot grant votes or restore excluded
records. Model outputs require a fresh selected-input audit and affected-query
replay; a structure correction alone does not establish prediction equivalence.

On 2026-09-08, the user explicitly exchanged the scaffold valid/test names:
current test is the former 274-row valid cohort, and current valid is the former
274-row test cohort. Train, random, labels and the heldout identity union are
unchanged. The fresh Ames builder applies `swap_evaluation_splits=True` after
allocation. `data/conditioned_benchmark/Ames/provenance/scaffold_valid_test_swap_20260908.json`
pins the before/after hashes and index metadata rebinding. The renamed test
cohort was already inspected during validation; it is not an untouched test set.
Historical run directory names retain their original meaning; current result
roots and the exact cohort mapping are in `current_conditioned_results.json`.

There are 3,642 candidates after identity/structure/deduplication and before study
collapse, yielding 3,333 study votes across 1,423 parents. Across studies,
14 exact ties are rejected; 2,646 parent-condition rows survive agreement.
The condition-group support gate removes 172 of those rows, leaving **2,474
molecule-condition rows, 1,383 parents and 73 conditions**. Of these, 2,230 rows
have one accepted study vote. Compared with the preceding cohort, three rows
are added, sixteen removed, and no retained row changes label. Scaffold sizes
are 1,926 / 274 / 274; random sizes are 1,980 / 247 / 247. These counts describe
the checked source subset, not complete Ames literature coverage.

The raw-source counts remain 1,510,447 rows / 37,410 normalized parents; the
native `ames_base` bacterial family contains 91,511 rows / 9,643 parents.
Its 27.4% share of base rows is a source family proportion, not a retention rate.
The preceding deterministic funnel is preserved in
`source_review_v4/previous_vote_funnel.json`; `vote_funnel.json` now describes the
applied review, study collapse and benchmark gates. Pending identity requests
remain 33 candidate records / 24 names.

In the historical deterministic baseline, the largest semantic queues were activation/experimental-arm review (21,691
records), previous-study citation/provenance review (17,691), and unresolved or
nonstandard strain panels (11,889). These are first-failure rule counts, not
verified invalid experiments; parent counts overlap across reasons. Broad text
guards can defer otherwise valid experiments when a passage contains comparisons
or background citations. Expanding gold requires resolving their actual subject,
condition, outcome and identity. These nonvoters can remain retrievable in L2.

Parent/condition rejection after voting does not revoke source voter membership.
Families are assigned at record grain, even when one physical assay contributes
to several levels. L1-L5 describe evidence organization, not a universal ordering
of reliability or a guarantee of better predictions at higher levels.

Before either index is built, all L1/L2 records from valid+test parents are
removed, across conditions. The source builder additionally requires zero
bacterial-outcome-scoped records in L3-L5. The index defaults are
`scaffold_disjoint` for scaffold and `parent_disjoint` for random. Vote counts,
classification reasons and identity audit fields are not LLM prompt fields.

The former v1 WP2 index repair is historical provenance. The v2 revision rebuilds
both complete indices and fixes record-family containment at source level.
`source_review_v2/` records all 1,510,447 row transitions, 27 agent source-text
reviews, the bounded mouse-lymphoma endpoint-alias repair after the full source
rebuild, and byte-identical gold/split receipts. This is not a full-paper review
or an estimated source error rate. In that historical v2 retrieval revision, the 210,937 audit-only records were
mostly unsupported materials/identities (162,144 material rows), with 562
unresolved endpoint records and 17 exact duplicates; they are not all
scientifically unrelated. Run the full card-lineage, direct-containment, heldout and
cumulative/progressive retrieval audit with:

```sh
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m \
  tools.chembl_tool.tasks.ames.validate_retrieval
```

## Outputs

| Split scheme | Train | Valid | Test |
|---|---:|---:|---:|
| Scaffold | 1,926 | 274 | 274 |
| Random | 1,980 | 247 | 247 |

Scaffold's nominal heldout size is 247 per split. Exact coverage with complete
scaffolds requires at least 274 per split; this minimum is solved from the data,
not hardcoded. Both schemes retain all 73 conditions in every partition.

The shared `fresh_conditioned_benchmark.v2` builder uses external 60% / unreported
70% agreement. Current unique parent counts are 1,102 / 137 / 144 for scaffold
and 1,217 / 80 / 86 for random. `source_review_v4/` records the applied semantic
revision; `vote_funnel.json` describes the current source-to-benchmark gates.
The earlier `agreement_revision.json` and source-review v2 preservation receipts
are historical inputs, not assertions that current votes or splits are unchanged.
The retrieval validator independently replays current vote agreement and
condition-group gates, verifies every reviewed withdrawal/quarantine and checks
actual L1 voter membership and source-card lineage.

- `data/starling_data/ames/canonical_v1/`: canonical records, complete source-row
  audit, final study-unit source votes, study-collapse audit, pending identity
  ledger and hash manifest.
- `data/conditioned_benchmark/Ames/{scaffold,random}/`: standard minimal and
  detailed split files, heldout unions, condition distributions and summaries.
- `data/conditioned_benchmark/Ames/provenance/scaffold_build_summary.json`:
  scientific group gates, voting policy and scaffold optimization receipt.
- `outputs/paper/starling_conditioned_assay_family_curve_v1/family_catalogs/ames/`
  and `indices/ames/{scaffold,random}/`: catalog and split-specific indices.
- `data/starling_data/ames/dataset_manifest.json`: source, benchmark, catalog and
  index hashes, with explicit review method and model-evaluation status.
- `retrieval_validation.json` and `dataset_validation.json` in the same Ames
  source directory: indexed-card provenance, heldout/family/neighbor checks,
  representative cumulative/progressive retrieval, and data-preservation receipts.

Ames is registered in the shared progressive runner and its matched full-flat
control. `--fresh-query-priors` prepares resumable single/None outputs using the
scientific prompt in `query_prior.py`, the shared tool prefetch, JSON validation
and bounded query pool. The single branch receives structure/property inputs;
the None final branch additionally receives the exact reported condition.
Neither prompt receives the query label.

The Ames reasoning contract explicitly allows sufficiently strong, transferable
indirect evidence to support, contradict or overturn a prediction without new
direct Ames data. The model must explain the mechanistic link to the query's
structure, strain panel and activation condition, its limitations, and distinguish
the prediction from a measured Ames outcome. This clarification is shared by
progressive, matched full-flat and the None final task instructions. Existing
scaffold-valid scores used the earlier wording; evaluate the revised prompt in
a separate output root rather than resuming or relabeling the frozen run.

For the current scaffold-valid 4/2 run, use the exact commands and runtime status
in `replicate_suites.ames_scaffold_valid_4_2` of
`tools/chembl_tool/paper_experiments/current_conditioned_results.json`.
Its launch receipt freezes the 256-request cap, model and output roots.
The matched full-flat run copies the progressive prepared evidence/tools and
single/None outputs, then judges every level independently. Data-builder
`model_evaluation_performed=false` describes dataset construction; evaluation
status and scores belong to the experiment registry.

The five matched baselines use train-only reference/selection. Ames valid v2
uses `same_condition_then_null_then_all` for both condition-first KNN methods:
same-condition neighbors take priority, then no-reported-condition neighbors,
then unrestricted train neighbors fill the remaining k=3 slots without duplicate
parents. Ames has no no-reported-condition rows. All five methods now cover
274/274 rows; each KNN method fills 30 slots across 22 queries, preserving the
previous 252 predictions and their neighbors exactly. The v1 partial-coverage
results remain historical. The v2 receipt records fallback provenance and keeps
the original train-only head and unrestricted-KNN artifacts as dependencies.

## Scientific reference

The endpoint and separate activation regimes follow
[OECD Test Guideline 471](https://www.oecd.org/en/publications/test-no-471-bacterial-reverse-mutation-test_9789264071247-en.html).
The narrower source acceptance, study-unit aggregation, condition grouping and
retrieval-family policies above are this project's explicit dataset decisions,
not assertions that every source experiment satisfies the complete OECD guideline.
