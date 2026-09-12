# Carcinogens: reviewed Starling-only, five organism groups

The released benchmark contains 4,692 molecule-condition rows: 3,578 positive
and 1,114 negative. Scaffold and random both split 3,754/469/469, with all five
conditions in each subset. No accepted condition needs a rare-group exclusion.

| Condition | Positive | Negative | Total |
|---|---:|---:|---:|
| rodent | 2,163 | 697 | 2,860 |
| human | 1,233 | 337 | 1,570 |
| dog | 94 | 40 | 134 |
| monkey | 41 | 28 | 69 |
| rabbit | 47 | 12 | 59 |

There are 260 parent-condition pairs below the consensus requirement and 107
with only internally conflicting publication units. These 367 pending pairs
are separate from records held for unresolved identity or organism.

Condition coverage does not guarantee both labels within each condition/subset.
The shared scaffold allocator puts 1 rabbit row in valid and 4 in test, all positive;
all 12 rabbit negatives are in train. Random has one rabbit negative in valid and
none in test. Per-condition negative recall is undefined for those zero-negative
subsets; the overall heldout sets each contain 111 negative rows. Full counts are
in `independent_gold_validation.json`.

Build entry: `python -m tools.chembl_tool.tasks.carcinogens.build_reviewed_starling`.
Use `--fetch-identities` to resolve missing official PubChem parent synonyms;
the default reuses frozen evidence and caches. No new LLM direction review is
performed. The release consumes all 354,386 original base rows and the complete
hash-bound `targeted_review_v2/carcinogens_source_labels.parquet` ledger.

The only query conditions are `species=rodent`, `species=human`, `species=dog`,
`species=monkey`, and `species=rabbit`. Explicit rodent species and synonyms are
pooled; absent species are never silently asserted to be rodents. Sex, strain,
route, dose, duration and other model details remain in raw evidence, without
creating additional query axes. This is pooled source consensus, not a claim
that every species member or exposure has the same outcome. In-vitro-only,
unresolved and cross-group claims retain their record-level directions and
provenance but do not independently create an organism-specific gold label.

Identity requires exact name/parent or official synonym agreement, preserving
known specimen holds. The source resolver records a specific correction for
`Urethane (ethyl carbamate)`: both names and the supplied single-component
structure match PubChem CID 5641; the composite name also returns an unrelated
mixture entry. Five misnamed records attached to PMID 33636299 are held: four Methanal
records and one methyl-isothiocyanate record. That article studies methyl acrylate. These decisions are bound
to exact UIDs, raw payload hashes and, for clearances, unchanged structures.

Each reporting-publication–parent–organism unit contributes at most one vote;
exact repeated long passages across publications are also deduplicated.
Internally conflicting units do not vote. At least 60% agreement and a strict
majority are required across retained units. We reuse the shared fresh scaffold
and parent-grouped random allocators, with all five conditions in all splits.
A publication is not necessarily an independent primary experiment; this release
does not claim a new full-paper or human-expert audit.

`summary.json` reports the full funnel and per-condition positive/negative counts.
`record_conditions.parquet`, `identity_ledger.parquet` and
`record_dispositions.parquet` account for every base UID. Gold and supporting
votes are in `gold_labels.jsonl` and `gold_label_provenance.jsonl`; undecided
parent-condition pairs and publication conflicts have separate ledgers.
No TDC label contributes to these gold labels or their splits.

L1 contains actual vote representatives; nonrepresentative source support remains
L2. Only heldout L1 is prefiltered, and all levels apply query-time scaffold/parent
disjoint rules. Exact raw copies are removed from working records with a duplicate
ledger, preserving original acquisition.

The final source incorporates R18 corrections and is frozen in
`data/starling_data/carcinogens/retrieval_final/`. Its `publication.json` and the
shared `current_starling_retrieval.json` identify the active source and both indices.
The initial `retrieval/canonical_publication.json` and
`retrieval/publication_completion.json` remain historical publication receipts;
local `retrieval/rebuild.sh` and `retrieval/publish.py` have been removed.

Use the shared restore/build/verify commands documented in
`tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md`. The latest completed
progressive diagnostic is R18; matched baselines and other completed runs are
registered in `current_conditioned_results.json` with their actual input lineage.
Data restoration does not launch an experiment or authorize reusing predictions
under changed selected evidence.
