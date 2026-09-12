# DILI / Carcinogens identity repair (2026-09-08)

The active canonical cohorts are `data/conditioned_benchmark/{DILI,Carcinogens}`.
Publication is byte-exact and recorded in
`data/conditioned_benchmark/tautomer_identity_repair_publication.json`.
The old augmented cohort is frozen at
`data/.build/conditioned_benchmark_sources/<Task>/pre_tautomer_repair_snapshot/`;
the earlier source-only snapshots and source gold remain separate provenance.
Raw records and permanent source UIDs were not rewritten.

## Identity and split contract

`new_task_tautomer_identity.v2` applies only to these two tasks. Gold uses RDKit
canonical tautomer enumeration with stable sp3 stereochemistry retained; stereo
on bonds participating in tautomer transformation may be removed and is logged.
Enumeration is capped at 10,000 tautomers / transformations. Non-complete results
retain the previously verified source parent, without selecting a truncated
canonical answer. Sixteen retained source identity forms use this explicit policy;
the two additional reviewed specimen mismatches described below were withdrawn.

Split and retrieval grouping does not enumerate tautomers. After fragment-parent
normalization and stereochemistry removal it combines RDKit `ElementGraph`,
`CalcMolFormula(separateIsotopes=True)`, and net charge. Thus equivalent H/bond-order
representations and specified/unspecified stereochemistry stay together, while
hydrogenation states with different formulas remain distinct. Isotope species and
counts are retained; this is a conservative leakage boundary, not a gold merge.

Scaffold grouping uses a terminal-pruned Murcko ring/linker core and `ElementGraph`.
It preserves elements and connectivity but ignores bond order and stereochemistry.
Benzene and cyclohexane consequently share a scaffold group. This is deliberately
more conservative than the original Bemis–Murcko contract of the other five tasks;
it is not described as the same algorithm. Original scaffold SMILES remain in
identity lineage; allocator scaffold fields use the explicit topology representative.

Neither RDKit `HetAtomTautomer` version suffices as the sole grouping key:
v2 distinguishes the two observed dacarbazine representations, and v1 distinguishes
acetone from its enol. Both counterexamples, stable R/S and E/Z retention, formula
separation, non-complete enumeration behavior, and within-study conflict handling
are regression tested.

## Votes and reviewed withdrawals

Source voting eligibility starts only from the existing accepted `gold_v2` vote
ledger. No old rejected claim is promoted. Equivalent identities are regrouped by
study ID, gold parent, and condition; duplicate reports cannot add votes, and an
internal-study label conflict removes all votes for that unit. The shared builder
recomputes >=70% agreement for no condition, >=60% for an external condition, and
rejects ties. Actual L1 UID membership is the union of previously accepted vote UIDs
supporting surviving study votes; historical duplicate descriptions do not gain L1.

Two DILI source UIDs were individually reviewed and removed from gold and molecular
retrieval, with payload hashes and full original records preserved:

- `sr_7ef199545505459b8e256010c00d5ae3`: Amanita phalloides mushroom exposure does not
  identify a specific isolated amatoxin congener.
- `sr_9b1493645aa14fc7a77af3bfd86bb582`: clinical teicoplanin formulation exposure does
  not establish the supplied single-congener structure.

These are exact-record holds, not whole-name or whole-structure exclusions.
DILI retains 4,727 study votes (4,494 positive / 233 negative); Carcinogens retains
12,571 (12,165 positive / 406 negative). This repair found no newly duplicated or
conflicting accepted source-study units. TDC internal opposite-label parents remain
withheld. TDC-null and Starling-conditioned labels remain separate units with a
cross-source disagreement ledger; no external label becomes an assay vote.

## Active data

| Task | Rows / gold parents | Scaffold train / valid / test | Random train / valid / test |
|---|---:|---:|---:|
| DILI | 1,034 / 840 | 828 / 103 / 103 | 828 / 103 / 103 |
| Carcinogens | 1,464 / 715 | 840 / 312 / 312 | 1,164 / 150 / 150 |

The stronger scaffold constraints require Carcinogens equal heldout sets of 312
rows. This is the minimum feasible size under the recorded constraints; eligible
records were not deleted to force smaller splits. Valid+test Starling singleton
counts are minimized first, their imbalance second, and labels afterward; TDC has
zero assay-support count and does not enter the vote-quality objective.

`data/.build/new_task_tautomer_repair/validation.json` independently recomputes
agreement, study uniqueness, unchanged source payloads, exact actual-voter membership,
TDC label provenance, optimizer objective values, full leakage-group disjointness,
and scaffold-topology disjointness. It passed for both split schemes and tasks.
The four index heldout inputs changed and therefore require fresh identity-aware
indices; the publication receipt does not authorize reuse of historical indices.

## Rebuild and provenance

- New gold and L1 ledger: `data/.build/new_task_tautomer_repair/<Task>/gold/`.
- Repaired Starling-only source cohort: `.../source_only/<Task>/scaffold/`.
- Reviewed augmented build: `.../augmented/<Task>/{scaffold,random}/`.
- Original global receipts and dataset manifests: previous snapshot `publication_provenance/`.

Use `build_new_task_tautomer_repair` for source staging and
`build_tdc_augmented_benchmark --tautomer-aware --source-root
 data/.build/new_task_tautomer_repair/source_only --output-root <fresh-root>` for a
new augmented trial. Do not use the older publisher/TDC default source snapshot
for a repaired-source rebuild. Current staged inputs are frozen publication
provenance; choose a fresh output root for subsequent trials. The shared global
identity normalizer and the other five task datasets are unchanged.
