# Canonical prediction prompts and TDC inputs

Prediction and query-prior assets now have one owner:

```text
predict/prompts/
  __init__.py
  tasks/{bbb_martins,bioavailability_ma,skin_reaction,ames,dili,carcinogens}/profiles.yaml
  flat/final_full_flat/{system.jinja,user.jinja,levels/,provenance.json}
  query_prior/{profile.yaml,system.jinja,native_system.jinja,user.jinja}
  variants/<named experiment>/
```

Each task YAML holds the canonical flat contract and named native profiles.
Python handles rendering and execution; Jinja/YAML holds prompt prose and
schemas. Specialized overrides remain explicit. Historical bundles are read
directly from `archive/branches/prompts/`; the old prompt symlinks and three
duplicated Python profile modules were removed. Retrieval-scoring, semantic
ranking and evidence-construction prompts retain their distinct owners.

Exact message comparison passed 36 native-prior cases across BBB/Oral/Skin and
all six generic query-prior cases. Default profile choices and scientific prose
were preserved. Current flat task contracts were compared with their original
YAML before extraction; the TDC instruction key is now unversioned.

## One TDC configuration

`data/gold_labels/TDC/manifest.json` binds all six task directories, split and
ranking policies, level limits, published label hashes and cache inputs. Labels
live directly under `TDC/<Task>/`. There is no current TDC edition argument or
`CURRENT` pointer.

| Task | Selected cohort | Train | Valid | Test |
| --- | --- | ---: | ---: | ---: |
| BBB | Original source splits | 1421 | 203 | 406 |
| Oral | Scaffold conditioned | 448 | 64 | 128 |
| Skin | Scaffold conditioned | 282 | 40 | 82 |
| Ames | Original source splits | 5094 | 727 | 1457 |
| DILI | Original source splits | 332 | 47 | 96 |
| Carcinogens | Original source splits | 196 | 28 | 56 |

These are the former Oral/Skin v1 and remaining four tasks' v2 cohorts. Rows,
labels, ordering and UIDs are unchanged. Previous editions, 19 retired TDC cache
YAMLs, original headers and historical construction sources are preserved under
`data/legacy/tdc/`. Its explicit relocation table supports historical reads
without aliases. Recorded historical cohort identities stay historical.
Completed prediction and FINAL artifacts were not migrated or rewritten.

## Optimizer integration

Frozen retrieval indexes moved to
`data/caches/assay_reranking/active/flat_v5/tdc/<task>/{l1,l2plus}/`.
All 72 relocated binary payloads retain their original inode, size and mtime.
Existing verified hashes were carried forward for 78 payloads after stat checks;
changed small headers were hashed. New full-molecule card payloads were verified
at publication. The six artifact bundles and receipts include both retrieval
and optimizer cache dependencies.

The manifest distinguishes `retrieval.L1` from `selection.direct_cache`:

- Frozen `l1/` contains parent-grouped contexts and their original scores.
- New `l1_cards/` contains full-molecule cards for the new optimizer, with
  authoritative training labels and ordered physical members. Parents are used
  for exclusion only. These caches were built offline from the same cohorts.
- `scores/` indexes existing frozen score values and keys; no model was rerun.
  Parent-grouped L1 scores cannot be attached to individual canonical cards.
  New card assay scores remain uncomputed and missing.

`optimization.selection.search` defaults to these configured inputs for
`--benchmark tdc`. Direct and indirect searches export physical UID manifests;
`compose` joins matching panels. Prediction accepts them through `--uids-file`,
checks the configured cache identities and hydrates the selected physical rows.
Concrete runs use `experiments/selection/tdc.sh`.

## Validation and remaining data issue

The focused prompt, benchmark, artifact, retrieval and optimizer suite passes
226 tests. Another 12 checks cover saved metadata, frozen scoring prompts,
tokenization and CPU scoring (238 passed overall); four GPU-only head checks
are skipped on this login node. Compilation and focused whitespace checks pass.

One validation query per task passed direct optimization, ordered physical UID
export, flat preparation and final prompt rendering. Composed direct/indirect
panels passed for BBB, Oral, Skin, Ames and DILI.

The first Carcinogens validation query,
`TDC_CARCINOGENS_125BE2B05F81BD8C8039`, exposes a pre-existing frozen-cache issue:
L3–L6 include 50 records for the same connectivity with unspecified stereochemistry.
For example, `sr_0017ab20d83f46cca53da35edfacc694` names
`CNCCCC12CCC(c3ccccc31)c1ccccc12`, while the query is
`CNCCC[C@]12CC[C@H](c3ccccc31)c1ccccc12`. The shared parent-disjoint policy rejects
these candidates as `same_connectivity_variant`, so indirect optimization stops
before inference. That guard and the frozen cache remain intact. This migration
does not claim that every frozen candidate universe satisfies the newer guard.

No prediction, API inference, assay rescoring or scientific evaluation batch ran.
