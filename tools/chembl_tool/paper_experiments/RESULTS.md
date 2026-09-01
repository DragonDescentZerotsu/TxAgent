# Current paper results and artifact status

Updated: 2026-09-01.

This file is the human-readable companion to
`current_conditioned_results.json`. It records only the retained paper result
families and their freshness. Historical experiment narratives no longer live
here; a historical score is shown only when it is still the last complete
reference for a retained paper comparison.

## Evaluation data

All active experiments use the Conditioned Benchmark:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

| Task | Scaffold train / valid / test | Random train / valid / test |
|---|---:|---:|
| BBB_Martins | 3,053 / 397 / 393 | 3,075 / 384 / 384 |
| Bioavailability_Ma | 1,956 / 262 / 269 | 1,989 / 249 / 249 |
| ClinTox | 1,144 / 142 / 142 | 1,142 / 143 / 143 |
| Skin_Reaction | 1,997 / 246 / 248 | 1,993 / 249 / 249 |

The construction and leakage contract is
`tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md`. Scaffold and
random contain the same molecule-condition labels. Random is parent-grouped,
not molecule-row independent.

## Freshness rules

A result is current only when all relevant identities match:

- benchmark input and split;
- held-out-filtered retrieval index and source-family catalog;
- evidence source and organization;
- model and prompt profile;
- visibility and neighbor-identity policies.

Matching a directory name, molecule count, or valid-input hash is not enough.
`current_conditioned_results.json` is the machine-readable authority.

## Retained result families

| Paper result family | Current state |
|---|---|
| ChEMBL versus Starling retrieval | Last complete identity-blind reference exists on historical record-supported-v2; a complete current-conditioned matrix is still required |
| One-shot full-flat cumulative levels | Last complete reference exists; current source-purity rerun is required |
| Append-only progressive levels | BBB and Skin scaffold/random are current; Bioavailability scaffold/random require targeted replay |
| Full-flat versus full-mechanism | Preserved in the source/reasoning matrix; current-conditioned rerun is required |
| Scaffold versus random | Current for BBB and Skin progressive curves; Bioavailability requires targeted replay |
| Identity-blind versus visible | Both implementations and complete historical artifacts are retained, but there is no complete current-conditioned matched pair yet |

The last complete visible and blind source matrices use different historical
benchmark lineages. They must not be used as a matched visibility comparison.
The previously indexed record-supported-v2 visible directory was only a partial
reuse checkpoint and contains no batch-level predictions or metrics.

## Current scaffold progressive curve

Macro-F1 on valid:

| Task | None | L1 | L2 | L3 | L4 | L5 | L6 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBB | 0.6207 | 0.7314 | 0.7475 | 0.7475 | 0.7563 | 0.7563 | — | current |
| Bioavailability | 0.6314 | 0.6900 | 0.7312 | 0.7378 | 0.7545 | 0.7545 | 0.7608 | pre-fix reference; replay required |
| Skin | 0.6119 | 0.6004 | 0.5939 | — | — | — | — | current |

The Bioavailability benchmark valid rows are unchanged, but six erroneous
nitrendipine-identity source records were removed and the held-out retrieval
index hash changed. Therefore evidence-dependent predictions are stale even
though the valid input hash still matches.

## Current random progressive curve

Macro-F1 on valid:

| Task | None | L1 | L2 | L3 | L4 | L5 | L6 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBB | 0.5165 | 0.6412 | 0.6624 | 0.6579 | 0.6792 | 0.6779 | — | current |
| Bioavailability | 0.5948 | 0.7695 | 0.7572 | 0.7612 | 0.7634 | 0.7615 | 0.7675 | pre-fix reference; replay required |
| Skin | 0.6848 | 0.7204 | 0.7212 | — | — | — | — | current |

Random retrieval uses `parent_disjoint`. Applying scaffold exclusion to these
rows would define a different experiment and must use a distinct manifest.

## Latest scaffold baselines

Macro-F1 on valid:

| Task | MiniMol head | MiniMol KNN same→null | MiniMol KNN all train | Morgan KNN same→null | Morgan KNN all train | Status |
|---|---:|---:|---:|---:|---:|---|
| BBB | 0.6674 | 0.6071 | 0.5708 | 0.5958 | 0.6174 | current |
| Bioavailability | 0.5872 | 0.5984 | 0.6229 | 0.5891 | 0.5888 | pre-fix reference; retrain required |
| Skin | 0.6030 | 0.5755 | 0.5755 | 0.5208 | 0.5180 | current |

The Bioavailability correction removed two train rows. Trained and KNN
baselines must be regenerated before they are paired with the repaired agent
results.

## Last complete source and organization reference

The retained identity-blind GLM reference is:

```text
outputs/paper/
  molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_glm_5_2_nvfp4/
```

It is complete and useful for preserving the ChEMBL-versus-Starling and
full-flat-versus-full-mechanism implementations, but it is not a current
Conditioned Benchmark result. Macro-F1:

| Task | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 0.6601 | 0.6800 | 0.6936 | 0.6954 | 0.7197 | 0.7092 |
| Bioavailability | 0.5917 | 0.6583 | 0.6190 | 0.6158–0.6099 | 0.6405 | 0.6182 |
| Skin | 0.6057 | 0.5965 | 0.5873 | 0.6164 | 0.6289 | 0.6216 |

Bioavailability has separate historical numeric and full direct conditions,
hence the direct range. These values must remain labeled as a historical
reference until the matched current-conditioned matrix is complete.

## Canonical artifact roots

```text
BBB progressive scaffold:
outputs/paper/starling_conditioned_assay_progressive_visible_v8_global_molecule_source_purity_v5/scaffold_valid_deepseek_v4_flash_0731

Bioavailability progressive scaffold, last complete pre-fix:
outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/scaffold_valid_deepseek_v4_flash_0731

Skin progressive scaffold:
outputs/paper/starling_conditioned_assay_progressive_visible_v7_source_purity_v1/scaffold_valid_deepseek_v4_flash_0731

Progressive random combined root:
outputs/paper/starling_conditioned_assay_progressive_random_quality_v2_parent_disjoint/random_valid_deepseek_v4_flash_0731

One-shot family curve, last complete reference:
outputs/paper/starling_conditioned_assay_family_curve_v1/scaffold_valid_epyc_deepseek_v4_flash_0731
```

For the exact status, source/index hashes, and result roles, read
`current_conditioned_results.json`; this Markdown summary deliberately does not
duplicate every manifest field.

## Publication boundary

- Do not publish Bioavailability post-fix scores until targeted progressive
  replay and baseline retraining are complete.
- Do not present the historical blind and visible roots as a matched comparison.
- Do not tune a method on formal test results.
- Do not mix ClinTox source-role labels with assay-vote counts.
- Do not promote transient smokes, timeout retries, router/RL no-go experiments,
  or source probes into the paper result registry.
