# Archive and root artifacts review

Inspected 2026-10-14 on `login01` in the authoritative TxAgent checkout. The inventory below records the pre-deletion state. Subsequently removed `archive/workspace_history/` (six files, 1,581,894,738 bytes) and, after verifying Git preservation, `archive/chembl_tool/` (56 files, 1,801,784 bytes, three links) plus its `tools/chembl_tool` alias. Three uncommitted README versions are preserved in `docs/look/chembl_tool_readmes/`. Also removed root `artifacts/` after approval: 11 Git-preserved files totaling 172,117,189 bytes and four links. Verified the V10 inputs and linked datasets remain intact. Prompt bundles and instruction snapshots remain unchanged. Details are in `cleanup_candidates.md`, Groups 12, 13 and 15.

The root `artifacts/` cleanup is complete; its descriptions below are historical inspection findings. Do not delete all of `archive/` on the assumption that it is already saved in Git: the relocated directory is untracked, some exact file contents have no demonstrated Git coverage, and linked prompt bundles still support historical rendering.

## Inventory

Counts exclude symlink targets. Sizes are logical bytes; GB/MB below use decimal units.

| Directory | Regular files | Bytes | Symlinks inside | Contents |
| --- | ---: | ---: | ---: | --- |
| `archive/` | 545 | 1,584,952,143 (1.585 GB) | 3 | Prompts, reports, configurations, instruction snapshots, relocation index and scratch backup. |
| `archive/workspace_history/20261012T043138Z/` | 6 | 1,581,894,738 | 0 | Backup of the removed `.scratch/` directory plus inventories/receipts. |
| Remaining archive | 539 | 3,057,405 (3.06 MB) | 3 | Small historical text/template assets; some still loaded for replay. |
| `artifacts/` | 11 | 172,117,189 (172.12 MB) | 4 | Nine BBB/Oral/Skin evidence Parquets, README, manifest and compatibility links. |

## Git coverage

Checked the index, HEAD tree, Git blob identities and locally reachable history without fetching, staging or altering Git state. HEAD was `a209fb4422b5fe62927746f92ac6d04ccb242f11`.

- No entries under `archive/` are in the index or HEAD at their current paths. 544 regular files and three links are untracked; `scratch.tar.gz` is ignored and untracked.
- Exact bytes of 513 of the 545 archive regular files match blobs in HEAD at former or other committed paths. Relocation does not make those contents new. The current relocation directory itself has not been committed.
- For 31 other small files (886,674 bytes in total), no matching blob was found in the checked local history. The additional unmatched file is the 1,581,328,502-byte scratch tarball; it was not rehashed or unpacked, and is neither tracked nor indexed at this path.
- All 11 regular files and all four symlinks under `artifacts/` are tracked, indexed and exactly unchanged from HEAD.

This is a partial clone. Reachable-history enumeration reported 1,939 missing objects; a referenced missing blob still establishes committed identity, but cannot be restored offline until available. No remote availability or un-fetched branches were checked. Absence of a match is not a proof that a file never appeared in any remote or unreachable historical commit.

## What the large scratch backup contains

The saved inventory lists 1,960 regular files totaling 4,101,881,012 uncompressed bytes, nine links and 227 directories. The tarball compresses these to 1.581 GB. Contents include:

- Semantic aggregation queues and JSONL request payloads, including two SQLite queues of about 494 MB and 482 MB.
- A 736 MB BBB semantic-groups JSONL file.
- Intermediate BBB pair-bucket build stages (about 490 MB combined), plus collapsed V8 record tables.
- Skin/BBB/Oral exclusion and unit-mapping analysis outputs, including Skin tables of about 278 MB and 142 MB.
- Small historical smoke-test metrics, scripts, virtual-environment packages and bytecode.

This is a backup of old working material, not an active cache consumed by prediction. It is not accurate to describe every member as empty or disposable: the inventory includes derived analysis tables and smoke-test results. Deleting it means giving up this local scratch recovery copy unless the user chooses to preserve it elsewhere.

Its README also describes restoring a historical `.scratch/skin_endpoint_unit_profile_v3.json` alias. That member is a symlink; the actual profile remains under `data/artifacts/evidence_library_assets/skin_reaction/construction_assets/measurement_resolution_v1/`. Preserve the underlying data independently of any decision about the tarball. The old provider configurations mentioned in `pinned_inputs.json` were outside `.scratch/` and were never included in this tarball.

## Archive dependencies

Found 15 incoming links in maintained ownership trees: 12 prediction prompt bundles, the progressive asset directory, its runbook and `tools/chembl_tool`. The prompt loader in `predict/harnesses/branches/flat.py` resolves exact versions through these directories and reads their templates/tasks/provenance. Deletion would break those supported render/replay paths even though current default prediction uses `final_full_flat/`.

| Incoming path | Archive target |
| --- | --- |
| `predict/PROGRESSIVE_PREDICTION_RUNBOOK.md` | `archive/progressive/PROGRESSIVE_PREDICTION_RUNBOOK.md` |
| `predict/harnesses/progressive` | `archive/progressive` |
| `predict/harnesses/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1` | `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1` |
| `predict/harnesses/branches/prompts/full_flat_context_v5` | `archive/branches/prompts/full_flat_context_v5` |
| `predict/harnesses/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_scores_v2` | `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_scores_v2` |
| `predict/harnesses/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_v1` | `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_v1` |
| `predict/harnesses/branches/prompts/full_flat_context_v5_six_tasks_upstream_scores_v2` | `archive/branches/prompts/full_flat_context_v5_six_tasks_upstream_scores_v2` |
| `predict/harnesses/branches/prompts/full_flat_context_v6_oral_high_low` | `archive/branches/prompts/full_flat_context_v6_oral_high_low` |
| `predict/harnesses/branches/prompts/full_flat_context_v7_oral_final_only` | `archive/branches/prompts/full_flat_context_v7_oral_final_only` |
| `predict/harnesses/branches/prompts/full_flat_context_v8_context_scoped_transfer_v1` | `archive/branches/prompts/full_flat_context_v8_context_scoped_transfer_v1` |
| `predict/harnesses/branches/prompts/joseph_flat_context_v4_v1` | `archive/branches/prompts/joseph_flat_context_v4_v1` |
| `predict/harnesses/branches/prompts/joseph_flat_v1` | `archive/branches/prompts/joseph_flat_v1` |
| `predict/harnesses/branches/prompts/joseph_flat_v2` | `archive/branches/prompts/joseph_flat_v2` |
| `predict/harnesses/branches/prompts/tianang_flat_v1` | `archive/branches/prompts/tianang_flat_v1` |
| `tools/chembl_tool` | `archive/chembl_tool` |

`tests/data/test_data_layout.py` also explicitly expects the `tools/chembl_tool` archive alias. Root README, prediction documentation and the prior cleanup inventory describe this layout. A full archive removal needs an explicit decision about preserving or retiring these paths and corresponding reference/test updates.

The three links inside the archive lead to two live evidence-construction policy files and an unavailable external ChEMBL directory. Removing links must not delete or follow their targets.

## What root artifacts serves

The nine Parquets form a frozen collaborator snapshot of historical BBB, Oral and Skin evidence: source-record membership by level, plus representative-card references for random and scaffold splits. They are candidate evidence surfaces, not saved predictions and not the ordered UID selections for a new query.

| Task | Source-membership rows recorded in manifest | Scaffold representative cards | Random representative cards |
| --- | ---: | ---: | ---: |
| `bbb_martins` | 498,556 | 246,167 | 246,133 |
| `bioavailability_ma` | 435,472 | 186,242 | 186,284 |
| `skin_reaction` | 66,967 | 25,689 | 24,799 |

Found no maintained code reading these working-tree files directly. `data/processing/evidence_library/build_source_uid_levels.py` references the artifact path but loads it through `git show` at pinned commit `26d44f121141f3738764c0b00dc09e185445341b`. The main-universe importer likewise restores a different historical `current_records` manifest/archive through Git. Removing the working-tree `artifacts/` directory does not erase either committed input.

The artifact README names a retired exporter; it is not a current regeneration command. Normal prediction uses published `data/evidence_libraries/`, frozen retrieval caches and context selections instead.

The four artifact links point to `data/legacy/` evidence stores (Clintox and three V7 archives). No maintained executable reference to the root `artifacts/chembl_tool/tasks/` aliases was found. Deleting those links would not reclaim or delete their target datasets.

## Archive regular files without demonstrated Git coverage

Preserve, commit or explicitly discard these exact working-tree versions before relying on Git as their backup. The tarball is excluded from content hashing; the other rows were compared by exact Git blob identity.

| File | Bytes |
| --- | ---: |
| `archive/AGENTS.md` | 1,227 |
| `archive/RELOCATION.json` | 169,909 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/provenance.json` | 7,201 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/system.jinja` | 2,475 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/tasks.yaml` | 16,773 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/levels/L1.yaml` | 556 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/levels/L2.yaml` | 567 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/levels/L3.yaml` | 490 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/levels/L4.yaml` | 454 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/levels/L5.yaml` | 428 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/levels/L6.yaml` | 280 |
| `archive/branches/prompts/final_full_flat_tdc_carc_direct_priority_v1/levels/L7.yaml` | 193 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_scores_v2/provenance.json` | 4,842 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_scores_v2/tasks.yaml` | 15,689 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_scores_v2/user.jinja` | 2,504 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_scores_v2/levels/L1.yaml` | 651 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_v1/provenance.json` | 4,766 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_v1/tasks.yaml` | 15,689 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_v1/user.jinja` | 2,504 |
| `archive/branches/prompts/full_flat_context_v5_six_tasks_tdc_skin_llna_v1/levels/L1.yaml` | 651 |
| `archive/chembl_tool/README.md` | 754 |
| `archive/chembl_tool/paper_experiments/README.md` | 54,261 |
| `archive/chembl_tool/paper_experiments/rl_lora/README.md` | 573 |
| `archive/progressive/PROGRESSIVE_PREDICTION_RUNBOOK.md` | 7,085 |
| `archive/progressive/README.md` | 7,346 |
| `archive/progressive/prompts/README.md` | 2,570 |
| `archive/workspace_history/20261012T043138Z/README.md` | 2,170 |
| `archive/workspace_history/20261012T043138Z/pinned_inputs.json` | 2,138 |
| `archive/workspace_history/20261012T043138Z/scratch.inventory.json` | 561,189 |
| `archive/workspace_history/20261012T043138Z/scratch.sha256` | 170 |
| `archive/workspace_history/20261012T043138Z/scratch.tar.gz` | 1,581,328,502 |
| `archive/workspace_history/20261012T043138Z/scratch.verification.json` | 569 |

## Recommended cleanup groups

1. Remove the working-tree root `artifacts/` directory after approval: its full contents are Git-preserved, and no direct maintained reader was found. Keep Git-addressed acquisition code and `data/legacy/` targets.
2. Review the 1.581 GB scratch backup separately. It provides almost all potential archive space savings. The decision is whether to discard those historical intermediate analyses/results, not whether Git already holds this backup.
3. Preserve the needed archived prompt bundles under their prediction owner, replacing incoming aliases, before considering complete archive removal. Keep exact prompt bytes and supported version behavior unless explicitly retired.
4. Preserve the small unmatched files and any wanted reports before deleting the rest. The remaining archive totals only 3.06 MB, so blanket deletion offers little disk benefit and needs more reference cleanup.

Approval remains per deletion group. This investigation did not change scientific inputs, completed results, templates or Git state, and did not make model calls.

Inspection receipts (temporary): `/tmp/txagent-archive-artifacts-inventory.json`, `/tmp/txagent-archive-artifacts-git-review.json`, `/tmp/txagent-archive-artifacts-history-status.json`, `/tmp/txagent-archive-artifacts-inbound-links.json`.

## Follow-up: relevance to the current workflow

Historical interest alone is not a reason to retain a working-tree copy. Root
`artifacts/` is a historical evidence snapshot, not run tracking; its local
copies have no identified role in current prediction or evidence construction.
The user's stated preference is to discard history unrelated to their work.

The remaining archive has 539 regular files totaling 3,057,405 bytes:

| Area | Exact contents | Present role |
| --- | --- | --- |
| `branches/prompts/` | 12 flat prompt bundles, 113 files: system/user templates, task and level instructions, mode definitions and provenance. Includes Joseph v1/v2, Tianang v1, context v4-v8, upstream-score and TDC Skin variants, and a Carcinogens direct-priority variant. | Explicit historical/variant rendering still resolves these bundles through symlinks. None is the canonical `final_full_flat` default. Keep or relocate only if retaining that rendering capability; retire it explicitly before deleting its assets. |
| `progressive/` | 75 progressive/filter prompt bundles (347 files), plus nine README/runbook/matrix/template/cache-policy files. | The progressive harness is retired. No maintained executable reader of these archive paths was found; the import-boundary test mentions the retired namespace only to reject it. |
| `instructions/` | 12 ordinary Markdown snapshots covering ClinTox, raw Starling data, harness architecture, activity-transfer benchmarks, paper operations, evidence-task conventions, BBB, Oral, DILI, Skin, BBB assay screening and the trace viewer. | They are not active agent instructions. Some contain old scientific notes and results; no current execution depends on reading them. Active AGENTS.md files and current scientific contracts remain outside this subtree. |
| `chembl_tool/` | 56 regular files: old experiment plans/results, BBB/Oral/Skin audits, two HTML casebooks, Oral example prompts, three provider settings, two generic analog templates and documentation. Three links lead to current unit/qualifier policies and an unavailable external ChEMBL dataset. | No maintained executable reader of its historical templates/reports/settings was found. The index builder's optional precomputed-fingerprint default still uses the external dataset alias; it currently falls back to SMILES fingerprints because the external path is unavailable. Full cleanup should update that default/alias rather than leave a dangling path. |
| Root archive metadata | `AGENTS.md` and `RELOCATION.json`. | Instructions and a source-to-archive mapping for this archive layout. They have no separate prediction role. |

The canonical `predict/harnesses/branches/prompts/final_full_flat/` bundle and
its four declared source prompt versions are outside the archive. Its
provenance points to upstream v2/v3, DILI polarized and TDC-v2 mixed bundles;
none belongs to these twelve archived bundles. Current task-specific prediction
and semantic-ranking prompts also live with their maintained owners. FINAL
publication separately preserves selected prompt snapshots and traces.

Do not infer that all twelve archived bundles are important merely because
the renderer recognizes their names. Some describe unevaluated experiments
or preparation for review. Their current reason to remain is the explicitly
supported historical rendering capability, not generic historical value.

The deleted workspace backup had no relocation-index entries. Its six reviewed
files were rechecked before removal; no nested mounts or local process cwd/argv
uses were found. The underlying Skin endpoint-profile file remains unchanged.
Deletion receipt: `/tmp/txagent-workspace-history-deletion.json`.

## Specific experiment results under chembl_tool

These are report summaries of earlier experiments, principally BBB, Oral and
Skin. The directory does not contain the full current selected FINAL collection.

| Report | Experiments/results described |
| --- | --- |
| `paper_experiments/RESULTS.md` | Earlier TDC and scaffold validation/test matrices: query-only, direct, full-flat and mechanism-split prediction; identity-hidden/visible and parent-disjoint controls; Starling/ChEMBL comparisons; assay-transfer reranking; collapsed-assay KNN/LR/RF validation sweeps; a ClinTox diagnostic. Some cohorts have explicit overlap and post-selection caveats. |
| `paper_experiments/STARLING_BENCHMARK_RESULTS.md` | BBB/Oral/Skin scaffold comparisons across GPT-OSS-120B, GLM-5.2 and DeepSeek variants, Morgan/MiniMol baselines, retrieval scaling, evidence-scope/aggregation changes and router/selector diagnostics. It includes validation results and dated test/sensitivity studies. |
| `paper_experiments/current_conditioned_results.json` | A historically named summary of DeepSeek-V4-Flash-0731 progressive validation: BBB 397 queries at none/L1-L5, Oral 262 at none/L1-L6, Skin 246 at none/L1-L2, plus Morgan/MiniMol baselines. Its source output identities describe earlier V7/V8/V10 progressive runs; the filename does not identify the present default or selected FINAL results. |
| `paper_experiments/rl_lora/ONE_PASS_BASE_VALID_RESULTS.md` | Frozen GPT-OSS-120B one-pass versus three-stage prediction on 820 validation queries, with paired uncertainty, branch metrics and token/call counts. The report flags the historical Bio prompt sanitization caveat. |
| `paper_experiments/rl_lora/VISIBLE_ONE_PASS_VALID_RESULTS.md` | Visible-prefetched one-pass validation on 820 queries and bounded three-step RL/LoRA smoke receipts. Training-status statements describe the dated receipt, not current running jobs. |
| `paper_experiments/reports/skin_reaction_visibility_trace_casebook/` | A portable report and compact case records examining identity-hidden versus identity-visible reasoning errors, including four Skin false-positive examples. |
| `paper_experiments/reports/bbb_gold_voter_coverage_audit/` | A portable audit of BBB gold-voter coverage, missing direction annotations and the source-purity correction. This is a data audit, not a separate prediction-performance sweep. |

Most full predictions/metrics/traces are referenced under `outputs/paper/`;
this archive contains the explanatory reports and, for the Skin casebook,
compact case-level source records. Current selected results are documented in
`docs/results.md` and frozen under
`outputs/paper/assay_transfer_harness/joseph/final/`. The archived documents
should be retained only for particular comparisons or scientific explanations
the user wants, rather than because they happen to contain historical scores.

The user subsequently approved deleting the ChEMBL archive after Git
verification. All result/report files matched committed blobs at the former
`tools/chembl_tool/` paths. Three unmatched README updates are preserved in
[chembl_tool_readmes/](chembl_tool_readmes/). The report listing above describes
the deleted archive and can be used to locate its Git-preserved contents.
The remaining archive has 483 regular files totaling 1,239,204 bytes: the
flat/progressive prompts, twelve instruction snapshots and root metadata.
