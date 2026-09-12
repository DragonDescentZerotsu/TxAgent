> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI / Carcinogens progressive levels

> Publication update (2026-09-07): the reviewed merged cohort is now active at
> `data/conditioned_benchmark/{DILI,Carcinogens}/{scaffold,random}`. All staged
> input/heldout bytes and existing progressive indices are unchanged. References
> below to the staged trial describe construction provenance; current path mapping
> is in `data/conditioned_benchmark/external_augmentation_publication.json`.


This release builds seven record-level families per task, with source-preserving overlays and split-specific retrieval for the **TDC-augmented staged benchmark**. It does not launch model evaluation or change the other tasks' evidence/index inputs. The source-only active cohorts and the TDC-augmented trial remain distinct lineages.

| Level | DILI meaning | DILI source records | Carcinogens meaning | Carcinogens source records |
|---|---|---:|---|---:|
| L1 | 实际参与严格 gold 投票的记录 | 4,729 | 实际参与严格 gold 投票的记录 | 12,571 |
| L2 | 其他肝损伤结局、预测及包含 direct 内容的混合段落 | 270,203 | 其他致癌/肿瘤结局、预测及包含 direct 内容的混合段落 | 483,163 |
| L3 | 肝细胞损伤、存活与功能储备 | 141,909 | 细胞转化、永生化与生长失控 | 750,313 |
| L4 | 胆汁酸、肝胆转运与胆小管稳态 | 76,122 | 遗传毒性、DNA 损伤与修复 | 650,552 |
| L5 | 代谢活化与线粒体能量损伤 | 294,036 | 反应性代谢、亲电反应与氧化损伤 | 749,494 |
| L6 | 肝脏免疫作用与分子响应 | 196,579 | 表观遗传与细胞间生长约束 | 218,403 |
| L7 | 肝脏氧化还原、脂质、ER/钙、溶酶体与自噬应激 | 403,029 | 损伤后再生与受体介导增殖 | 287,294 |

These are eligible source-membership counts before split-specific held-out filtering, not benchmark row counts or representative-card counts. Family membership is per record, never inferred from the source filename or a physical assay's first level.

## Source preservation and direct containment

All **4,915,196 source records** remain in both canonical source and progressive overlay. Raw payloads, permanent UIDs and original scientific fields are conserved. Retrieval excludes 258,502 DILI and 118,297 Carcinogens records for source identity, reviewed identity, explicit off-task scope or unresolved endpoints; it does not delete those records. DILI's largest additional exclusion is explicit nonhepatic assays without a hepatic endpoint (186,403 rows). Low support count, negative direction and protective roles alone do not justify deleting source records.

From v1–v5 mechanism acquisition sources, **93,407 DILI /156,182 Carcinogens records** carrying direct-related content move into L2. Predictions, secondary claims, mixed protective/other-agent passages and uncertain direct assertions do not obtain a vote from L2 placement. Original roles and conditions stay in the complete cards. Actual voters alone enter L1, including accepted votes whose parent-condition may subsequently fail aggregate or split coverage gates.

Both L1 and L2 have `heldout_filter_scope=direct_outcome`. Each scheme removes this scope for the complete valid+test parent union across every source and condition. The filter never relies only on base source IDs. TDC labels do not enter source records, votes or evidence cards. Higher mechanism evidence remains available subject to the shared query parent/scaffold policy.

## Semantic review and gold repair

Initial targeted and stratified review read 69 distinct DILI and 72 Carcinogens complete raw records across all six acquisition sources, including actual voters, excluded records, direct-containing mechanism passages and ordinary mechanisms. Additional review re-read all 11 DILI /57 Carcinogens votes affected by the first citation-rule correction. These counts overlap. DILI also re-read two existing base study records and the newly recovered pravastatin record. This is **Codex record-text semantic reading, not human-expert or exhaustive original-paper review**.

The independent guard also replayed 103 Carcinogens diagnostic records. Their apparent direct associations came from concatenating unrelated fields; preserving each complete field boundary resolves all 103. The boundary audit records UID, payload hash and the previous matched substring. This is rule diagnostics, not 103 additional full-paper reviews.

Every placement decision binds a permanent UID to a raw-payload SHA-256 and a concrete reason. Twelve reviewed identity problems remain explicit holds; propagation uses exact structure plus name/claim or an explicitly justified material/coordination scope. Proper source identities sharing an unrelated erroneous name are not automatically discarded.

- DILI: 9 old votes withdrawn; 1 original pravastatin hepatitis case recovered because `reference 0.44–3.40 g/L` is a laboratory interval, not a cited study. Final **4,729 votes (4,495 positive /234 negative)**.
- Carcinogens: 53 old votes withdrawn; 4 original-study records initially rejected by the new rule restored. Final **12,571 votes (12,165 positive /406 negative)**. The 4 restores are already included in that total.
- Rimegepant/rivaroxaban pooled-trial summaries and cited-only outcomes cannot become new independent studies. `reference [15]` is detected, while zero-dose reference groups and laboratory intervals are not literature references. Primary results with collateral method/background citations require payload-bound review.
- Oxaliplatin cannot transfer its label to a free-diamine FragmentParent. The affected source vote is withdrawn and the reviewed coordination input is excluded from retrieval.
- Two DILI mechanism-source clinical candidates already have base votes for the same original study; they do not add votes. Twelve Carcinogens candidates failed original-study, identity, overall-endpoint or agent-role requirements.
- Restored Carcinogens conditions remove outcome-bearing RR/CI text and preserve genotype/co-exposure context. Source gold and both split schemes were regenerated before the TDC trial and indices.

Audit artifacts: `data/starling_data/<task>/level_review_v1/{placement_decisions,identity_holds,gold_review_updates,restore_review_updates,withdrawn_vote_audit}.jsonl`; full reviewed samples are DILI `sample.jsonl` and Carcinogens `review_sample.jsonl`. `progressive_gold_rebuild.json` pins before/after vote hashes and exact changes.

## Current staged benchmark

| Task | Unique parents | Rows | Positive / negative | Scaffold train / valid / test rows | Random train / valid / test rows |
|---|---:|---:|---:|---|---|
| DILI | 845 | 1,034 | 760 / 274 | 828 / 103 / 103 | 828 / 103 / 103 |
| Carcinogens | 719 | 1,464 | 1,186 / 278 | 908 / 278 / 278 | 1164 / 150 / 150 |

Starling held-out singleton minimization and valid/test balancing remain higher priority than label balancing. TDC external labels remain exempt from assay vote-count objectives. Every reported minimum is checked against the selected rows; parent separation and full condition coverage remain hard constraints. Source-specific TDC class minima remain enforced.

## Build and validation

Use the installed vllm environment with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Gold/source updates use the existing `build_source_gold` prepare, votes, benchmark and finalize phases with the frozen review/name-resolution ledgers. Then rebuild `build_tdc_augmented_benchmark`; no new network identity lookups are needed.

```bash
python -m tools.chembl_tool.common.starling.build_progressive_sources --workers 64
python -m tools.chembl_tool.common.starling.validate_progressive_sources --phase all --workers 32
```

The builder uses row-group process workers, cached unique molecular identities, local NVMe staging, and the existing accelerated family-catalog/index implementation. It does not reconvert the raw corpus or initialize an LLM. Catalogs use `outputs/paper/starling_conditioned_assay_family_curve_v1/family_catalogs/{dili,carcinogens}`. Overlays, row ledgers, indices and receipts live at `data/starling_data/<task>/progressive_v1/`; split indices are `indices/{scaffold,random}`.

The task configs expose shared progressive contracts. The shared retrieval API is validated without model calls. Existing paper runners that default to active source-only paths must not silently use these staged-trial indices; any evaluation launch must bind its input and held-out hashes to this staged benchmark.

Only `validation_all.json` with status=passed is the full release gate. It verifies full source conservation, exact voter membership and raw hashes, independent direct containment, all indexed-card surfaces, held-out removal, record-level family visibility, and append-only progressive retrieval. The row-membership `manifest.json`, `retrieval_manifest.json`, per-split index manifests and validation receipt form the authoritative lineage.

## Built split-specific indices

| Task | Scheme | Molecules in evidence library | Assay-molecule groups | Held-out L1/L2 records removed | Held-out direct records remaining |
|---|---|---:|---:|---:|---:|
| dili | scaffold | 31,884 | 916,736 | 37,589 | 0 |
| dili | random | 31,886 | 910,472 | 76,914 | 0 |
| carcinogens | scaffold | 86,248 | 2,469,589 | 98,427 | 0 |
| carcinogens | random | 86,256 | 2,475,024 | 69,266 | 0 |

All four builds completed successfully. These molecule counts describe the retrieval libraries, not benchmark query molecules. Focused regression tests: **107 passed**. Source-gold and TDC split validation receipts were refreshed after the final gold changes. Full card/progressive validation status is recorded separately in each task’s `validation_all.json`.

The full card and seven-level cumulative/append-only checks passed for all four task/split combinations, with four positive/negative Starling/TDC query rows tested per combination. Every indexed representative card traces to its original UID, exact scientific surface and record-level family. Final input-hash stability verification also passed. Both tasks now have `validation_all.json` with **status=passed**; all four indices are validated for the staged TDC-augmented benchmark. Existing-task benchmark artifacts remain unchanged (86 protected files checked), and the TDC merge validation preserves all 150 input snapshot files. No model evaluations were run.
