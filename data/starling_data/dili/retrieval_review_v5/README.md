> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI v5: nine confirmed residual source defects

This is an isolated scaffold diagnostic on the frozen v4 source. It implements the nine confirmed findings in `outputs/paper/starling_conditioned_dili_retrieval_review_v4_no_prior_grounded_sim0/error_record_audit_20260912/source_findings.json`. No benchmark labels or old predictions decide a source correction. At the time of this diagnostic, the global source and random index remained v3. The later final publication adopted these v5 repairs for both index schemes in `retrieval_final`; the original diagnostic inputs stay immutable.

- Repair DPPD to N,N'-diphenyl-p-phenylenediamine; preserve the protective observation and the original abstract's lack of protection at 4 h. The earlier extracted numerical table values remain explicitly not independently table-verified.
- Correct two bG/cG graphene records to material identity and remove their false benzoguanamine retrieval associations. Original material/ROS-negative observations remain stored; no raw rows are deleted.
- Qualify three LO-2/HL-7702 records with the documented L-02 HeLa-derivative/misidentification limitation and unresolved authentication of the original culture. Preserve measured negatives, lack of protection and LDH injury. They remain nonvoter, qualified source-reported outcomes in L2 under the unchanged direct-containment contract; they are not primary human hepatocyte evidence.
- Repair the 4PBA isomer while keeping the BSEP endpoint in L4.
- Move fish hepatocyte cellular ultrastructure L4 -> L3, preserving species and assay discordance.
- Move AICAR-associated mitochondrial network morphology L4 -> L5 and correct the collagen-sandwich culture context. The original Methods explicitly use 'ribonucleotide' without a catalog number. Therefore the riboside/phosphate identity remains unresolved and visibly qualified; do not silently substitute acadesine.

`content_repairs.json` is hash-bound to the full v4 source and each original payload. `reviewed_rows.jsonl` stores before/after fields. `validate_source.py` verifies all 1,645,109 source rows, every unchanged row, raw acquisitions, voter flags and 27 frozen files. Only the scaffold catalog/index is rebuilt; L1-only heldout pre-exclusion, L2 retention and query-time scaffold disjoint remain unchanged.

`run_scaffold_repair.py` runs source repair, source/index validation, reconstruction of all 804 selected surfaces, fresh preparation of changed queries, exact prompt/tool/progressive-prefix reuse validation, and simultaneous valid/test progressive/full-flat replay. Each mode uses Hosted Flash0731, 256 slots, race width 6 and max_tokens 20480. New selected surfaces outside the nine confirmed repairs are listed in `pending_reselected_cards.jsonl`; this is not a clean-library certificate. Source review does not see benchmark labels/predictions.

Historical execution receipt: `workflow_progress.json`. Completed results: `outputs/paper/starling_conditioned_dili_retrieval_review_v5_no_prior_grounded_sim0/REPORT.md`. Final source and audit restoration now use the packaged `retrieval_final` release; the intermediate NVMe build tree is retired.
