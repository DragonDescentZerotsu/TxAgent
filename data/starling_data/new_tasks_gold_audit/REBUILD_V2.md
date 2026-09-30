> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI / Carcinogens repaired source gold v2.1

The active `gold_v2` files and both conditioned split schemes have been rebuilt
with source contracts v2.1. The frozen acquisition remains reference source commit
`45663daaad7fd8392a0254bd78793a9e0a581a7a`. No TDC labels were imported.
The initial `gold_v1` and pre-repair TDC comparison are historical audit provenance.

## Corrections and review scope

This repair addresses observed source-provenance, negative-endpoint and query-condition
errors; it does not establish exhaustive accuracy of every source extraction.
**91 additional unique source records** were individually read, bringing the active
ledger to **297** (DILI164, Carcinogens133). This is Codex semantic reading of supplied
record text, **not human-expert or full-paper review**. Sampling covers24 previously
diagnosed cases,39 rule-change samples,4 records from two negative cohorts, and24
retained candidates. Deliberately targeted samples do not estimate overall accuracy.

- Withdrawn the five diagnosed Tacrine/PABA/Azathioprine/Melatonin/Monocrotaline votes:
  unresolved summaries, old case references and unnamed cited experiments cannot
  create independent reporting-PMID votes. Dose or duration alone is insufficient.
  Multi-digit refs, summary-of, named-author and personal-communication patterns are
  checked. `study by hepatic artery` is an administration route, not an author.
- Original-study attribution preserves explicitly current experiments despite
  background method or prior-study citations. Explicit NTP bioassay institution/year
  references are handled; multiple original reports remain pending.
- Query conditions are checked after normalization, including reviewed replacements.
  Leukemia/lymphoma/adenoma/sarcoma, liver failure/VOD and outcome predicates cannot
  be copied as unreviewed query context. Shared baseline diagnoses, doses, genotypes,
  co-treatment and genuinely pooled exposure arms remain where supported.
- Rifampicin's null concentration-to-hepatotoxicity association is not an absence
  of DILI; benzene additive-risk-model assumptions are not observed carcinogenicity.
  Source-label direction alone cannot bypass these distinctions.
- Recovered two explicit Ketamine negative cohorts (PMIDs28492426 and34092024).
  Four extractions collapse to two study votes. Conditions retain oral depression
  treatment versus chronic-pain sublingual treatment, retrospective ascertainment,
  regimen and duration. These are conditioned negatives, not universal drug safety.
- NTP1992 GBL and NTP2004 cinnamaldehyde negative bioassays were retained through
  original-study attribution and pure exposure conditions. Unresolved chlorpheniramine,
  sulfamethazine and lithocholic-acid negative subclaims stay in records and review
  queues; no unknown original study or test article was invented to add a vote.

Each verdict is bound to permanent UID and full raw payload SHA-256.
`rebuild_reviewed_records.jsonl` contains all297 records, notes, final decisions and
voter membership. `v2_repair/reviewed_records.jsonl` contains this round's91 reviews.
`v2_repair/*_vote_changes.jsonl` records every representative-vote change; additions
and withdrawals include deduplication/representative changes, not just new experiments.
The five known bad UIDs are checked against every vote's supporting UID list.

## Current counts

Counts in parentheses are negatives. Columns use different units.

| Task | Source candidates | Identity-verified study votes | Complete parent-condition gold | Benchmark subset |
|---|---:|---:|---:|---:|
| DILI | 5,589 (285) | 4,737 (237) | 2,615 (196) | 569 (43) |
| Carcinogens | 18,482 (565) | 12,624 (409) | 10,849 (407) | 1,191 (60) |

All **4,915,196 source records** remain: DILI1,645,109, Carcinogens3,270,087.
Strict vote filters do not delete records. Raw fields, original SMILES, UIDs, nulls,
zeros and source ordinals are retained. Molecular identities and converted records
were reused; the source-decision pass uses32 forked workers and identity lookups reuse
cached exact names/official synonyms. The original128-worker conversion took230.26s;
that is a historical conversion time, not this repair's total duration.

The original base sources themselves are positive-enriched:
DILI104,029 positive /15,563 negative /69,580 other;
Carcinogens236,310 positive /33,727 negative /84,349 other.
These are extraction categories, not accepted independent gold labels.

## Gold and split contract

One resolved study-parent-condition contributes at most one vote. Within-study
conflicts do not vote. Across studies, accept >=60% agreement with reported external
conditions or >=70% without them, excluding ties. One qualifying study suffices;
there is no confidence0.9 cutoff or mandatory two-PMID requirement. Exact parent
identity is required for gold, with frozen PubChem evidence; it does not itself
verify the paper's actual test article.

`gold_v2/parent_condition_labels.jsonl` retains ALL accepted gold, including sparse
conditions. Only benchmark selection requires three parents and three nonempty
scaffolds per condition. Coverage failures retain original gold and reasons in
`parent_condition_exclusions.jsonl`. The benchmark phase now automatically exports
complete gold, exclusions and selection audit before finalization.

| Task | Scheme | Train | Valid | Test |
|---|---|---:|---:|---:|
| DILI | scaffold | 457 (35 negative) | 56 (4 negative) | 56 (4 negative) |
| DILI | random | 455 (35 negative) | 57 (4 negative) | 57 (4 negative) |
| Carcinogens | scaffold | 693 (36 negative) | 249 (15 negative) | 249 (9 negative) |
| Carcinogens | random | 943 (52 negative) | 124 (3 negative) | 124 (5 negative) |

Both schemes share the exact same parent-condition-label cohort. Parents never
cross splits; scaffold also forbids scaffold overlap. Every retained condition and
both labels appear in all three partitions. Existing quality objectives and minimum
feasible equal held-out sizes remain in effect. Existing five tasks'86 frozen split
files are unchanged.

Negative evaluation remains limited: DILI valid/test each have4 negatives;
Carcinogens scaffold15/9 and random3/5. This repair does not solve acquisition bias
or justify reliable negative-class performance claims. Mechanism-family review and
held-out-filtered retrieval indices remain pending; no model evaluation was run.

## Verification and reproduction

68 regression/shared-builder tests pass. The release validator checks current policy
and artifact hashes, identity proofs, vote uniqueness/aggregation, reviewed-withhold
membership, rendered-condition guards, complete-gold inclusion, split equivalence,
parent/scaffold disjointness, old split preservation and source-field conservation.
The exact release receipt is `data/starling_data/new_tasks_processing_receipt.json`.

Build commands: `tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md`.
From the repository root in the existing vllm environment:

```bash
PYTHONPATH=. python data/starling_data/new_tasks_gold_audit/validate_rebuild.py
```
