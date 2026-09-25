---
name: finalizing
description: Publish completed Joseph optimization or inference results into the FINAL collection with pinned selections, prompts, metrics, and traces; build the Git archive only for a requested push.
---

# Finalizing Joseph results

Use [finalize.md](../../../finalize.md) for the current FINAL schema and verification commands. Keep completed source batches and existing FINAL arms immutable; give a changed prompt, model, selection, or request configuration a new arm path.

For each new arm:

1. Verify complete predictions and per-query traces, zero failed queries, and metrics recomputed from predictions. Pin the source batch, retrieval and selection manifests, and actual served model/provider for every query.
2. Save the frozen optimization hyperparameters and selected direct/indirect manifests, the exact prompt assets used by inference, metrics, predictions, combined trace, per-query requests/responses/traces, route ledger, and file hashes. Preserve the validation decision separately from test results; disclose any prompt or model choice made after viewing test.
3. Add the arm to the model index and `final/collection.json`, preserving all existing entries. Save a compact report, numeric TSV, and provenance manifest under `outputs/analysis/record_selection/<study-id>/`.

**Compress only when the user explicitly requests a push, after all intended FINAL additions and their checks are complete.** Ordinary FINAL updates do not trigger compression or a commit: keep the new readable arms, collection index, report, and documentation local, and leave `final/git_bundle/` untouched. On a requested push, rebuild the bundle once, verify its part checksums, archive integrity, and collection hash, then commit the collection, archive, report, and documentation together and push.

## Final test results by model

Keep the DeepSeek and Luna tables synchronized with `AGENTS.md` when a completed result is explicitly promoted. Each value must match a complete FINAL test arm and be documented in its report. A dash means no completed FINAL arm is recorded for that model and selection. Starling means Gold-v1; TDC includes the version pinned by each arm. Preserve older arms for audit and do not silently replace a selected cell with the highest score across prompts or versions.

### DeepSeek

| Task | Benchmark | Direct macro F1 | Direct + Indirect macro F1 |
| --- | --- | ---: | ---: |
| bbb_martins | Starling | 0.673178 | 0.725340 |
| bioavailability_ma | Starling | 0.768091 | 0.861340 |
| skin_reaction | Starling | 0.599320 | 0.615042 |
| ames | Starling | 0.708578 | 0.774211 |
| dili | Starling | 0.639755 | 0.659233 |
| carcinogens | Starling | 0.532669 | 0.647253 |
| bbb_martins | TDC | 0.865673 | 0.880814 |
| bioavailability_ma | TDC | 0.746709 | 0.830732 |
| skin_reaction | TDC | 0.634097 | 0.666875 |
| ames | TDC | 0.782289 | 0.799350 |
| dili | TDC | 0.819609 | 0.821346 |
| carcinogens | TDC | 0.817392 | 0.790611 |

The DeepSeek TDC Skin cells are complete 82-query **TDC-v1** upstream-v2 runs. The final pair was retained after reviewing test comparisons, and the same test cohort was used in later prompt and model iterations; treat these as historical descriptive scores, not an untouched TDC-v2 holdout estimate.

The DeepSeek TDC Carcinogens direct cell uses the validation-selected 56-query exact-score upstream-v3 arm. The separate Gold DILI `sr=0` control scored 0.680085 on 402 test queries; it is an exploratory FINAL successor and does not replace the validation-selected 0.659233 cell.

### Luna

| Task | Benchmark | Direct macro F1 | Direct + Indirect macro F1 |
| --- | --- | ---: | ---: |
| bbb_martins | Starling | 0.666929 | 0.722610 |
| bioavailability_ma | Starling | 0.760093 | 0.856127 |
| skin_reaction | Starling | 0.559252 | 0.564175 |
| ames | Starling | 0.667730 | 0.706003 |
| dili | Starling | 0.621456 | 0.665266 |
| carcinogens | Starling | 0.482328 | 0.623776 |
| bbb_martins | TDC | 0.828823 | 0.846330 |
| bioavailability_ma | TDC | 0.718372 | 0.774023 |
| skin_reaction | TDC | 0.700912 | 0.611005 |
| ames | TDC | 0.787868 | 0.811343 |
| dili | TDC | 0.790850 | 0.801545 |
| carcinogens | TDC | 0.841539 | 0.878261 |

The Luna Gold-v1 Skin direct arm uses a profile selected on 100 `valid_small` queries and scored 0.559252 on all 241 test queries. The Luna TDC Skin direct arm is an equivalent-cohort TDC-v1 replay filed in the TDC-v2 FINAL view. Its mixed successor was selected on all 40 TDC-v2 validation queries and scored 0.611005 on test; an exploratory test arm scored higher but did not change the validation choice. The Luna TDC Carcinogens mixed value uses the positive-gate prompt selected after viewing test results; its FINAL provenance records that decision. The Gold Carcinogens Luna arms are model-specific results below their existing DeepSeek counterparts. TDC-v2 AMES Luna exceeds both existing DeepSeek cells; its mixed profile was selected on a 100-query validation cohort.
