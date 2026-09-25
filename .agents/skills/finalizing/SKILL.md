---
name: finalizing
description: Publish completed Joseph optimization or inference results into the FINAL collection with pinned selections, prompts, metrics, traces, and a Git archive.
---

# Finalizing Joseph results

Use [finalize.md](../../../finalize.md) for the current FINAL schema and verification commands. Keep completed source batches and existing FINAL arms immutable; give a changed prompt, model, selection, or request configuration a new arm path.

For each new arm:

1. Verify complete predictions and per-query traces, zero failed queries, and metrics recomputed from predictions. Pin the source batch, retrieval and selection manifests, and actual served model/provider for every query.
2. Save the frozen optimization hyperparameters and selected direct/indirect manifests, the exact prompt assets used by inference, metrics, predictions, combined trace, per-query requests/responses/traces, route ledger, and file hashes. Preserve the validation decision separately from test results; disclose any prompt or model choice made after viewing test.
3. Add the arm to the model index and `final/collection.json`, preserving all existing entries. Save a compact report, numeric TSV, and provenance manifest under `outputs/analysis/record_selection/<study-id>/`.

**Compress once per commit, after all intended FINAL additions and their checks are complete.** Before committing, rebuild `final/git_bundle/`, verify its part checksums and archive integrity, and confirm its collection hash. Do not rebuild the compressed archive after each arm or intermediate edit. Commit the updated collection, archive, report, and documentation together; report whether the commit and any requested push succeeded.

## Reference result table

Maintain this same table in `AGENTS.md` when a result is explicitly promoted. Preserve the listed reference series; do not silently replace cells with the highest test score across models. Starling means Gold-v1. Each promoted value must match a complete FINAL arm and be documented in its report.

| Task | Benchmark | Direct macro F1 | Direct + Indirect macro F1 |
| --- | --- | ---: | ---: |
| bbb_martins | Starling | 0.673178 | 0.725340 |
| bioavailability_ma | Starling | 0.768091 | 0.861340 |
| skin_reaction | Starling | 0.599320 | 0.615042 |
| ames | Starling | 0.708578 | 0.773887* |
| dili | Starling | 0.627614 | 0.650819 |
| carcinogens | Starling | 0.532669 | 0.647253 |
| bbb_martins | TDC | 0.865673 | 0.880814 |
| bioavailability_ma | TDC | 0.746709 | 0.830732 |
| skin_reaction | TDC | 0.700912 | 0.666875 |
| ames | TDC | 0.782289 | 0.800280* |
| dili | TDC | 0.829711 | 0.799900* |
| carcinogens | TDC | 0.695652 | 0.878261 |

*Reference value supplied for the table, not a verified complete arm in the current `final/collection.json`. The AMES values came from provisional coverage. Verify and replace a marked cell only when its completed result is explicitly promoted. The TDC Skin direct value is an equivalent-cohort v1 replay; its FINAL manifest pins that provenance.
