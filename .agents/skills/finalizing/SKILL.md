---
name: finalizing
description: Publish completed Joseph optimization or inference results into the FINAL collection with pinned selections, prompts, metrics, and traces; build the Git archive only for a requested push.
---

# Finalizing Joseph results

Use [finalize.md](../../../docs/finalize.md) for the current FINAL schema and verification commands. Keep completed source batches and existing FINAL arms immutable; give a changed prompt, model, selection, or request configuration a new arm path.

For each new arm:

1. Verify complete predictions and per-query traces, zero failed queries, and metrics recomputed from predictions. Record source batch UIDs, named retrieval and selection versions, effective scientific settings, and actual served model/provider for every query.
2. Save the frozen optimization hyperparameters and selected direct/indirect manifests, the exact prompt assets used by inference, metrics, predictions, combined trace, per-query requests/responses/traces, route ledger, and one shared arm contract (`joseph_final_arm.v3`). Query stage state stays in checkpoints; do not copy query manifests or embed paths/content fingerprints as experiment identity. Preserve the validation decision separately from test results; disclose any prompt or model choice made after viewing test.
3. Add the arm to the model index and `final/collection.json`, preserving all existing entries. Save a compact report, numeric TSV, and provenance manifest under `outputs/analysis/record_selection/<study-id>/`.

**Compress only when the user explicitly requests a push, after all intended FINAL additions and their checks are complete.** Ordinary FINAL updates do not trigger compression or a commit: keep the new readable arms, collection index, report, and documentation local, and leave `final/git_bundle/` untouched. On a requested push, rebuild the bundle once, verify its part checksums, archive integrity, and collection hash, then commit the collection, archive, report, and documentation together and push.

## Selected test results

Maintain the selected DeepSeek and Luna tables and their provenance caveats in
[docs/results.md](../../../docs/results.md) when a completed arm is explicitly promoted.
Preserve historical arms and validation choices; never silently select the highest test score.
