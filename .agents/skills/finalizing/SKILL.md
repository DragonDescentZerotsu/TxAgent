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
