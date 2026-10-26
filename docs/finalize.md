# Frozen FINAL collection

The [Git bundle](../outputs/paper/assay_transfer_harness/evidence/final/git_bundle/README.md)
contains the published FINAL results, per-query traces, source bindings and frozen
assembly snapshots. Restore it using that guide before inspecting raw arm files.
Those files are inside the archive, so their paths are not browsable GitHub links.
All paths below are relative to the repository root.

| Restored location under `outputs/paper/assay_transfer_harness/evidence/final/` | Purpose |
| --- | --- |
| `collection.json` | Complete arm inventory, query counts and named selections |
| `<model>/test/<benchmark>/<task>/<selection>/` | Arm contract, predictions, metrics, source provenance and traces |
| `test_macro_f1_12_task_benchmarks_*_selected_*.tsv` | Selected model tables with arm paths and hashes |
| `ablations/README.md` | Frozen ablation/scaling tables and protocol caveats |
| `publication_projection.json` | Publication-copy transformations and original-hash bindings |

[Results and provenance limits](results.md#existing-final-collection) records the
retained comparison tables. Historical arms, validation-selected arms and successors
chosen after viewing test results remain distinct. In particular, the archived
best-test Gold selections do not replace the retained comparison or establish the
paper's user-reported Ours provenance. Read the selected arm's manifest and report;
never choose a result just because it has the highest score or newest directory.

The [Chinese handoff status](release_refactor.md) owns current acceptance and
remaining dependencies. Work is shared on `merge_release_v1`; main remains unchanged
until collaborator review and handoff are complete. Updating documentation or
pushing an unchanged collection does not require rebuilding the archive.

## Updating the final files

Work from the repository root after restoring the collection. These steps apply
only to explicitly approved new results. Keep published source batches and existing
arms unchanged; a replacement needs a reviewed successor collection.

1. Confirm the selected source arm is complete: every test query has a valid
   prediction and per-query trace, and recomputed metrics match `metrics.json`.
   Check that its query UIDs, named retrieval/selection versions and effective settings match the intended optimization.
2. Install the approved arm under
   `outputs/paper/assay_transfer_harness/evidence/final/<model>/test/<benchmark>/<task>/<selection>/`.
   Include `manifest.json`, `metrics.json`, `predictions.jsonl`,
   `query_provenance.tsv`, `files.tsv`, `trace_messages.jsonl`, and all
   `runs/<query>/` payload and checkpoint files. Use one `evidence_final_arm.v3`
   contract with source run UIDs, named versions, effective settings, ordered query
   UIDs and sparse overrides; omit copied query manifests and experiment hashes.
   Record each query's actual served model and
   provider in `query_provenance.tsv`.
3. Add its relative path, run UID, query count, macro-F1, and route
   status to `final/collection.json`. Keep `arms` sorted, recompute the total
   query count, and update `updated_at_utc`. Save the result report, numerical
   TSV, and compact provenance under `outputs/analysis/record_selection/<study-id>/`;
   synchronize selected tables and their caveats in `docs/results.md`.
4. Check the collection index and fully verify only new arms. Existing published
   arms must retain their collection identities and scores. Leave the Git
   bundle unchanged until a push is requested.

   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   from predict.harnesses.branches.manifests import read_record

   final = Path('outputs/paper/assay_transfer_harness/evidence/final')
   collection = json.loads((final / 'collection.json').read_text())
   assert len({a['path'] for a in collection['arms']}) == len(collection['arms'])
   assert collection['n_queries'] == sum(a['n_queries'] for a in collection['arms'])
   for arm in collection['arms']:
       path = final / arm['path']
       manifest = read_record(path / 'manifest.json')
       metrics = json.loads((path / 'metrics.json').read_text())
       assert manifest['run_uid'] == arm['run_uid']
       assert manifest['status'] == 'complete'
       assert manifest['n_queries'] == arm['n_queries']
       assert metrics['n_total'] == metrics['n_successful'] == arm['n_queries']
       assert metrics['macro_f1'] == arm['macro_f1']
       assert sum(1 for _ in path.glob('runs/*/trace_messages.jsonl')) == arm['n_queries']
   print(len(collection['arms']), 'arms;', collection['n_queries'], 'queries')
   PY
   ```

5. When publishing approved collection changes, rebuild and check the tracked
   archive once after all additions. Preserve publication-copy hash bindings and
   review new payloads for private metadata. Do not recompress for documentation-only
   edits or ordinary branch pushes.
   Raw files under `final/` are ignored
   by Git; the compressed parts carry their complete copy, including traces.

   ```bash
   bash outputs/paper/assay_transfer_harness/evidence/final/git_bundle/build.sh
   (
     cd outputs/paper/assay_transfer_harness/evidence/final/git_bundle
     sha256sum -c SHA256SUMS
     cat payload.tar.zst.part-* | zstd -tq
   )
   ```

6. Stage the changed result documentation, `final/git_bundle/`, and the new
   report, result/validation TSVs and provenance. The restored `collection.json`
   travels inside the bundle; do not separately force-track raw collection files.
   The analysis directory is ignored, so stage those files with `git add -f`.
   Replace `STUDY_ID` with the new study directory name:

   ```bash
   git add docs/results.md docs/finalize.md \
     outputs/paper/assay_transfer_harness/evidence/final/git_bundle/
   git add -f outputs/analysis/record_selection/STUDY_ID/{report.md,results.tsv,provenance.json}
   git diff --cached --check
   git diff --cached --name-only
   git commit -m "Add approved Evidence final arms and traces"
   test "$(git branch --show-current)" = merge_release_v1
   git push origin merge_release_v1
   git ls-remote --heads origin refs/heads/merge_release_v1
   ```

   Confirm the remote hash matches `git rev-parse HEAD`. The raw arm directories
   do not need `git add -f`; they are inside the tracked compressed bundle.
