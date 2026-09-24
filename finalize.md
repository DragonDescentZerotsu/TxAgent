# Final Joseph full-test collection

The [final collection](outputs/paper/assay_transfer_harness/joseph/final/collection.json)
contains 24 complete arms and 8,546 successful arm-query results. It includes
Gold BBB, Oral Bioavailability, AMES, Carcinogens, DILI, and Skin Reaction
direct and direct+indirect arms; TDC Oral Bioavailability and Skin Reaction
direct and direct+indirect arms; and TDC-v2 AMES, BBB, Carcinogens, and DILI
direct and direct+indirect arms.
Each arm includes predictions, metrics, per-query run artifacts and traces,
a combined trace, a query-level model/provider ledger, and
hash-pinned source and copied-file manifests. The original batches remain at
their source paths.

The [original results report](outputs/analysis/record_selection/joseph_final_gold_bbb_oral_carcinogens_tdc_oral_20260924_v1/report.md)
covers the first seven arms. The [TDC Skin report](outputs/analysis/record_selection/joseph_final_tdc_skin_upstream_v2_test_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_tdc_skin_upstream_v2_test_20260924_v1/results.tsv)
cover the selected upstream-v2 pair: direct macro-F1 0.634097 and
direct+indirect macro-F1 0.666875, each on all 82 test queries. Gold Carcinogens
direct uses `ga075_mc000_label000`. The [Gold Skin and TDC Oral direct report](outputs/analysis/record_selection/joseph_final_gold_skin_tdc_oral_direct_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_gold_skin_tdc_oral_direct_20260924_v1/results.tsv)
cover three further arms. TDC Oral direct+indirect joins 87 recovery-leaf queries
with 41 valid predecessor runs. Each arm's `query_provenance.tsv` records its
actual served model and provider.

The [recovery snapshot](outputs/analysis/record_selection/full_test_upstream_v2_recovery_20260924_v1/report.md)
is a dated selection aid; final arm manifests pin their installed source leaves.
The [TDC-v2 Carcinogens report](outputs/analysis/record_selection/joseph_final_tdc_v2_carcinogens_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_tdc_v2_carcinogens_20260924_v1/results.tsv)
cover the two new 56-query arms: direct macro-F1 0.695652 and
direct+indirect macro-F1 0.726830. OpenRouter upstream routes differed between
these arms; their query-level provenance records the actual route.

The [TDC-v2 BBB report](outputs/analysis/record_selection/joseph_final_tdc_v2_bbb_filtered_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_tdc_v2_bbb_filtered_20260924_v1/results.tsv)
cover the 406-query direct and filtered direct+indirect arms: macro-F1 0.865673
and 0.880814, respectively. The indirect selector removed prediction-only
records before its joint 50-record selection.

The [TDC-v2 DILI report](outputs/analysis/record_selection/joseph_final_tdc_v2_dili_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_tdc_v2_dili_20260924_v1/results.tsv)
cover the 96-query direct and direct+indirect arms: macro-F1 0.829711 and
0.760436, respectively. The indirect selector uses the parent-disjoint top-100
universe, DILI tool and semantic scores, and tied diversity terms. Its semantic
projection extension remains marked `candidate_unreviewed`. The prompts and
provider routes differ between arms; the comparison does not isolate the effect
of indirect records. A [same-prompt validation control](outputs/analysis/record_selection/dili_tdc_v2_direct_same_prompt_valid_20260924_v1/report.md)
and its [TSV](outputs/analysis/record_selection/dili_tdc_v2_direct_same_prompt_valid_20260924_v1/results.tsv)
found direct-only macro-F1 0.831541 with five false positives among 17 negative
queries, versus 0.825465 and six false positives with indirect records. Only
two predictions changed; their upstream OpenRouter providers also differed.

The [AMES collection report](outputs/analysis/record_selection/ames_final_collection_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/ames_final_collection_20260924_v1/results.tsv)
cover four new full-test arms. Gold-v1 AMES direct and direct+indirect macro-F1
are 0.708578 and 0.774211 on 274 queries each; TDC-v2 AMES scores are
0.782289 and 0.799350 on 1,457 queries each. TDC mixed query 1158 used a
fresh, identical rendered prompt with a lower token ceiling, recorded in its
arm provenance. The source response sets are complete across preserved runs;
some historical individual matrices remain partial.

The full collection, including traces, is preserved in Git as
[compressed parts](outputs/paper/assay_transfer_harness/joseph/final/git_bundle/README.md).

## Updating the final files

Work from the repository root. Keep the published source batches and existing
final arms unchanged; a replacement needs a reviewed successor collection.

1. Confirm the selected source arm is complete: every test query has a valid
   prediction and per-query trace, and recomputed metrics match `metrics.json`.
   Check that its retrieval manifest pins the intended optimization manifest.
2. Install the approved arm under
   `outputs/paper/assay_transfer_harness/joseph/final/<benchmark>/<task>/<selection>/`.
   Include `manifest.json`, `metrics.json`, `predictions.jsonl`,
   `query_provenance.tsv`, `files.tsv`, `trace_messages.jsonl`, and all
   `runs/<query>/` files. Hash-pin the source batch, optimization manifest, and
   copied files in the arm manifest. Record each query's actual served model and
   provider in `query_provenance.tsv`.
3. Add its relative path, manifest SHA-256, query count, macro-F1, and route
   status to `final/collection.json`. Keep `arms` sorted, recompute the total
   query count, and update `updated_at_utc`. Save the result report, numerical
   TSV, and compact provenance under `outputs/analysis/record_selection/<study-id>/`;
   update this page's counts and links.
4. Check the collection index and fully verify only new arms before rebuilding
   the Git bundle. Existing published arms must retain their collection entries
   and manifest hashes; the archive rebuild reads their files once.

   ```bash
   python - <<'PY'
   import csv, hashlib, json, subprocess
   from pathlib import Path

   final = Path('outputs/paper/assay_transfer_harness/joseph/final')
   sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
   collection = json.loads((final / 'collection.json').read_text())
   prior = json.loads(subprocess.check_output([
       'git', 'show', 'HEAD:outputs/paper/assay_transfer_harness/joseph/final/collection.json'
   ]))
   published = {a['path']: a for a in prior['arms']}
   assert len({a['path'] for a in collection['arms']}) == len(collection['arms'])
   assert [a['path'] for a in collection['arms']] == sorted(a['path'] for a in collection['arms'])
   assert collection['n_queries'] == sum(a['n_queries'] for a in collection['arms'])
   assert published.keys() <= {a['path'] for a in collection['arms']}
   for arm in collection['arms']:
       path = final / arm['path']
       assert sha(path / 'manifest.json') == arm['manifest_sha256']
       if arm['path'] in published:
           assert arm == published[arm['path']]
           continue
       manifest = json.loads((path / 'manifest.json').read_text())
       metrics = json.loads((path / 'metrics.json').read_text())
       assert manifest['status'] == 'complete'
       assert manifest['n_queries'] == arm['n_queries']
       assert metrics['n_total'] == metrics['n_successful'] == arm['n_queries']
       assert metrics['macro_f1'] == arm['macro_f1']
       assert sum(1 for _ in path.glob('runs/*/trace_messages.jsonl')) == arm['n_queries']
       for name, digest in manifest['files'].items():
           assert sha(path / name) == digest
       with (path / 'files.tsv').open() as handle:
           for row in csv.DictReader(handle, delimiter='\t'):
               assert sha(final / row['final_file']) == row['sha256']
   print(len(collection['arms']), 'arms;', collection['n_queries'], 'queries')
   PY
   ```

5. Rebuild and check the tracked archive. Raw files under `final/` are ignored
   by Git; the compressed parts carry their complete copy, including traces.

   ```bash
   bash outputs/paper/assay_transfer_harness/joseph/final/git_bundle/build.sh
   cd outputs/paper/assay_transfer_harness/joseph/final/git_bundle
   sha256sum -c SHA256SUMS
   cat payload.tar.zst.part-* | zstd -tq
   cd ../../../../../..
   ```

6. Stage the updated `finalize.md`, `final/collection.json`,
   `final/git_bundle/`, and the new report, TSV, and provenance. The analysis
   directory is ignored, so stage those three files with `git add -f`.
   Replace `STUDY_ID` with the new study directory name:

   ```bash
   git add finalize.md outputs/paper/assay_transfer_harness/joseph/final/collection.json \
     outputs/paper/assay_transfer_harness/joseph/final/git_bundle/
   git add -f outputs/analysis/record_selection/STUDY_ID/{report.md,results.tsv,provenance.json}
   git diff --cached --check
   git diff --cached --name-only
   git commit -m "Add approved Joseph final arms and traces"
   git -c fetch.ifMissing=false -c pack.window=0 \
     -c core.sshCommand='ssh -o BatchMode=yes -o ConnectTimeout=5 -p 443' \
     push --no-thin ssh://git@ssh.github.com/DragonDescentZerotsu/TxAgent.git HEAD:refs/heads/joseph
   git -c core.sshCommand='ssh -o BatchMode=yes -o ConnectTimeout=5 -p 443' \
     ls-remote ssh://git@ssh.github.com/DragonDescentZerotsu/TxAgent.git refs/heads/joseph
   ```

   Confirm the remote hash matches `git rev-parse HEAD`. The raw arm directories
   do not need `git add -f`; they are inside the tracked compressed bundle.
