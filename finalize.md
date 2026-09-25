# Final Joseph full-test collection

The [final collection](outputs/paper/assay_transfer_harness/joseph/final/collection.json)
contains 60 complete test arms and 19,843 successful arm-query results. It includes
Gold BBB, Oral Bioavailability, AMES, Carcinogens, DILI, and Skin Reaction
direct and direct+indirect arms; TDC Oral Bioavailability and Skin Reaction
direct and direct+indirect arms; and TDC-v2 AMES, BBB, Carcinogens, and DILI
direct and direct+indirect arms. The [Luna directory](outputs/paper/assay_transfer_harness/joseph/final/luna/README.md)
indexes frozen Gold-v1 and TDC-v2 Skin, BBB, and Oral Bioavailability validation
and test arms, plus DILI, Gold-v1 Carcinogens, and TDC-v2 AMES test arms, with
their traces and source provenance.
Each arm includes predictions, metrics, per-query run artifacts and traces,
a combined trace, a query-level model/provider ledger, and
hash-pinned source and copied-file manifests. The original batches remain at
their source paths.

The [original results report](outputs/analysis/record_selection/joseph_final_gold_bbb_oral_carcinogens_tdc_oral_20260924_v1/report.md)
covers the first seven arms. The [TDC Skin report](outputs/analysis/record_selection/joseph_final_tdc_skin_upstream_v2_test_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_tdc_skin_upstream_v2_test_20260924_v1/results.tsv)
cover the selected upstream-v2 pair: direct macro-F1 0.634097 and
direct+indirect macro-F1 0.666875, each on all 82 TDC-v1 test queries. The
pair was retained after test comparisons, and later DeepSeek prompt and model
iterations reused the same cohort. These are historical descriptive scores,
not an untouched TDC-v2 holdout estimate. Gold Carcinogens
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

The selected [DeepSeek V4 Carcinogens arm](outputs/paper/assay_transfer_harness/joseph/final/deepseek/test/tdc_v2/carcinogens/direct_plus_indirect/manifest.json)
uses the stronger direct-trust Final Full Flat prompt, ten frozen direct cards,
and fifty validation-selected indirect records. Its 56-query TDC-v2 test scored
0.790611 macro-F1. The [selection report](outputs/analysis/record_selection/tdc_v2_carcinogens_final_full_flat_strong_direct_50_test_20260925_v1/report.md)
and [TSV](outputs/analysis/record_selection/tdc_v2_carcinogens_final_full_flat_strong_direct_50_test_20260925_v1/results.tsv)
compare prompt and model variants; the prompt and model choice was made after
viewing test results. The DeepSeek final directory preserves the chosen prompt,
parameters, per-query traces, and served-provider provenance.

The [DeepSeek Carcinogens direct arm](outputs/paper/assay_transfer_harness/joseph/final/deepseek/test/tdc_v2/carcinogens/direct/manifest.json) adds the validation-selected exact-score upstream-v3 direct panel: 0.817392 macro-F1 on all 56 test queries. The separate [Gold DILI `sr=0` control](outputs/paper/assay_transfer_harness/joseph/final/deepseek/test/gold_v1/dili/direct_plus_indirect_sr000_v3_control/manifest.json) scored 0.680085 on all 402 test queries. It was evaluated after viewing test ablations; `canonical_dili` now points to it because the user chose the better completed test result. The prior validation-selected 0.659233 arm remains preserved. The further diversity ablation at 0.693906 was not promoted. Their [report](outputs/analysis/record_selection/joseph_final_deepseek_carc_direct_dili_sr0_20260925_v1/report.md), [TSV](outputs/analysis/record_selection/joseph_final_deepseek_carc_direct_dili_sr0_20260925_v1/results.tsv), and provenance pin the source runs, prompts, frozen selections, weights, routes, and traces. The upstream-v3 and polarized DILI prompt snapshots and selected TDC DILI optimization snapshot are in `final/deepseek/`.

The [Gold-v1 best-test full-control promotion](outputs/analysis/record_selection/gold_v1_best_test_full_control_final_promotion_20260925_v1/report.md) adds complete BBB, Carcinogens, and Skin mixed successor arms and selects the already packaged DILI `sr=0` arm. `collection.json` records all six selected DeepSeek Gold-v1 mixed paths under `selected_gold_v1_deepseek_test_best`; Oral and AMES retain their earlier FINAL arms. This is a choice after viewing test results. The [numeric TSV](outputs/analysis/record_selection/gold_v1_best_test_full_control_final_promotion_20260925_v1/results.tsv) records the previous and selected scores. The separate [ablation and scaling tables](outputs/paper/assay_transfer_harness/joseph/final/ablations/README.md) include 10, 25, 50, 75, and 100 indirect records.

The [12-row DeepSeek test table](outputs/paper/assay_transfer_harness/joseph/final/test_macro_f1_12_task_benchmarks_deepseek_selected_20260925_v1.tsv) lists the current complete selected direct and direct+indirect FINAL arms for each Gold-v1 (Starling) and TDC task. It includes the selected Gold-v1 mixed successors and the completed newer TDC-v2 AMES, DILI, and Carcinogens arms; each row pins both arm paths and manifest hashes.

The matching [12-row Luna test table](outputs/paper/assay_transfer_harness/joseph/final/test_macro_f1_12_task_benchmarks_luna_selected_20260925_v1.tsv) includes the new Gold-v1 Carcinogens direct arm at 0.482328 macro-F1 on 469 queries and the TDC-v2 Carcinogens direct arm at 0.841539 on 56 queries. Each row pins its selected FINAL arms and manifest hashes.

The [TDC-v2 BBB report](outputs/analysis/record_selection/joseph_final_tdc_v2_bbb_filtered_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_tdc_v2_bbb_filtered_20260924_v1/results.tsv)
cover the 406-query direct and filtered direct+indirect arms: macro-F1 0.865673
and 0.880814, respectively. The indirect selector removed prediction-only
records before its joint 50-record selection.

The earlier [TDC-v2 DILI report](outputs/analysis/record_selection/joseph_final_tdc_v2_dili_20260924_v1/report.md)
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
Those arms remain in the collection as historical results.

The current DILI results are the eight model-specific arms indexed under
`canonical_dili` in `collection.json`. The [joint DILI report](outputs/analysis/record_selection/dili_v2_partitioned_joint_gold_tdc_20260924_v1/report.md),
[validation table](outputs/analysis/record_selection/dili_v2_partitioned_joint_gold_tdc_20260924_v1/validation.tsv),
and [test table](outputs/analysis/record_selection/dili_v2_partitioned_joint_gold_tdc_20260924_v1/test.tsv)
record the frozen validation choices and complete test results:

| Benchmark | Model | Direct macro-F1 | Direct+indirect macro-F1 |
|---|---|---:|---:|
| Gold-v1 | GPT-6-Luna | 0.621456 | 0.665266 |
| Gold-v1 | DeepSeek-V4-Flash-0731 | 0.639755 | 0.659233 |
| TDC-v2 | GPT-6-Luna | 0.790850 | 0.801545 |
| TDC-v2 | DeepSeek-V4-Flash-0731 | 0.819609 | 0.821346 |

Each new arm preserves its 402 Gold or 96 TDC per-query runs, combined trace,
served provider ledger, selection manifest, and hash-pinned source batch. The
partitioned V2 DILI semantic snapshot remains an unselected experimental input.

The [AMES collection report](outputs/analysis/record_selection/ames_final_collection_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/ames_final_collection_20260924_v1/results.tsv)
cover four new full-test arms. Gold-v1 AMES direct and direct+indirect macro-F1
are 0.708578 and 0.774211 on 274 queries each; TDC-v2 AMES scores are
0.782289 and 0.799350 on 1,457 queries each. TDC mixed query 1158 used a
fresh, identical rendered prompt with a lower token ceiling, recorded in its
arm provenance. The source response sets are complete across preserved runs;
some historical individual matrices remain partial.

The [Luna Skin report](outputs/analysis/record_selection/luna_skin_final_publication_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/luna_skin_final_publication_20260924_v1/results.tsv)
cover the frozen Gold-v1 and TDC-v2 indirect validation winners and their
complete test runs. Their test macro-F1 scores are 0.564175 on 241 Gold-v1
queries and 0.605770 on 82 TDC-v2 queries. The new `final/luna/` directory
preserves the full validation profile traces and both prompt bundles. The two
original test winners remain in the full-test collection index.

The [Luna Gold-v1 Skin direct arm](outputs/paper/assay_transfer_harness/joseph/final/luna/test/gold_v1/skin_reaction/direct/manifest.json)
uses the direct profile frozen on 100 `valid_small` queries. It scored 0.559252
macro-F1 on all 241 test queries. Its exact upstream-v2 prompt, selection,
Azure route, and per-query traces are preserved. The [publication report](outputs/analysis/record_selection/luna_gold_skin_direct_final_20260925_v1/report.md)
and [TSV](outputs/analysis/record_selection/luna_gold_skin_direct_final_20260925_v1/results.tsv)
record the result and validation provenance.

The [Luna TDC Skin direct arm](outputs/paper/assay_transfer_harness/joseph/final/luna/test/tdc_v2/skin_reaction/direct/manifest.json)
adds a frozen 82-query TDC-v1 direct test run to the TDC-v2 view. The v1 and v2
test files contain the same molecule structures, labels, and condition fields in
different order; the arm retains its original v1 retrieval, query prior, and
prompt. It scored 0.700912 macro-F1. The [equivalence report](outputs/analysis/record_selection/luna_skin_tdc_v1_v2_direct_final_20260925_v1/report.md)
and [TSV](outputs/analysis/record_selection/luna_skin_tdc_v1_v2_direct_final_20260925_v1/results.tsv)
document this as an equivalent-cohort replay rather than a native v2 rerun.

The [Luna TDC-v2 Skin mixed successor](outputs/paper/assay_transfer_harness/joseph/final/luna/test/tdc_v2/skin_reaction/direct_plus_indirect_valid_selected_v2/manifest.json)
uses the best of eight full-validation mixed profiles with the direct panel
frozen. Its validation macro-F1 was 0.702502 on 40 queries; its complete 82-query
test macro-F1 was 0.611005. Five new validation profile arms, the selected test
arm, exact prompt and parameter snapshots, and recovery traces are in `final/luna/`.
The previous 0.605770 Skin mixed arm remains for audit. The [comparison report](outputs/analysis/record_selection/luna_skin_tdc_v2_indirect_final_20260925_v1/report.md)
and [TSV](outputs/analysis/record_selection/luna_skin_tdc_v2_indirect_final_20260925_v1/results.tsv)
show the validation decision and all three tested profiles.

The [Luna BBB/Oral report](outputs/analysis/record_selection/luna_bbb_oral_20260924_v1/report.md)
and [comparison TSV](outputs/analysis/record_selection/luna_bbb_oral_20260924_v1/comparisons.tsv)
cover five direct validation profiles and five indirect validation profiles per
benchmark/task panel. The frozen direct and direct+indirect test macro-F1 scores
are 0.666929 and 0.722610 for Gold BBB, 0.760093 and 0.856127 for Gold Oral,
0.828823 and 0.846330 for TDC-v2 BBB, and 0.718372 and 0.774023 for TDC-v2
Oral. All eight test arms use `final_full_flat` and GPT-6-Luna with high reasoning;
the Luna directory preserves all 40 validation arms and eight test arms.

The [Luna AMES/Carcinogens report](outputs/analysis/record_selection/luna_ames_gold_tdc_carcinogens_20260925_v1/report.md)
and [comparison TSV](outputs/analysis/record_selection/luna_ames_gold_tdc_carcinogens_20260925_v1/comparisons.tsv)
cover five direct and five indirect full-validation profiles for Gold-v1 AMES
and TDC-v2 Carcinogens. Frozen direct and direct+indirect test macro-F1 scores
are 0.667730 and 0.706003 for Gold AMES, and 0.841539 and 0.737500 for
TDC-v2 Carcinogens. Their four test arms use `final_full_flat` and GPT-6-Luna;
the Luna directory also preserves all 20 validation arms.

The [Luna TDC-v2 Carcinogens positive-gate arm](outputs/paper/assay_transfer_harness/joseph/final/luna/test/tdc_v2/carcinogens/direct_plus_indirect_positive_gate_v1/manifest.json)
preserves the validation-frozen direct and indirect selection, the exact prompt
bundle used by the test, provider provenance, and all 56 per-query traces. It
scored 0.878261 macro-F1. The [comparison report](outputs/analysis/record_selection/luna_tdc_v2_carcinogens_positive_gate_test_20260925_v1/report.md)
and [TSV](outputs/analysis/record_selection/luna_tdc_v2_carcinogens_positive_gate_test_20260925_v1/results.tsv)
also record the second tested profile. The prompt was revised after viewing test
results, so this is an exploratory successor arm rather than a validation-selected
prompt result.

The [Luna TDC-v2 AMES direct arm](outputs/paper/assay_transfer_harness/joseph/final/luna/test/tdc_v2/ames/direct/manifest.json)
and [direct+indirect arm](outputs/paper/assay_transfer_harness/joseph/final/luna/test/tdc_v2/ames/direct_plus_indirect/manifest.json)
use the same upstream-v3 prompt and the v3 parent-100 exact-score cache for
their 1,457-query test. They scored 0.787868 and 0.811343 macro-F1, above the
existing DeepSeek FINAL values of 0.782289 and 0.799350. The mixed profile was
chosen on a 100-query validation cohort; the full-validation mixed run was
stopped and did not inform selection. The [comparison report](outputs/analysis/record_selection/luna_gold_carcinogens_tdc_ames_20260925_v1/final_publication_v1/report.md),
[test TSV](outputs/analysis/record_selection/luna_gold_carcinogens_tdc_ames_20260925_v1/final_publication_v1/results.tsv),
and [validation TSV](outputs/analysis/record_selection/luna_gold_carcinogens_tdc_ames_20260925_v1/final_publication_v1/validation.tsv)
also record the [Gold-v1 Carcinogens Luna direct](outputs/paper/assay_transfer_harness/joseph/final/luna/test/gold_v1/carcinogens/direct/manifest.json)
and [direct+indirect](outputs/paper/assay_transfer_harness/joseph/final/luna/test/gold_v1/carcinogens/direct_plus_indirect/manifest.json)
arms. Their 469-query test macro-F1 values are 0.482328 and 0.623776. They
were added as model-specific results; both trail the DeepSeek FINAL direct
score of 0.532669 and the selected mixed score of 0.658598.

The complete collection, including its traces, is preserved in Git as
[compressed parts](outputs/paper/assay_transfer_harness/joseph/final/git_bundle/README.md).

## Updating the final files

Work from the repository root. Keep the published source batches and existing
final arms unchanged; a replacement needs a reviewed successor collection.

1. Confirm the selected source arm is complete: every test query has a valid
   prediction and per-query trace, and recomputed metrics match `metrics.json`.
   Check that its retrieval manifest pins the intended optimization manifest.
2. Install the approved arm under
   `outputs/paper/assay_transfer_harness/joseph/final/<model>/test/<benchmark>/<task>/<selection>/`.
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
4. Check the collection index and fully verify only new arms. Existing published
   arms must retain their collection entries and manifest hashes. Leave the Git
   bundle unchanged until a push is requested.

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

5. Only when the user wants to push, rebuild and check the tracked archive once
   after all intended additions. Do not recompress after each arm or intermediate
   edit.
   Raw files under `final/` are ignored
   by Git; the compressed parts carry their complete copy, including traces.

   ```bash
   bash outputs/paper/assay_transfer_harness/joseph/final/git_bundle/build.sh
   cd outputs/paper/assay_transfer_harness/joseph/final/git_bundle
   sha256sum -c SHA256SUMS
   cat payload.tar.zst.part-* | zstd -tq
   cd ../../../../../..
   ```

6. When pushing, stage the updated `finalize.md`, `final/collection.json`,
   `final/git_bundle/`, and the new report, result and validation TSVs, and
   provenance. The analysis directory is ignored, so stage those files with
   `git add -f`.
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
