# Prediction output migration

New prediction campaigns use:

```text
outputs/predictions/
  flat/<prompt>/<run-name>/
    <benchmark>/<task>/<split>/<profile>/
      manifest.json
      metrics.json
      runs/<original-query-directory>/
        retrieval.json
        request.json
        single_molecule_reasoning_output.json
        final_reasoning_output.json
    traces/
  query-prior/<prompt>/<run-name>/
    <benchmark>/<task>/<split>/query-prior/
```

The files shown inside a query depend on which stages have completed. Supply
`--run-name` to public flat, matrix and query-prior launchers. The flat namespace
is `flat`; historical runs retain their recorded harness and prompt names.
Manual batch roots and trace roots remain supported. A run name describes the
experiment; manifests still determine scientific compatibility.

## Completed relocation

Relocated **2,644 batch folders and 431 matrix/execution index files**, totaling
**3,075 entries**. The migration renamed frozen containers without rewriting
prediction payloads or manifests. Original query directory names and run UIDs
remain intact. FINAL and compressed publication bundles were excluded.

Checked every moved entry: the old source is absent, its destination exists,
its old locator resolves to the destination, and saved batch run UIDs match.
No verification errors were found. Representative historical manifest and
retrieval reads also passed across 15 harness namespaces. These were metadata
and representative-read checks, rather than a full payload rehash.

The operational [relocation map](../../outputs/predictions/relocations.json)
lets maintained readers follow old references without symlinks or rewriting
historical metadata. Applications outside these readers must use the new paths
or this map. Applied receipts:
[initial pass](../../outputs/predictions/relocation_report.json) and
[second pass](../../outputs/predictions/relocation_remaining.json).

## Manual decisions remaining

**233 entries stay at their original paths**: 195 have destination collisions,
and 38 lack one or more identity fields. Names alone were insufficient to
assign missing scientific identities. The
[complete remaining inventory](../../outputs/predictions/relocation_final_pass.json)
contains exact source paths, known identity fields, and reasons.

| Campaign | Entries | Decision needed |
| --- | ---: | --- |
| `ames_direct_v3_valid100_20260924_v1__ames_tdc_v2_v3_valid_balanced100_20260924_r2` | 8 | Destination collision; choose a distinct semantic run name |
| `ames_direct_v3_valid100_20260924_v1__ames_tdc_v2_v3_valid_balanced100_20260924_r3` | 8 | Destination collision; choose a distinct semantic run name |
| `dili_geomean_tied12_gold_valid_openrouter_auto_recovery_20260924_v1` | 6 | Missing: prompt |
| `dili_tdc_v2_geomean_parent_disjoint_valid_20260924_v1__k10_m0_20260924_072714` | 18 | Destination collision; choose a distinct semantic run name |
| `direct_context_grid_gold6_valid_small_20260922` | 1 | Missing: harness, prompt, task, benchmark, subset |
| `direct_context_grid_gold6_valid_small_20260922__direct12_gold6_valid_small_20260922` | 32 | Destination collision; choose a distinct semantic run name |
| `gold_direct_optimized_full_flat_test_dual_scores_v2` | 1 | Destination collision; choose a distinct semantic run name |
| `gold_v1_carcinogens_indirect_scaling_parent100_full_test_20260925_v1_recovery_v1__k10_m0_20260925_094157` | 2 | Destination collision; choose a distinct semantic run name |
| `legacy` | 1 | Missing: harness, prompt, task, benchmark, subset |
| `luna_bbb_gold_direct_top5_full_valid_20260924_v3__k10_m0_20260924_233743` | 5 | Destination collision; choose a distinct semantic run name |
| `safety_direct_full_test_upstream_v2_20260923_v1` | 1 | Missing: harness, prompt, task, benchmark, subset |
| `six_task_gold_v1_test_upstream_openrouter_20260923_r1` | 4 | Missing: harness |
| `six_task_valid_small_query_priors_v2_20260921_1152` | 6 | Missing: harness, prompt, benchmark |
| `skin_direct_gold_tdc_upstream_v3_valid_20260924_v1` | 54 | Destination collision; choose a distinct semantic run name |
| `skin_indirect_tied_record_upstream_v3_test_20260924_v1` | 1 | Missing: harness, prompt, task, benchmark, subset |
| `skin_indirect_tied_record_upstream_v3_test_20260924_v1__k10_m0_20260924_055201` | 11 | Destination collision; choose a distinct semantic run name |
| `tdc_deepseek_0731_indirect_scaling_20260925_v1` | 34 | Destination collision; choose a distinct semantic run name |
| `tdc_deepseek_0731_indirect_scaling_20260925_v1` | 1 | Missing: prompt |
| `tdc_full_flat_v5` | 2 | Missing: prompt |
| `tdc_full_flat_v5__k10_w25_m0_20260920_1104` | 6 | Destination collision; choose a distinct semantic run name |
| `tdc_full_flat_v5__k10_w25_m0_20260920_1110` | 8 | Destination collision; choose a distinct semantic run name |
| `tdc_full_flat_v5__k10_w25_m0_20260920_1351` | 8 | Destination collision; choose a distinct semantic run name |
| `tdc_v1_deepseek_v4_flash_0731_high_20260920_1055` | 2 | Missing: task, benchmark, subset |
| `tdc_v1_test_upstream_deepseek_openrouter_20260923_r1` | 3 | Missing: task, benchmark, subset |
| `tdc_v1_valid_skin_upstream_v1_20260922_r3` | 1 | Missing: task, benchmark, subset |
| `tdc_v1_valid_three_task_upstream_v1_20260922` | 1 | Missing: task, benchmark, subset |
| `tdc_v1_valid_three_task_upstream_v1_20260922_r2` | 1 | Missing: task, benchmark, subset |
| `tdc_v2_bbb_direct_upstream_v3_canary_20260924_v2` | 1 | Missing: prompt |
| `tdc_v2_original_high_20260924_r1` | 6 | Missing: task, benchmark, subset |

Provide reviewed overrides in a JSON object keyed by an exact source path
(relative to the repository, or absolute). Only include fields being changed:

```json
{
  "outputs/paper/assay_transfer_harness/joseph/<exact-old-batch>": {
    "prompt": "<reviewed-prompt-name>",
    "run_name": "<distinct-semantic-run-name>"
  }
}
```

Supported identity fields are `harness`, `prompt`, `run_name`, `benchmark`,
`task`, `subset`, and `profile`. A collision needs a distinct run name; a
missing identity needs the verified value. Generate a fresh dry-run report
before applying reviewed overrides:

```bash
python -m predict.harnesses.branches.migrate_manifests \
  --relocate-runs --classifications /tmp/reviewed-identities.json \
  --report /tmp/reviewed-relocations.json
python -m predict.harnesses.branches.migrate_manifests \
  --relocate-runs --report /tmp/reviewed-relocations.json --apply
```

The apply step uses the saved plan, rejects changed source headers, and defers
active writers. Unknown entries remain in place. No inference or rescoring is
part of this migration.
