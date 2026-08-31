# Conditioned Benchmark

This is the only active four-task evaluation dataset:

```text
data/conditioned_benchmark/<Task>/scaffold/
```

The public name is **Conditioned Benchmark**. The machine-readable contract is
`conditioned_benchmark.v1`; task-specific historical version strings are not
part of active paths, runner flags, or result labels.

## Tasks and sizes

| Task | Train | Valid | Test | Target |
|---|---:|---:|---:|---|
| BBB_Martins | 3,053 | 397 | 393 | experimentally meaningful systemic CNS access |
| Bioavailability_Ma | 1,958 | 262 | 269 | oral bioavailability under the reported condition |
| ClinTox | 1,144 | 142 | 142 | clinical-trial toxicity failure versus approved comparator |
| Skin_Reaction | 1,997 | 246 | 248 | skin sensitization/contact allergy |

Every split row has the same condition-aware schema:

```text
drug, Y, condition_group, condition_scope, molecule_identity_key,
bemis_murcko_scaffold, benchmark_row_id
```

ClinTox has no accepted external condition groups. Its rows use
`no_reported_external_condition`; the prompt renderer omits the condition
sentence for this value. The other tasks use the same null-group behavior.

## Split and leakage contract

- The evaluation unit is one molecule-condition row.
- Parent identity and Bemis-Murcko scaffold do not cross train/valid/test.
- Retrieval removes held-out direct-outcome rows and applies the configured
  query-level scaffold-disjoint policy.
- A null condition is a real benchmark unit, not missing schema.

## Provenance and result reuse

`data/conditioned_benchmark/migration_receipt.json` records the former source
artifact, hashes, row counts, label counts, and ordered `(drug, Y)` checks.
BBB, Bioavailability, and Skin split files are byte-identical to the previously
evaluated selected conditioned cohorts. ClinTox adds only null-condition
metadata; its ordered molecules, labels, and splits are unchanged. Therefore
existing predictions are reusable when their manifest input hash matches the
hash recorded in the receipt; no score is copied by name alone.

The old molecule-only and selected-vN names are source-lineage provenance, not
alternative active gold datasets. New code must import paths from
`tools.chembl_tool.common.starling.conditioned_benchmark` and must not hard-code
those historical paths.

## Entrypoints

```bash
python -m tools.chembl_tool.common.starling.publish_conditioned_benchmark
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve
```

Task-specific source review and voting code remains responsible for scientific
label construction. The publisher only standardizes the final interface and
does not revote or change task labels.
