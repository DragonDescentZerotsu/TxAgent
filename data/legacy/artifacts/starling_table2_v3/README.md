# Starling Table 2 release snapshot

This directory contains the compact molecule-level CSV files used to reproduce
Table 2 of *Self-Driving Datasets: From 20 Million Papers to Nuanced Biomedical
Knowledge at Scale*, arXiv `2605.07022v3`.

## Layout

```text
raw/
  {bbb,oral_ba,ld50}_tdc_plus_lit_train.csv
  {bbb,oral_ba,ld50}_tdc_test.csv
  {bbb,oral_ba,ld50}_lit_train.csv
  {bbb,oral_ba,ld50}_lit_test.csv
source_manifest.json
```

The files were copied byte-for-byte from the Mac-side release snapshot at:

```text
/Users/kirianozan/Documents/Study/Penn/projects/Starling/tdc_starling_data/
```

on 2026-08-12. `source_manifest.json` records SHA-256 receipts.

## Label contract

- BBB is classification. Literature `label` is the fraction of extractions
  labeled positive; the hard evaluation target is `label >= 0.5`.
- Oral bioavailability is classification. Literature `label` preserves the
  molecule-level bioavailability target on `[0, 1]`; the hard evaluation target
  is `label >= 0.20`.
- LD50 is regression and remains continuous.
- `*_tdc_plus_lit_train.csv` contains both the TDC-only training subset
  (`source=tdc`) and the augmented training pool (`source in {tdc,lit}`).

The default reproduction preserves the released soft literature targets during
classification training because the upstream MiniMol head uses binary
cross-entropy, which accepts targets in `[0, 1]`. The `hard` label policy is
available as an explicit ablation and is never silently substituted.

On 2026-08-12, the data author clarified that these fractions should be passed
directly to `BCEWithLogitsLoss` and that collapsed molecule rows should be
upweighted by their `n_extractions`, suggesting `sqrt(n_extractions)` as a
reasonable scale. The runner exposes this explicitly as `--extraction-weight
sqrt`; missing TDC counts receive weight 1. This changes only the training and
inner-validation loss, not the raw CSVs or hard evaluation targets. A separate
`--evaluation-extraction-weight sqrt` option applies the same row weights only
to classification AUROC. It is an explicit sensitivity analysis rather than the
default molecule-level evaluation policy, and was not specified by the author's
clarification. Even linear `n_extractions` weighting repeats each molecule's
thresholded hard label; it is not equivalent to reconstructing positive and
negative extraction-level test records from the aggregate fraction.

MiniMol and the scaffold splitter both require a sanitized RDKit molecule. One
BBB literature-training SMILES (`C[CH3+]OC(=O)c1ccc(N)cc1`) fails that contract;
the runner excludes it from the modeled training pool and records the molecule,
reason, source, label, and surface in `metrics.json`. The raw release file and
its hash remain unchanged, and neither BBB test set contains an invalid SMILES.

## Reproduction

The maintained entrypoints and complete protocol are indexed in
[`baselines/minimol/README.md`](../../baselines/minimol/README.md). Results,
diagnostics, commands, and artifact receipts are kept in
[`STARLING_TABLE2_REPRODUCTION.md`](../../baselines/minimol/STARLING_TABLE2_REPRODUCTION.md)
so this data directory only documents the immutable source snapshot and its
label/featurization contract.
