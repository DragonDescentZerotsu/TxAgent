# TDC benchmark import

This tree pins only the TDC datasets whose endpoints match a TxAgent gold-label
task: BBB, oral bioavailability, skin reaction, Ames, DILI, and carcinogenicity.
It intentionally excludes ClinTox and unrelated TDC tables.

`python -m data.processing.gold_labels.build_tdc_benchmark` publishes normalized,
parent- and scaffold-disjoint splits under
`data/gold_labels/TDC/<Task>/v1/scaffold/`. The release is explicit and never
changes a task's Gold-v1 `CURRENT` pointer. Source manifests pin every imported
file and identify the upstream commit. The Ames CSVs are byte-identical copies
of `therapeutic-tuning/data/raw/original/AMES/{train,val,test}.csv` (with
`val.csv` published locally as `valid.csv`).

BBB and oral bioavailability have an opt-in L1-only adapter:

```bash
python -m predict.harnesses.progressive \
  --harness-version tdc-mixed-progressive-v1 \
  --tasks bbb_martins bioavailability_ma \
  --evaluation-subset valid \
  --prepare-only
```

It reads the immutable `tdc_mixed_l1_v1` cache: one Morgan-ranked top-10 union
of Gold-v1 training cards and TDC training cards, with the label source visible
in every card. It does not reuse later-level caches from the Gold-v1 query set.

Skin can use the existing direct harness with the explicit TDC split as its
input. In an otherwise fully configured direct-harness invocation, use:

```bash
--input-jsonl data/gold_labels/TDC/Skin_Reaction/v1/scaffold/valid.jsonl
```

Ames, DILI, and carcinogenicity are published as data lineages only; this import
does not introduce a prediction harness for them.
