# Joseph flat v2

This bundle is the default cache-matched flat harness for BBB and oral
bioavailability. It preserves the progressive selector's fixed Morgan,
assay-transfer, and joint record assignments, then presents them together as
molecule-grouped source-semantic cards.

Every nonempty source-contract-approved scientific field is either mapped to a
core card field or retained under `experimental_details`. Raw `extra_details`
is visible. Evidence-molecule names, duplicate source structures, source row
identifiers, and extraction metadata stay internal. Each physical record has a
stable short card ID.

Group comparison tools are disabled. The separate single-molecule branch still
uses `molecule_properties`. Group outputs cite evidence through `claims` and
card IDs rather than reproducing evidence rows.

Run one condition:

```bash
python -m predict.harnesses.branches --organization flat \
  --task bioavailability_ma \
  --reranking assay-transfer \
  --record_pool assay-transfer-trained
```

Use `--harness-version joseph-flat-v1` or `tianang-flat-v1` for the preserved
historical surfaces. `--prepare-only` verifies and materializes retrieval
without issuing model or tool calls.
