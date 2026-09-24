# Frozen BBB indirect ablation inputs and results

Use the [maintained reproduction guide](../../../tools/chembl_tool/tasks/bbb_martins/INDIRECT_ABLATION.md).

- `inputs.json.gz` and its checksum manifest: 406 queries, fixed Direct cards,
  cached priors, and the frozen union of saved indirect selections; no API keys.
- `provider.example.json`: OpenRouter settings using a local environment variable.
- `comparison.json` / `predictions.csv`: complete test-tuned results, including controls.
- `verification.json`: hashes of 2,436 verified local outputs.
- `refactor_parity.json`: unchanged messages and selected IDs for 1,624 one-pass inputs.
- `export_inputs.py`: rebuild the bundle from the original local baseline selections.

Historical two-stage ablations are reported, but not exposed as additional maintained preparation modes.
