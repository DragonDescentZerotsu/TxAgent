# V10 unit reconciliation

AMES, DILI, and Carcinogens use one lossless two-pass unit-reconciliation
protocol implemented by `unit_reconciliation.py`. The output is an exact unit
map; it does not modify Stage 1 or implement Carcinogens Stage 2.

The frozen input universe is the union of every nonempty raw `unit_text`, every
deterministic `measurement_resolution_exact_unit`, and every unit emitted by an
`ok` Stage-1 LLM measurement assignment. Pass A sorts this universe
deterministically and processes groups of 50. Pass B deduplicates the Pass-A
canonical units, weights them with their membership counts, shuffles them from
the Pass-A manifest hash, and processes groups of 50 again. Publication composes
the two maps and requires exact coverage of the original universe.

Both passes preserve magnitude and scientific notation. Typography aliases such
as `uM` and `µM` may be reconciled, but `nM`, `µM`, `mM`, and
`10^-6 mol/L` remain distinct. Percent, control-relative, inhibition, ratio,
fold, count, and denominator qualifiers are preserved. Every published rule is
an `action: map` rule with `scale: "1"`. An uncertain unit remains its own
canonical unit; no unit is excluded.

Each partition uses this exact retry sequence with `reasoning_effort=high`:

1. local `deepseek-ai/DeepSeek-V4-Flash-0731`, 32,768 output tokens;
2. the same local model with validation feedback, 65,536 output tokens;
3. `gpt-5.4-mini` through `OPENAI_API_KEY_ONE`, 65,536 output tokens.

If all three attempts fail structurally, the entire partition is identity-mapped
with `identity_after_retry_exhaustion`. Partitions are never recursively split.
The append-only terminal cache, pass mappings, manifests, and final publication
manifest retain request routing, retry outcomes, hashes, and coverage counts.

High-churn work should be staged on node-local storage and only validated final
artifacts published into each task's
`data_processing/canonicalization_v10/` directory. A typical task sequence is:

```bash
python -m data.processing.evidence_library.versions.v10.unit_reconciliation prepare \
  --task ames --run-root /local/$USER/unit-reconciliation/<run-id>
python -m data.processing.evidence_library.versions.v10.unit_reconciliation run \
  --task ames --run-root /local/$USER/unit-reconciliation/<run-id> \
  --pass-name a --workers 512
python -m data.processing.evidence_library.versions.v10.unit_reconciliation run \
  --task ames --run-root /local/$USER/unit-reconciliation/<run-id> \
  --pass-name b --workers 512
python -m data.processing.evidence_library.versions.v10.unit_reconciliation consolidate \
  --task ames --run-root /local/$USER/unit-reconciliation/<run-id>
python -m data.processing.evidence_library.versions.v10.unit_reconciliation validate \
  --task ames --run-root /local/$USER/unit-reconciliation/<run-id>
```
