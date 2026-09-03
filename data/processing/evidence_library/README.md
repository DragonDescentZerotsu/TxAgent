# Evidence-library construction code

Authoritative construction releases live under `versions/`:

- `versions/v7/`: BBB Martins, Bioavailability Ma, and Skin Reaction V7.
- `versions/v8/`: the successor BBB measurement-resolution and record-pruning
  implementation, plus Bioavailability, Skin, and Stage 0-1 Ames policies.

Both releases pin the same construction kernel from `shared/v1/`: normalization
contracts, canonical record construction, pair-bucket policy, deduplication, and
build runtime. It contains no benchmark publishing, retrieval views, artifact
packaging, task-specific source cleaning, or completed review workflows.

Release-wide construction modules live directly in `versions/vN/`; there is no
nested release `shared/` package. Each release owns its task policies, selected
prompts, scientific mappings, pruning behavior, and task-specific source
cleaning. Built data is separate under
`data/evidence_libraries/<task>/<version>/`.

`views.py`, `evidence_library.py`, `heldout_index.py`, `compact_artifacts.py`, and
`assay_catalog.py` are version-independent consumers of completed libraries.
`bucket_informativeness.py` is likewise a downstream analysis, while
`stage_artifact_store.py` packages completed stages. Gold-label dataset and
conditioned-benchmark construction lives under `data/processing/gold_labels/`.
Compatibility construction namespaces are not provided.

`record_informativeness.py` scores accepted V7 Stage-1 rows in
`(dataset, source_id, PMID, canonical_smiles)` groups. It writes a resumable
SQLite request ledger and publishes a row-level Parquet sidecar only after exact
coverage validation.

`relevance_bucket_tournament.py` is the downstream V9 BBB semantic-relevance
pilot. It builds the frozen pair-bucket map without model calls, then runs a
resumable 512-bucket GPT tournament and stops for review:

```bash
python -m data.processing.evidence_library.relevance_bucket_tournament build
python -m data.processing.evidence_library.relevance_bucket_tournament pilot
```

The pilot never starts a full-library run. Its manifest records the actual token
usage, served model, reversal checks, and whether the preregistered gates passed.

After the V1 gate, `relevance_bucket_diagnostic.py` runs the matched V2 N=10/N=5
diagnostic without changing or resubmitting V1 requests:

```bash
python -m data.processing.evidence_library.relevance_bucket_diagnostic build
python -m data.processing.evidence_library.relevance_bucket_diagnostic run
```

`relevance_bucket_pilot.py` runs the subsequent 512-bucket ranking pilot with a
connected degree-4 graph, matched N=5 candidate orders, and single-comparison
adjudication only for order disagreements. It does not expose a full-run command.

```bash
python -m data.processing.evidence_library.relevance_bucket_pilot build
python -m data.processing.evidence_library.relevance_bucket_pilot run
```
