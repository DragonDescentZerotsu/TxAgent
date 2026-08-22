# ClinTox send_v2 normalized evidence lineage

The active ClinTox data pipeline is rebuilt exclusively from
`/vast/projects/myatskar/lab/shared_docs/clintox_send_v2.tar.gz` (SHA-256
`bf6da36bf2ac347d4763e5c3a234292c7f5c04576cccc4b75f6a4741246ac4ea`).
The immutable source import is `data/starling_data/clintox/send_v2/`.
For repository portability, the exact supplied archive is tracked as verified
90 MB parts under
`artifacts/chembl_tool/tasks/clintox/clintox_send_v2_source/`; the extracted
Parquets are byte-identical local restores rather than duplicate Git blobs.

The release contains 4,846,914 rows across one direct human-clinical source
and six mechanism sources. Every row has a nonempty `SMILES`; 4,816,479
structures normalize successfully and 30,435 fail closed. There is no global
identifier in the delivery, and the pipeline does not synthesize one.

## Scope and ownership

Task-owned canonical evidence stops at Stage 05:

```text
outputs/chembl_tool/tasks/clintox/evidence_library/starling_normalized_v7/
  01_cleaned/
  02_canonicalized/
  03_records/
  04_pair_buckets/
  05_distance_calibration/
```

Benchmark-dependent evidence belongs to the paper lineage:

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/
  evidence/clintox_starling_v7/
    06_records/
    07_molecule_evidence/
    08_neighbor_index/
    09_audits/
```

Stage 06 removes held-out parent rows only from
`human_clinical_toxicity`, the direct label source. Mechanism rows remain
available; query-time retrieval must still use `parent_disjoint`.

The earlier `clintox_base_v1` and `starling_raw_v1` data pipelines are
superseded and are not active inputs. Existing model results are retained as
historical artifacts, but results whose manifests do not pin `clintox_send_v2`
and the current Stage-08 index are incompatible with this lineage.

## Candidate gold split

The direct adapter defines a new task: explicit human clinical toxicity
category versus exact `toxicity_absent`. It is not the TDC/MoleculeNet CT_TOX
clinical-trial-failure task.

The adapter yields 7,628 binary parent molecules: 755 negative and 6,873
positive. The `record_supported_v2` scaffold split contains 6,104 train, 762
valid, and 762 test parents. Valid and test each contain only multi-record
parents; parent-identity and Bemis-Murcko scaffold overlap are both zero.

This split remains `candidate_pending_qa`. The supplied direct schema lacks
`qualifying_conditions`, so that semantic exclusion cannot be reconstructed.
The frozen measurement/unit audit does not promote the benchmark; the
category-stratified direct-source QA gate still applies.

The 2026-08-18 manual review failed that gate. Of 360 frozen source records,
252 passed, 96 failed, and 12 were uncertain: 108/360 (30.0%) were therefore
non-passing. The proposed positive class had 252 pass, 74 fail, and 4
uncertain records. None of the 30 proposed `toxicity_absent` records passed;
22 failed and 8 were uncertain. The largest failure modes were unresolved
combination/comparator attribution (27), formulation or entity mismatch (22),
wrong outcome direction (21), and endpoint-limited absence incorrectly
treated as global safety (17).

The frozen 7,628-parent split is consequently diagnostic only. It must not be
used for paper-facing training, validation, or test. The source must first be
re-adjudicated across all direct records for entity alignment, causal
attribution, outcome direction, and material qualifiers. Failed sampled rows
must not be deleted or replaced, and audit-specific keyword filters are not a
valid correction.

The durable audit contract is:

```text
tools/chembl_tool/tasks/clintox/data_processing/gold_qa_v1/
  REVIEW_PROTOCOL.md
  reextraction_contract_v1.json
  reviewed_rows.tsv
  summary.json
```

`clintox_base build-benchmark` now revalidates the pinned sample and
synchronizes the failed gate into `CANDIDATE_STATUS.json` and the runtime QA
receipt. A corrected gold and dependent held-out evidence view have not been
built because there is not yet a source-wide adjudication policy to build them
from.

## Canonical measurement and unit policy

Shared parsing and conversion live in
`tools/chembl_tool/common/starling/assay_transfer_measurements.py` and
`tools/chembl_tool/common/units.py`. ClinTox-specific selectors, exact aliases,
domains, reference semantics, and fail-closed rules live in:

```text
tools/chembl_tool/tasks/clintox/starling_measurement_semantics.py
tools/chembl_tool/tasks/clintox/starling_reference_semantics.py
tools/chembl_tool/tasks/clintox/data_processing/canonicalization_v7/
tools/chembl_tool/tasks/clintox/data_processing/assay_transfer_measurements_v2/
```

Important policy boundaries:

- A measurement and its unit are normalized only as one recognized atomic
  pair. Unknown, comparator-relative, or malformed forms remain source-visible
  but are ineligible for scalar transfer.
- Shared percent-rate parsing converts `%/wk` to `ratio/wk`; it does not treat
  percent-of-cells, dose-qualified percentages, or duration annotations as a
  rate.
- A point value explicitly outside its source interval fails closed.
- A plain point estimate whose support text reports a censored value is not
  transferred as exact.
- Reviewed log transforms apply only to valid positive scalars. Retrieval
  eligibility and original source fields are invariant.

The frozen v2 transfer policy contains 1,437 reviewed buckets: 1,410 `log10`
and 27 `raw`. It has 54 record-level exclusions: 53 source points outside
their own reported intervals and one implausible literal `6×10^12 nM`
concentration.

## Frozen measurement/unit audit and build receipt

The three manual audits are complete with zero failures:

- main measurement/unit sample: 360/360 pass;
- censored-support sample: 72/72 pass;
- source point/interval consistency sample: 112/112 pass.

The current canonical build contains:

| Stage | Receipt |
|---|---:|
| cleaned rows | 4,846,914 |
| organized records | 4,846,810 |
| retrieval-eligible records | 4,744,068 |
| assay-transfer-eligible records | 3,050,084 |
| assay-transfer-excluded records | 1,796,726 |
| pair buckets | 1,749,120 |
| pairable buckets | 250,531 |
| calibration-valid buckets | 4,301 |

Stage-03 SHA-256 is
`99bbe9013737b475aa69d98565901a805cf867d8a03802e17d7e74970ed155a8`.
Stage-04 records SHA-256 is
`dc99a4500f69ac74912f528a5492bc570633d92973fc1b14233d3c141632cd5a`.
Stage-05 SHA-256 is
`28c9df737acf70f0982d20aa5f8321b9ad6209a42c42cbe12c8915b9fc8f3803`.

The scaffold benchmark view contains 4,584,827 Stage-06 records, 192,605
molecule-family evidence rows, and 136,369 indexed molecule rows across seven
groups. It matches all 1,524 held-out parents in the direct filter scope and
removes 159,181 direct records. The direct group has zero held-out parent
overlap. Mechanism evidence intentionally retains 1,296 held-out parent
identities; runtime `parent_disjoint` filtering is therefore mandatory. The
Stage-08 manifest SHA-256 is
`fd1b60fa44d864accb64677f2132970425f42d11b6f02accd8ff805843e36a6b`.

The assay-transfer rebuild restores its pre-transform tuple from immutable
Stage 02 before applying the frozen policy. This makes Stage 03 idempotent and
keeps retrieval content unchanged; the final retrieval-invariance digest is
`dc12b3d198827f5b86d2c3f014f03ae2befc70ec889d5d67a2b4a8e9cf45eb41`.

## Rebuild commands

These artifacts were built on `dgx018` from
`/vast/projects/myatskar/design-documents/joseph/TxAgent` with system Python
3.12.3:

```bash
/usr/bin/python -m tools.chembl_tool.tasks.clintox.starling_source_artifact_store \
  restore-source

/usr/bin/python -m tools.chembl_tool.tasks.clintox.build_normalized_starling_evidence_library \
  --from-stage clean --through-stage distance --workers 16 \
  --progress-every 100000 --cache-mode off

/usr/bin/python -m tools.chembl_tool.paper_experiments.build_starling_benchmark_indices \
  --indices clintox_starling_v7 --splits scaffold \
  --benchmark-data-root data/processed_starling_record_supported_v2 \
  --benchmark-lineage record_supported_v2 \
  --heldout-filter-mode direct_source_only --workers 32
```

These commands build evidence artifacts only. They do not promote the
candidate or authorize a formal model matrix.
