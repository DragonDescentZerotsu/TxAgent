# Skin source review — 2026-09-14

Current retrieval uses the identity-rebound Stage-03 snapshot and source-purity v7.
This is a focused primary-source review of implicated records, not a full library
review. Benchmark votes, labels, splits, prompts, tools and query priors are unchanged.

## Current repairs

`rebindings.json` pins the complete original row, corrected fields, primary-document
hash/locator, molecular identity and mass checks. Seven records are correctly rebound:

| Acquisition keys (`sensitization_aop:`) | Correction |
|---|---|
| 18020, 18027 | PMID18844695 Figure 1 identifies DO3 as Disperse Orange 3, not the bound anthraquinone. Restore its azo-dye structure. Preserve the responding-clone count and concurrent polyclonal responses; neither is a clinical sensitization incidence. |
| 38405 | PMID35498820 INF is infigratinib, not ifosfamide. Correct name, structure and support text. This is the same Table 6 prediction already represented by 38411: retain the repaired source row, but use 38411 as its sole retrieval representative. |
| 38407–38410 | Restore infigratinib M4/M6/M8/M9 from supplementary Figures S4/S6/S8/S9. Their calculated nominal [M+H]+ matches 510/542/626/608. These are proposed LC-MS/MS structures and DEREK predictions, not experimentally measured sensitization outcomes; route to L2. |
| 20616–20618 | Retain the earlier L3→L2 correction for the overall EMIM in vivo/in vitro negative result (NTP TOX103, p.6). |

M16/M17 (38402/38403) remain quarantined. Figures 9/10 were obtained and read, but
precursor/product linkage and oxygenation depictions have not yielded a unique,
mass-consistent whole-molecule assignment. Do not replace either conjugate with the
parent drug. Their source rows and earlier provenance remain intact.

The primary supplement and Figures 9/10 came from Europe PMC's supplementaryFiles
archive for PMC9052791. URLs and hashes are in `documents/manifest.json`. The review
uses original documents and a ChEBI identity cross-check; it does not infer repairs
from benchmark labels or observed performance.

The current family ledger is `decisions_rebound.json`; the original quarantine ledger
`decisions.json` stays frozen for the v6 comparison. The ordinary purity builder still
changes only family assignment and adds provenance. Identity/content corrections are
in the separate versioned source snapshot, not hidden inside the family classifier.

## Reproducibility and validation

- `rebinding_validation.json`: all 806,599 rows compared; exactly seven changed,
  every unapproved row/field identical; all 18 pinned benchmark JSONL files identical.
- `rebind_source.py --input <pinned-original-records> --output <new-records>` replays
  the full-row/hash-bound corrections. Run with the project environment and `PYTHONPATH=.`.
  `pre_rebinding_snapshot.json` identifies the original packaged Stage-03 snapshot.
- `rebinding_publication.json`: new snapshot published with the existing
  `current_retrieval_artifacts.publish_snapshot` function. No new production runner.
- `rebinding_retrieval_verification.json`: current package, local snapshot, v7 overlay,
  catalog and both split indices verified; `rebinding_level_export_receipt.json`
  records the canonical ordinary-Parquet export and card-link validation.
- Current retrieval-eligible L1/L2/L3 counts: **42,435 / 12,524 / 12,005**.
  New source/index paths and hashes are in `current_starling_retrieval.json`.
  Historical v5/v6 indices remain untouched.

Rebuild the current source-derived retrieval with the shared entrypoint:

```bash
python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval build --tasks skin_reaction --workers 8
python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval verify --tasks skin_reaction
```

## Paired replay

`outputs/paper/skin_identity_rebinding_replay_20260914/` holds the v7 replay.
Its frozen legacy runtime differs from v6 only in the Skin index/catalog paths;
all 375 Python files were checked. It retains `progressive_compact_tools_short_aliases.v2`,
the old tools and the frozen query prior/None. Tool pairs use the original batch service.

The complete selected surfaces of all 480 scaffold valid/test rows were checked,
including agreement with the original hybrid runs' actual prepared inputs.
Against original v5 hybrid inputs, test changes #39/#45/#82/#133/#154/#179 at L3;
valid changes #9 at L2/L3 and #222 at L3. Against the completed v6 quarantine replay,
only **test #39 L3 and valid #222 L3** are new changes. All unchanged prefixes are
reused through the shared validator: 17 test levels and 5 valid levels.

The shared endpoint cap is 1,024 with race width 6. Both new L3 calls completed and passed actual-request/state verification in
`<split>/verification.json`. Neither label changed: full-test final Macro-F1 is
0.582343 (171/241 correct), valid is 0.650761 (178/239 correct). Registry suite
`skin_identity_rebinding_replay_20260914` holds the paired metrics.
The strict-policy surface audit is retained, but this experiment replays the
requested L3+ parent-disjoint configuration only.

The previous v6 experiment is `outputs/paper/skin_trace_source_replay_20260914/`:
all six affected queries completed, with no label changes on test or valid.
That result concerns quarantining records; it is not the result of correct rebinding.
Other retained/pending source checks are in `retained_and_pending.json`.
