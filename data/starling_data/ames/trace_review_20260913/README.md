# Ames trace-source follow-up

`verification.json` is the completion and preservation receipt. `decisions.json`
records two placement corrections and one qualifying-context repair, applied via
the existing v6 placement and v4 accepted-review ledgers. Raw source records and
study-unit labels are preserved. This is a source repair, with no new model calls.

- `ames_v2:425939`: L2 → L4, SOS/umu response rather than performed Ames.
- `ames_base:186648`: L2 → L3, plant chromosome/micronucleus outcome.
- `ames_base:80868`: retain its cinnoline TA1537 without-S9 positive outcome;
  restore the original paper's quinoline exception in the series structural context.

PMID7022455's original Table 2 and pp. 3782–3783 confirm quinoline TA98 with-S9
positivity. The retained PDF was obtained from
<https://europepmc.org/articles/PMC319656?pdf=render>; its hash is in the receipt.
PMID39543775's later TA98 negative result remains a cross-study disagreement.
PMID1373859's abstract is verified, but its original tables were not accessible;
no source or gold deletion follows from that limitation.

`sample.jsonl` and `audit_annotations.jsonl` feed the independent placement audit.
The previous manifests, votes and ledgers document the before state.
`snapshot_publication.json` and `sharing_export.json` record the updated compressed
Stage-03 and shared Parquet publication. `validation.log` records the independent
full-source, full-index and sampled cumulative/progressive checks. Reproduction
uses the existing Ames builder and validator; no new build or inference entrypoint
is introduced. See the task README for the random row-order and prediction-reuse
boundary.
