# ClinTox Human Toxicity candidate benchmark

This active candidate uses the tracked ClinTox base human clinical source. It is not equivalent to TDC/MoleculeNet CT_TOX clinical-trial failure.

- Binary molecular parents: 7,628
- Accepted source rows before structure normalization: 511,805
- Parent label counts: `{"0": 755, "1": 6873}`
- Vote policy: one accepted source row per vote, 70% parent agreement, exact ties rejected.
- `qualifying_conditions` is unavailable in the delivered source schema; it was not treated as empty.
- Frozen source QA: `failed_manual_review` (108/360 non-passing rows: 96 fail and 12 uncertain).
- Status: candidate pending source-wide claim-policy revision and a new frozen QA sample.
