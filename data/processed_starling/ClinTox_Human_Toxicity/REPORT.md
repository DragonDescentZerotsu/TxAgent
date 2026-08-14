# ClinTox Human Toxicity candidate benchmark

This candidate predicts explicit human clinical organ injury and is not equivalent to TDC/MoleculeNet CT_TOX clinical-trial failure.

- Binary molecular parents: 3,319
- Accepted source rows before structure normalization: 79,571
- Parent label counts: `{"0": 454, "1": 2865}`
- Vote policy: one accepted source record per vote, 70% parent agreement, exact ties rejected.
- Status: candidate pending deterministic source-record QA.
