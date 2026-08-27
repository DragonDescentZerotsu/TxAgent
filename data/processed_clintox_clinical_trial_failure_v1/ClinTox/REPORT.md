# ClinTox clinical-trial-failure benchmark v1

- status: `benchmark_ready`
- lineage: `clinical_trial_failure_v1`
- parent molecules: 1,428
- Y=0/Y=1: 1,324/104
- scaffold train/valid/test: 1,144/142/142
- valid Y=0/Y=1: 132/10
- test Y=0/Y=1: 132/10
- parent identity overlap: 0
- Bemis-Murcko scaffold overlap: 0

Y=1 is an existential source event: an AACT-derived record says the drug was associated with a clinical trial that failed for toxicity. FDA approval does not erase such an event. Y=0 is the FDA-approved comparator class with no positive source record after parent normalization.

General toxicity evidence remains retrieval context only and cannot create or override this benchmark label.
