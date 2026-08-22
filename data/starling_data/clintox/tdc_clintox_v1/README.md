# Frozen ClinTox definition sources v1

This directory preserves the original DeepChem/MoleculeNet ClinTox source
artifacts at the pinned upstream dataset revision recorded in
`SOURCE_MANIFEST.json`.

The strict binary target is reconstructed from source roles:

- `aacttox.csv.gz`: positive drugs associated with clinical trials that failed
  for toxicity reasons;
- `sweetfda_approved_processed.csv.gz`: FDA-approved comparator candidates;
- `clintox.csv.gz`: the upstream joined table, used only to reconcile the
  reconstruction;
- `AACT201603_pipe_delimited.tar.gz`, phase data, SMILES cache, and join script:
  retained supporting provenance.

The benchmark does not infer labels from literature prose. After molecular
parent normalization, any AACT positive event takes precedence when a parent
also appears in the FDA-approved comparator source. FDA approval and a failed
trial for another indication or program can coexist.
