# Skin canonical-gold sensitivity audit

This is a deterministic, valid-only diagnostic for the frozen
`record_supported_v2` Skin benchmark. It does not rebuild the benchmark, write
labels, or read the test split.

It compares the frozen label with:

- canonical direct record voting;
- canonical direct voting after an explicit direct-quality gate;
- PMID-level voting, with within-paper conflicts treated as abstentions;
- predictive animal/human study voting;
- human predictive, diagnostic, and case/occupational study voting.

Every canonical direct row is assigned one inspectable evidence-design family.
Generic human `patch test` rows are marked as inferred diagnostic evidence;
unresolved designs remain `other_direct_ambiguous` rather than being silently
discarded.

Run:

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m \
  tools.chembl_tool.paper_experiments.skin_canonical_gold_sensitivity.run
```

Default artifacts:

```text
outputs/paper/skin_canonical_gold_sensitivity_record_supported_v2_valid/
  summary.json
  report.md
  molecule_sensitivity.jsonl
  study_votes.jsonl
  classified_records.jsonl
```

The input SHA-256 values, split boundary, label contracts, coverage, unresolved
counts, family conflicts, and per-query flips are recorded in `summary.json`.

## Frozen result

The 245-row scaffold-valid audit completed without reading test. Canonical
record voting covers 238/245 molecules and produces 0 label flips; PMID-level
voting covers 231/245 and also produces 0 flips. Predictive-only study voting
covers only 79/245 and produces three `1 -> 0` flips. Twelve molecules have
conflicting resolved labels across evidence-design families. These results are
diagnostic and do not rewrite `record_supported_v2`.
