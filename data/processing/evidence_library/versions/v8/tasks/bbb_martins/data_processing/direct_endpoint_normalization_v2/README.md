# Direct BBB endpoint normalization v2 errata

This directory is a human-gated correction layer over the immutable approved
v1 mapping. The proposal is generated from deterministic endpoint/value/unit
contradictions; it does not ask an LLM to infer endpoints or units.

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v8.tasks.bbb_martins.data_processing.build_direct_endpoint_errata
```

`proposal/proposed_endpoint_mapping.json` always starts with
`human_approved=false`. The runtime loader rejects it until a person reviews
the complete candidate table, records an approver and timestamp, and places
the approved full mapping under `approved/endpoint_mapping.json`. The v1 file
is never edited or replaced.
