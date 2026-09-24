# Shared prompt bundle provenance

For labels and integration, use the [prompt handoff](../README.md#task-prompts).
`provenance.json` is authoritative for the bundle version, file hashes and label
mappings; `tasks.yaml` defines the six task contracts.

The templates originate from Joseph v5 commit
`3c36c2640ada12119f4a74a9bdf8511b0d7b8f6e`. The native-label revision changes only
the four risk tasks' label instructions, allowed values and mappings; templates,
scientific evidence rules and BBB/Bioavailability remain unchanged. Each revised
task also asks that the final label agree with the overall evidence assessment.
The original pass/fail bundle is at `945380eb`, and the DILI-only revision is at
`3f1e7b27`; historical assets and predictions are not overwritten.

`task_contract_sha256` identifies the original local source contracts, not the
edited YAML blocks. `files` contains hashes of the current bundle assets.
The legacy portable exporter emits the old contract and is not a regeneration
entrypoint for this maintained bundle.

Joseph's inspected `5c3a6ae2` branch already has six-task integration, but its
system/user templates have evolved. Apply the small task/mapping changes there
without replacing those templates. Our local runtime retains task-specific
prediction fields; this bundle retains `final_prediction`. The prior renderer
supports six tasks and excludes `exact_chembl_evidence_assessment`.
