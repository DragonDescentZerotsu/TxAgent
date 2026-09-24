# Joseph v5 task export: native risk labels

Upstream templates: `3c36c2640ada12119f4a74a9bdf8511b0d7b8f6e`.
Current revision: `joseph_full_flat_v5_risk_native_labels.v1` (2026-09-24).

The four risk tasks now use the same label values as our local runtime:
DILI `dili_risk / no_dili_risk`, Ames and Carcinogens `positive / negative`,
and Skin `risk / no_risk`. See the [handoff table](../README.md#task-prompts).
Only label instructions, allowed values and mappings change; each task also
asks that the final label agree with the overall evidence assessment.

`system.jinja`, `user.jinja`, `claims` + `final_prediction`, scientific evidence
rules and BBB/Bioavailability definitions are unchanged. The templates derive
label text from `tasks.yaml`. The existing harness derives allowed values and
the required JSON shape from the same configuration, and its mapper accepts
the identity mappings. Apply the task and mapping changes together to the
collaborator's integrated harness using a fresh prompt identity/output root.
No new parser, runner or inference stage is needed. Offline rendering and
existing validation/mapping/scoring function checks passed; no LLM rerun yet.

The original pass/fail bundle remains at `945380eb`; the DILI-only revision is
at `3f1e7b27`. The [DILI handoff](../DILI_HANDOFF.md) and
[trace audit](../LABEL_ENCODING_AUDIT.md) explain the observed encoding errors.
Skin was changed to remove the same ambiguity, although the screening did not
confirm a final Skin inversion.

`provenance.json` contains current file hashes and label mappings. Its
`task_contract_sha256` values identify the original local source contracts,
not hashes of the edited YAML blocks. The legacy portable exporter still emits
pass/fail; do not regenerate this revision with that exporter unchanged.

This export does not change retrieval or copy our complete runtime prompt.
Joseph's inspected `5c3a6ae2` branch has six-task integration and completed
traces; its system/user templates have evolved. Apply these small configuration
and mapping changes without replacing those templates. Our local runtime keeps
its task-specific JSON fields, while this bundle keeps `final_prediction`.
The prior renderer supports all six tasks and excludes the obsolete
`exact_chembl_evidence_assessment` field; historical caches remain intact.
