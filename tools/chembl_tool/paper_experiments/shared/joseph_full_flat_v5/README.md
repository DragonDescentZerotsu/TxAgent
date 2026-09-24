# Joseph v5 task export: DILI label correction

Upstream: `3c36c2640ada12119f4a74a9bdf8511b0d7b8f6e`.
Current revision: `joseph_full_flat_v5_dili_native_labels.v1` (2026-09-24).

The [DILI handoff](../DILI_HANDOFF.md) explains the observed pass/fail inversions.
This revision replaces only DILI's label instruction, positive/negative values
and provenance mapping with `dili_risk` / `no_dili_risk`, and adds one sentence
requiring the final label to agree with the evidence assessment. The original
assets and hashes remain in Git at `945380eb`.

`system.jinja`, `user.jinja`, `claims` + `final_prediction`, scientific evidence
instructions and the other five task definitions are unchanged. The templates
derive label text from `tasks.yaml`. The existing harness derives the required
JSON shape and allowed values from the same configuration; its mapper accepts
the new identity mapping. Use a fresh prompt identity/output root when applying
this patch to the collaborator's integrated harness. Offline rendering and
existing parser-function checks passed; no new LLM results are claimed.

`provenance.json` contains current file hashes and label mappings. Its
`task_contract_sha256` values identify the original local source contracts,
not hashes of the edited YAML blocks. The unchanged legacy portable exporter
still emits pass/fail; do not regenerate this revision with it unchanged.

The task file adds Skin_Reaction, Ames, DILI and Carcinogens to BBB/Bioavailability.
The user template displays task-specific cached prior fields. Other risk tasks
still use `pass` for the positive/risk class, not safety; the
[trace audit](../LABEL_ENCODING_AUDIT.md) confirms this also causes errors in
Ames and Carcinogens. Their configurations are not changed by this DILI patch.

This export does not copy a runner or change retrieval. Joseph's inspected
`5c3a6ae2` branch has six-task integration and completed traces. Its DILI task
configuration matches the original export after YAML parsing, while its
system/user templates have evolved. Apply the small DILI configuration and
mapping changes to that harness; no template replacement or new parser is needed.

Our local adapter keeps our existing JSON response contract and native DILI
labels (`dili_prediction: dili_risk | no_dili_risk`), uses cached single-molecule
analysis, and omits raw tools. It does not adopt Joseph's molecule-role
exclusivity or explicit molecule-citation statistics. The obsolete
`exact_chembl_evidence_assessment` prior field is excluded from the portable
template and local prior projection; historical source caches remain intact.
