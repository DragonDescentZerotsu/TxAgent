# Reference v5 task handoff

Upstream: `3c36c2640ada12119f4a74a9bdf8511b0d7b8f6e`.

The task file adds Skin_Reaction, Ames, DILI and Carcinogens to BBB/Bioavailability.
It preserves the latest upstream two-field output and output instructions.
`provenance.json` records source contracts and pass/fail-to-local-label mappings.
In risk tasks, pass denotes the positive/risk class, not safety.
The user template also displays task-specific cached prior fields for added tasks.

Use these templates in the existing full-flat prompt directory. This bundle does
not copy a runner or change retrieval. To run the four added tasks on the reference
branch, register their task/data modules and task level counts in the existing
runner/retrieval registries (the inspected flat.py TASKS currently lists only
BBB and Bioavailability). Dataset and index provisioning remains necessary.
No remote branch was modified or six-task Reference experiment launched.

Our local adapter separately keeps our existing JSON response contract, uses
cached single-molecule analysis, and omits raw tools. It does not adopt the reference
molecule-role exclusivity or explicit molecule-citation statistics.

The obsolete exact_chembl_evidence_assessment prior field is excluded from the
portable template and local prior projection; historical source caches remain intact.
