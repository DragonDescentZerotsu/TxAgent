# Joseph v5 historical task export

Upstream: `3c36c2640ada12119f4a74a9bdf8511b0d7b8f6e`.

For new DILI runs, follow the [DILI handoff](../DILI_HANDOFF.md): use native
`dili_risk` / `no_dili_risk` labels with matching validation and mapping.
The pass/fail translation below caused confirmed reasoning/output inversions.
These template/configuration files and their hashes remain frozen for provenance;
they have not been silently replaced with a new, unevaluated prompt.

The task file added Skin_Reaction, Ames, DILI and Carcinogens to BBB/Bioavailability.
It preserves the pinned upstream two-field output and output instructions.
`provenance.json` records source contracts and pass/fail-to-local-label mappings.
In risk tasks, pass denotes the positive/risk class, not safety.
The user template also displays task-specific cached prior fields for added tasks.

This export does not copy a runner or change retrieval. At export time, the four
additional tasks still needed runner/data registration; Joseph's inspected
`5c3a6ae2` branch now has that integration and completed traces. The DILI task
configuration is unchanged from this export after YAML parsing, although his
system/user templates have evolved.

Our local adapter separately keeps our existing JSON response contract and native
DILI labels (`dili_prediction: dili_risk | no_dili_risk`), uses
cached single-molecule analysis, and omits raw tools. It does not adopt Joseph's
molecule-role exclusivity or explicit molecule-citation statistics.

The obsolete exact_chembl_evidence_assessment prior field is excluded from the
portable template and local prior projection; historical source caches remain intact.
