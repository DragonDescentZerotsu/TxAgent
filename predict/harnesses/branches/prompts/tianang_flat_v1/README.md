# Tianang flat v1

This bundle freezes the group-stage prompt used by the opt-in full-flat
harness. The task contracts come from Tianang main commit
`e56a79f371d1029201eff29e348d30e7beb94d5c`; the Morgan and assay-transfer
files add only retrieval-specific interpretation.

Single-molecule and final synthesis prompts remain owned by the selected task
profile. The CLI therefore requires the matching task profile recorded in
`tasks.yaml`. Retrieval, evidence selection, tool execution, and output labels
are unchanged. This prompt is unevaluated until a real pilot is completed.
