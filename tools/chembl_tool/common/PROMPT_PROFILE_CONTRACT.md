# Shared reasoning and prompt-profile contract

This contract separates source-independent reasoning transport from task-local
prompt semantics. It applies to BBB_Martins, Bioavailability_Ma, Skin_Reaction,
ClinTox, and DILI runners.

## Ownership

```text
reasoning_payload.py
  Identity-aware query payloads, ChEMBL context cleaning, JSONL input helpers,
  explicit env-file loading, and the shared single/group/final trace writer.

reasoning_validation.py / reasoning_calls.py
  Structured-output retry, schema-enum validation, optional cross-field
  validation, and shared single/group call transport.

prompt_profile.py
  Manifest lookup and branch-reuse guards for task prompt profiles.

final_decision_prior.py
  An orthogonal final-only decision profile. `standard` is a strict no-op;
  non-standard profiles must be explicit, versioned, and auditable.

tasks/<task>/prompt_profiles.py
  Task-local system roles, instructions, required schemas, allowed values,
  label scope, and task-specific cross-field validation.
```

Task pipelines assemble retrieval and call the shared helpers. They must not
copy the common payload, validation, trace, or artifact-reuse implementations.

## Two independent profile axes

`task_prompt_profile` selects the task semantics for all single/group/final
branches. A branch artifact may be reused only when its recorded task profile
matches the target profile. Historical manifests without a profile field map
to the task's explicit historical profile; they never inherit a new default.

`final_decision_profile` changes only final-stage adjudication. The default
`standard` profile adds no fields, instructions, or validation. A non-standard
profile requires a final-only source batch and cannot be mixed into an existing
batch lineage.

Neither axis may change retrieval, evidence rows, gold labels, deterministic
postprocessing, or the prediction after a valid model response.

## Current task defaults

| Task | Historical profile | New-run default |
|---|---|---|
| BBB_Martins | `meaningful_cns_access_v1` | `meaningful_cns_access_v1` |
| Bioavailability_Ma | `legacy_bioavailability_v1` | `f20_evidence_calibrated_v2` |
| Skin_Reaction | `legacy_skin_reaction_v1` | `sensitization_aligned_v2` |

BBB v2/v3 and Skin v3 remain explicit historical/diagnostic opt-ins. They are
not promoted defaults and do not authorize formal-test tuning.

## Artifact requirements

Batch and run manifests record:

```text
task_prompt_profile
label_scope
final_decision_profile
```

Resume, final-only replay, frozen single/group reuse, and global-pool execution
must reject profile mismatches before issuing model calls. Structured responses
must pass the selected profile's required fields, allowed values, and any
cross-field validator before they can become predictions.
