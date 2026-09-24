# Collaborator prompt handoff

## Task prompts

Use [tasks.yaml](joseph_full_flat_v5/tasks.yaml) and the matching
[provenance.json](joseph_full_flat_v5/provenance.json) together. The shared bundle
keeps `claims` + `final_prediction` and uses our native labels for the risk tasks:

| Task | Positive (1) | Negative (0) |
|---|---|---|
| DILI | `dili_risk` | `no_dili_risk` |
| Ames | `positive` | `negative` |
| Carcinogens | `positive` | `negative` |
| Skin | `risk` | `no_risk` |

Apply these task definitions and label mappings to the existing harness.
Its validator derives allowed values from the configuration; its mapper uses
identity mappings for these four tasks. Keep `final_prediction` as the field
name and use a fresh prompt identity/output root. Do not run the historical
portable exporter to recreate this bundle: it emits the superseded pass/fail
contract. Copy the checked-in bundle or apply its task/mapping changes directly.

Templates, scientific evidence rules and BBB/Bioavailability are unchanged.
Offline rendering, validation, mapping and scoring checks passed. A small TDC DILI
replay is summarized below; full-test prompt improvements remain unverified.
This is a compatible adapter, not a
byte-identical copy of our local runtime prompt. Version and file hashes live
in `provenance.json`; [bundle provenance](joseph_full_flat_v5/README.md) explains
its relationship to the collaborator's templates.

- [TDC DILI: parent-disjoint diagnosis and scoped rerun](TDC_DILI_HANDOFF.md) — current request; TDC DILI only.
- [Gold-v1 DILI observations and retrieval suggestions](DILI_HANDOFF.md) — separate benchmark; no rerun requested now.
- [Ames, Carcinogens and Skin trace audit](LABEL_ENCODING_AUDIT.md)
- [Historical baseline plotting instructions](baseline_curves/README.md) — separate from prompt setup.
