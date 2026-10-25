# Data processing

Construction code is versioned independently from experiment runners. The
[evidence-library guide](evidence_library/README.md) maps the shared kernels,
release builders, task policies and prompts. There is no `versions/vN/shared/`
package; release-wide modules live directly in `versions/vN/`.

[Gold-label construction](gold_labels/README.md) owns benchmark publishing.
Reasoning and retrieval execution live under `predict/`; these use the selected
frozen releases rather than rebuilding data as a side effect of prediction.
