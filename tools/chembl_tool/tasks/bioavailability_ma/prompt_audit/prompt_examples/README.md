# Compiled prompt examples

One example of the exact prompt sent to GLM at each pipeline stage, compiled
from a single frozen real datapoint (query index 0, group Fg). These are
**compiled prompts only** -- no model output. The group stage has one example
per format (legacy, morganfingerprint, assay_transfer_tool), plus the
evidence-centric assay-transfer output profile. The two text formats also
have a `.full.` variant showing `--presentation-style full` (per-source fields).

Do not edit by hand. Regenerate after any prompt-code change:

    python -m tools.chembl_tool.tasks.bioavailability_ma.prompt_audit.dump_prompt_examples --write

A golden-file test (tests/.../test_prompt_examples.py) fails if these drift.
