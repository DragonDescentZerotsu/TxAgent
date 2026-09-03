# Data processing

Evidence-library construction code is versioned independently from experiment
runners:

- `evidence_library/shared/v1/` contains immutable cross-release code.
- `evidence_library/versions/vN/shared/` contains release-specific shared code.
- `evidence_library/versions/vN/tasks/` contains release-specific task policies.
- `evidence_library/versions/vN/prompts/` contains that release's LLM prompts.

Construction imports must use these canonical packages directly. Reasoning and
retrieval workflows remain under `tools/chembl_tool/`; they do not host
evidence-library construction modules.
