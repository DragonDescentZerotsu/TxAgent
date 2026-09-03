# Evidence library V8 construction release

V8 contains the successor implementation previously developed as V9. The old
V8 implementation and complete BBB artifact were retired when this release was
renumbered. BBB Stage 0-1 and its V15 LLM measurement mapping are built;
downstream stages have not been built.

The BBB changes are isolated to measurement resolution:

- `tasks/bbb_martins/measurement_resolution_rules.py` is the sole ordered rule
  implementation.
- `tasks/bbb_martins/MEASUREMENT_RESOLUTION_RULES.md` records the reviewed rule
  contract and Stage-1 census.
- `prompts/measurement_resolution/bbb_v21.jinja` is the edited V8 prompt.
- Stage-1 scalar routing no longer invokes categorical encoding; the existing
  Stage-2 categorical fallback remains available when no scalar resolves.

The BBB build entrypoint is:

```bash
python -m data.processing.evidence_library.versions.v8.tasks.bbb_martins.build_normalized_starling_evidence_library
```
