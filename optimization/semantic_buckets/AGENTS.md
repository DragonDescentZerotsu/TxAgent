# Semantic bucket owner

Use only `optimization.semantic_buckets` and paths under this owner. Read [README.md](README.md).

- `bucket_construction.py` owns source-local splits/merges; `build.py` owns its stage workflow.
- `bucket_refinement.py` owns fresh audits and reviewed successor proposals.
- `bucket_scoring.py` owns initial batched scoring and repeated sorted sliding rescoring.
  Sliding current-pass anchors are the only policy; keep locked decisions, chain controls,
  overlap, replicas, late receipts and resumability. Defaults: two total passes.
- Keep prompts as Jinja templates in `prompts/`; scientific cards belong in `bucket_cards.py`.
- `execution.py` shares transport/execution; `storage.py` owns queues and receipts.
- `inputs.py` resolves selected manifests and physical membership. `ledger.py` owns UID/pair
  census, reviewed membership means and explicitly provisional weights/worklists.
- `publication.py` installs only explicitly reviewed successors. Finished candidates stay
  unselected until publication; never change evidence CURRENT or overwrite selected payloads.
- Preserve selected releases, essential assets and maintained pinned retrieval policies.
- Evidence level zero means unreviewed: keep those UID census rows unleveled, outside semantic assignment and pending atom construction. An L1-only evidence successor may reuse reviewed indirect maps and weights after exact row/level equivalence checks; rebuild its complete UID ledger and validate the resulting weight projection.
- Per-level selection, fixed-neighbor calibration, Bradley–Terry ranking and root aliases are
  retired. Do not recreate obsolete module aliases or historical recovery/migration commands.
- Task policies are invocation-local; no global task mutation. Python owns science and
  execution; Bash recipes instantiate experiments. No new SDK client outside shared transport.
