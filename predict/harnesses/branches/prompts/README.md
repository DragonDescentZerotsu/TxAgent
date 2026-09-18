# Flat group prompts

`joseph_flat_v2/` is the default BBB/oral cache-matched flat bundle. It uses
source-contract-complete semantic cards and no group comparison tools while
leaving single-molecule and final prompts task-owned. `joseph_flat_v1/` and
`tianang_flat_v1/` remain explicit historical bundles.

Bundles resolve by exact version with no fallback. Preserve any version used by
an experiment and create a new sibling directory for later prompt changes.
