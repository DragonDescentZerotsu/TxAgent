# Identity visibility experiment

Both visibility settings remain part of the paper scope.

## Contracts

`identity_blind` removes query/neighbor structures, identifiers, and source
names from the messages shown to the model. The harness may still use structure
to retrieve and precompute molecular comparison tools.

`deployment_visible_prefetched` replays the exact blind retrieval and tool
payload, then restores the permitted structures and source identities. This is
the matched visibility control because evidence availability and tool outputs
are held fixed.

`deployment_visible` performs fresh visible retrieval and allows end-to-end
model tool use. It remains supported for deployment studies but is not a pure
visibility comparison.

## Current evidence status

The latest complete historical blind and visible source matrices are retained,
but they use different benchmark lineages. They cannot support a matched causal
claim about identity visibility. A complete current Conditioned Benchmark
blind-visible pair has not yet been produced.

The append-only progressive experiment is currently
`deployment_visible_prefetched`; the one-shot cumulative family curve is
currently identity-blind. Their prompt/state protocols also differ, so their
scores must not be compared as a visibility ablation.

Current roots and freshness are recorded only in
`current_conditioned_results.json`. Any future matched comparison must use the
same task rows, source/index hashes, evidence organization, model, prompt,
neighbor policy, and prefetched tool payload, differing only in visibility.

## Required audit

- Verify the blind request messages contain no query/neighbor SMILES, stable
  identifiers, or known source molecule names.
- Verify the visible-prefetched retrieval and tool hashes equal the paired blind
  batch.
- Report structure/name visibility separately from tool execution.
- Do not infer compliance from CLI flags alone; inspect stored request messages.
- Report any model-generated identity guess as output behavior, not input leak,
  unless the identity was actually present in an upstream message.
