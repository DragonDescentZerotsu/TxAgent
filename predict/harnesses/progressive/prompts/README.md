# Progressive prompt assets

> Historical prompt index retained for reference. Used bundles remain immutable;
> the current front-shelf prompt contract is defined by the repository `AGENTS.md`.

`reranked_progressive_v8/` owns the active runtime behavior. Starting with
`reranked_progressive_l1_simple_v10`, SMILES visibility is expressed by suffix
modifiers on one immutable base prompt:

| Prompt version | Query SMILES | Evidence SMILES | Query-property prior |
|---|---:|---:|---:|
| `reranked_progressive_l1_simple_v10` | shown | shown | shown |
| `reranked_progressive_l1_simple_v10_no_smiles` | shown | hidden | shown |
| `reranked_progressive_l1_simple_v10_no_query_smiles` | hidden | shown | shown |
| `reranked_progressive_l1_simple_v10_no_smiles_no_query_smiles` | hidden | hidden | shown |
| `reranked_progressive_l1_simple_v10_no_query_prior` | shown | shown | hidden |

The suffixes do not have separate prompt directories. Their instructions and
field-removal rules live in the base version's `provenance.json`, are appended
by the renderer, and are hashed into the effective prompt manifest. Later simple
prompt versions retain the same ablations by carrying the small modifier
definitions forward.

The historical `reranked_progressive_l1_simple_v8` and
`reranked_progressive_l1_simple_v9` bundles remain immutable because completed
runs already pin their hashes. Their earlier hard-coded visibility is not
reinterpreted as a suffix modifier.

`reranked_progressive_l1_simple_v10` is also frozen by its first live pilot.
The example-based prompt revision and variable-claim `derived_v2` contract begin
at `reranked_progressive_l1_simple_v11`. Version `simple_v12` changes display
formatting only: experimental details use nested dashes and the query/evidence
section spacing is normalized. The same suffix modifiers apply to both versions.

Do not edit a used version in place. Clone it from a live run with
`python -m predict.live clone-prompt RUN_ID --to VERSION`, edit the new bundle,
and relaunch with `python -m predict.live relaunch RUN_ID --prompt-version VERSION`.
The clone inherits only v8 assembly behavior; its prompt files and hashes remain
independent. Prompt bundles may also inherit immutable assets transitively.

Historical prompt directories were removed from this active checkout after an
audit of recent run receipts. `migration_receipt.json` records the cutoff, source
commit, deleted scopes, and aggregate hashes. Existing result artifacts remain
immutable and retain their original manifests; exact old behavior belongs to its
pinned checkout.
