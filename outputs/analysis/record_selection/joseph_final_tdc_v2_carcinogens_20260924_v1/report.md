# Joseph final collection: TDC-v2 Carcinogens

Added two complete original-split TDC-v2 Carcinogens test arms to the [final collection](../../../paper/assay_transfer_harness/joseph/final/collection.json). The collection now has 16 arms and 4,080 successful test queries. Both new arms cover all 56 Carcinogens test queries and served the fixed `deepseek/deepseek-v4-flash-0731` model.

| Selection | Profile | Macro-F1 | Accuracy | Positive F1 | Coverage |
|---|---|---:|---:|---:|---:|
| Direct | `ga075_mc000_label010` | 0.695652 | 0.821429 | 0.500000 | 56/56 |
| Direct + indirect | `ga125_mc025_sr025_sd025_ld025_rd025` | 0.726830 | 0.785714 | 0.600000 | 56/56 |

The two arms use identical L1 context selections, the TDC-v2 successor prompt derived from upstream-v3, the same cached query prior, and a 128-slot OpenRouter request pool. Direct + indirect adds 50 selected L2-L7 UIDs per query. OpenRouter's automatic upstream routing differed between the arms, so the score difference cannot be assigned solely to indirect evidence. Each arm's `query_provenance.tsv` records the actual provider for every response.

[results.tsv](results.tsv) includes confusion counts and upstream-provider counts. [provenance.json](provenance.json) pins the source batches, optimization manifests, installed final arms, and collection index. Published source batches and prior final arms remain unchanged.
