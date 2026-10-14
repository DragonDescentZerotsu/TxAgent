# Level descriptions and compact record metadata

> Historical bundle documentation. Preserve it for artifact interpretation, but
> do not treat the V8 defaults below as the current launch contract.

V8 restores `level_context.full_level_plan` and removes `evidence_family` from
model-visible records. Each record retains its numeric `first_seen_level`; the
level plan supplies the corresponding biological descriptions. Internal family
assignment and grouping are unchanged.

The assay-transfer system guidance describes potentially more informative records
from less-similar structures. Morgan and joint guidance, score visibility, semantic
record fields, null omission, and retrieval caps remain unchanged from v7.

The final projection in `prompt.py` retains the other metadata omissions. The
internal `card.yaml` assembly schema still carries fields needed for grouping and
validation. V8 is the sole active prompt bundle and has independent hashes and
output paths. The harness CLI version remains `reranked-progressive-v2`.

## Retrieval cache

Flat and progressive use the same immutable cache-matched bundle configured by
the harness. Each task
and split has one immutable SQLite database containing the query ledger, source
payloads, parent SMILES, parent-scoped Morgan similarity, and both native ranks.
Prompt preparation performs indexed rank lookups only; joint L1 merges the
native Morgan and assay top-five lists without a separately materialized cache.

Publish a cache once from the preserved legacy inputs:

```bash
python -m predict.retrieval.assay_reranking.build_cache_matched_v2 \
  --task bbb_martins --subset valid \
  --library data/evidence_libraries/bbb_martins/v10 \
  --mapper data/evidence_libraries/level_mappings.v1.json \
  --allow-frozen-l1-vote-scores

python -m predict.retrieval.assay_reranking.build_cache_matched_v2 \
  --task bioavailability_ma --subset valid \
  --library data/evidence_libraries/bioavailability_ma/v10 \
  --mapper data/evidence_libraries/level_mappings.v1.json
```
