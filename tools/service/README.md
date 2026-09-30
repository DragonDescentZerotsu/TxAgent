# Resident molecular tool service

`tools.service.app` exposes the three frozen molecular tools used by all retrieval methods. Retrieval code
selects neighbors; this service only computes deterministic molecule and molecule-pair evidence. Morgan,
MiniMol, coverage-based, and future retrievers therefore share the same prefetch/cache path.

## Implementation map

```text
tools/service/app.py
  FastAPI routes, including the one-request-per-sample /tools/batch endpoint.
tools/service/config.py
  Environment-backed worker, cache, MolGpKa, and native-thread settings.
tools/service/registry.py
  Tool lifecycle, batch execution, cache lookup/write, and single-flight coordination.
tools/service/cache.py
  In-process LRU/single-flight plus versioned SQLite/WAL persistence.
tools/service/runtime.py
  Native-library thread caps applied before model initialization.
tools/service/molgpka_predictor.py
  Process-resident acid/base MolGpKa networks reused by molecule_properties.
tools/service/functional_group_tree.py
  Internal AccFG match-hierarchy renderer used by molecule_properties; no CLI or endpoint.
tools/chembl_tool/common/openai_reasoning_client.py
  Batch-capable client with backward-compatible fallback to individual invokes.
tools/chembl_tool/common/identity_blind.py
  Retriever-agnostic assembly of the fixed per-sample tool bundle.
```

Keep retrieval selection outside this package. A new retriever should emit the common retrieval payload and
reuse `identity_blind` plus this service; it should not add another cache or tool-prefetch implementation.

## Functional-group trees

The existing `molecule_properties` response includes `[functional_group_tree]` in
`output.text`; `properties_compare` includes `[reference_functional_group_tree]`
for its reference molecule. Both use the shared, cached properties calculation.
The per-query bundle remains one query properties call and two comparisons per
neighbor, submitted through `/tools/batch`; no extra service tool or LLM call is
needed. The progressive/full-flat renderer places these sections in separate
query and neighbor `functional_group_tree` fields and leaves numeric property
summaries free of functional-group lists. Query-only single/None priors receive
the complete query properties text, including its tree section, inside
`prefetched_molecule_properties`; they have no neighbor/comparison inputs.

AccFG trees show nested substructure matches, not bond connectivity. They omit
atom indices and retain counts within each parent branch; parent and child counts
must not be added. Edge reduction runs on actual matches, so instances of the
same group at different positions cannot hide one another. Internal atom mappings
remain available for debugging. Rendering
uses local strings rather than capturing AccFG's global stdout, so concurrent
requests cannot mix tree text. Extraction failures retain numeric properties and
explicitly mark the tree unavailable. The implementation uses installed AccFG
0.0.9 and does not require an environment upgrade.

This output revision uses cache namespace `tool-service-2026-09-13-v5` and a new
internal properties-cache key. Start updated workers on a side port before a
production rollover; a successful health check alone does not prove tree support.
Check both tree sections with `/tools/batch`. New progressive runs reject successful
old tool receipts that lack the tree sections; fresh prior manifests also bind
the tool text profile. Historical single/None reuse remains explicit and does not
regenerate its old reasoning when the service changes.

## Production launch on node002

```bash
env \
  TXAGENT_TOOL_NATIVE_THREADS=1 \
  TXAGENT_TOOL_BATCH_WORKERS=8 \
  TXAGENT_TOOL_CACHE_MEMORY_ENTRIES=5000 \
  TXAGENT_TOOL_CACHE_PATH=/local/tmp/txagent-tool-cache.sqlite3 \
  python -m uvicorn \
    tools.service.app:app --host 127.0.0.1 --port 8765 --workers 32
```

The 32 process workers use the machine's CPU cores for Python/RDKit/mmpdb work. Each process has eight
bounded batch threads, while PyTorch, OpenMP, MKL, OpenBLAS, and NumExpr each use one native thread per
call. Do not increase native threads: nested `request workers x torch threads` caused the former service to
grow to more than 5,000 threads and about 50 GB RSS.

For a laptop or smoke test, use one or two Uvicorn workers. `TXAGENT_TOOL_BATCH_WORKERS` defaults to half
the available CPUs capped at 128 when only one service process is used.

## Fast path

The service and client implement four common optimizations:

1. MolGpKa acid/base networks load once per service process instead of once per prediction.
2. `/tools/batch` accepts the full deterministic tool bundle for one retrieved sample in one HTTP request.
3. In-process single-flight LRU caches deduplicate concurrent molecule properties and mmpdb fragmentation.
4. A local SQLite/WAL cache persists versioned tool results and fragmentation across processes and runs.

The default cache is `/local/tmp/txagent-tool-cache-<uid>.sqlite3`. Set an explicit path for a named run.
An empty `TXAGENT_TOOL_CACHE_PATH` disables the persistent layer. Cache keys include tool version, inputs,
debug mode, MolGpKa availability, pH, and an implementation namespace. Bump the namespace in
`tools/service/registry.py` whenever an output-semantic change does not already require a tool version bump.

MolGpKa forwards are serialized per resident predictor because its graph
convolution mutates shared state. Independent service processes remain parallel;
`TXAGENT_TOOL_BATCH_WORKERS` controls tool requests, not concurrent model forwards.
Unrepresentable molecules retain RDKit descriptors and explicit missing pKa/logD
with warning/debug details; unexpected predictor errors still fail the call.
MMP rejects remaining disconnected inputs after salt normalization, preserving
Morgan/MCS results and marking the transformation not applicable.
Properties-tool and fragmentation cache revisions exclude pre-fix outputs.

The OpenAI-compatible client automatically falls back to individual calls when it reaches an older service
without `/tools/batch`, permitting rolling upgrades. `identity_blind.prepare_harness_prefetched_retrieval`
builds batches solely from the common retrieval payload (`query`, `groups`, `neighbors`); retriever-specific
branches must not duplicate this logic.

## Verification

```bash
conda run -n vllm \
  python -m pytest -q tests/service tests/chembl_tool/common/test_identity_blind.py

curl -fsS http://127.0.0.1:8765/health
```

Before replacing a live service, validate on a side port and compare all three
tool outputs. A semantic correction requires cache invalidation and a dependent
benchmark-output audit/replay; matching source/index hashes alone does not permit
reuse of changed tool text. Drain in-flight work before switching endpoints.
