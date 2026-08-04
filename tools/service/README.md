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
tools/chembl_tool/common/openai_reasoning_client.py
  Batch-capable client with backward-compatible fallback to individual invokes.
tools/chembl_tool/common/identity_blind.py
  Retriever-agnostic assembly of the fixed per-sample tool bundle.
```

Keep retrieval selection outside this package. A new retriever should emit the common retrieval payload and
reuse `identity_blind` plus this service; it should not add another cache or tool-prefetch implementation.

## Production launch on node002

```bash
env \
  TXAGENT_TOOL_NATIVE_THREADS=1 \
  TXAGENT_TOOL_BATCH_WORKERS=8 \
  TXAGENT_TOOL_CACHE_MEMORY_ENTRIES=5000 \
  TXAGENT_TOOL_CACHE_PATH=/local/tmp/txagent-tool-cache.sqlite3 \
  /data1/tianang/anaconda3/envs/vllm/bin/python -m uvicorn \
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

The OpenAI-compatible client automatically falls back to individual calls when it reaches an older service
without `/tools/batch`, permitting rolling upgrades. `identity_blind.prepare_harness_prefetched_retrieval`
builds batches solely from the common retrieval payload (`query`, `groups`, `neighbors`); retriever-specific
branches must not duplicate this logic.

## Verification

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m pytest -q tests/service tests/chembl_tool/common/test_identity_blind.py

curl -fsS http://127.0.0.1:8765/health
```

Before replacing a live service, start the new implementation on a side port, compare all three tool outputs
against the current service, run a cold/hot real-retrieval stress test, and switch only between matrix
conditions. Existing benchmark output text must remain byte-for-byte equivalent.

## node002 validation snapshot (2026-08-02)

- Resident and package MolGpKa outputs matched exactly on acidic, basic, phenolic, and mixed-site molecules;
  complete `molecule_properties` and `properties_compare` outputs also matched exactly.
- Forty real BBB full-mechanism retrievals (286 neighbor pairs) took 201.66 seconds on a cold cache and
  1.52 seconds on the immediate hot-cache replay (about 133x faster).
- The 32-process service initialized all workers without errors and used about 28.7 GB aggregate RSS before
  load, compared with about 50.8 GB and 5,318 threads in the former single long-lived process.
- After the production switch, tool connections drained before reasoning and all eight GPT-OSS GPUs reached
  99-100% utilization; the tool service was no longer the active feed bottleneck.
