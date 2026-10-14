#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?task ID}" "${CACHE_BUNDLE:?pinned retrieval bundle}" "${OUTPUT_ROOT:?fresh search directory}"
"${PYTHON_BIN:-python}" -m optimization.selection.search indirect --tasks "$TASK" \
  --benchmark "${BENCHMARK:-gold_v1}" --subset "${SUBSET:-valid_small}" --grid "${GRID:-sqrt48}" \
  --cache-bundle "$CACHE_BUNDLE" --budget "${BUDGET:-50}" --workers "${WORKERS:-1}" --output-root "$OUTPUT_ROOT" "$@"
