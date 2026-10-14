#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?task ID}" "${DIRECT_CACHE:?canonical L1 release index}" "${OUTPUT_ROOT:?fresh search directory}"
"${PYTHON_BIN:-python}" -m optimization.selection.search direct --tasks "$TASK" \
  --benchmark "${BENCHMARK:-gold_v1}" --subset "${SUBSET:-valid_small}" --grid "${GRID:-direct}" \
  --direct-cache "$DIRECT_CACHE" --output-root "$OUTPUT_ROOT" "$@"
