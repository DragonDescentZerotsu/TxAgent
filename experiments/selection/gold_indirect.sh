#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?task ID}" "${OUTPUT_ROOT:?fresh selection directory}"
"${PYTHON_BIN:-python}" -m optimization.selection.search indirect --tasks "$TASK" --output-root "$OUTPUT_ROOT" \
  --benchmark gold_v1 --subset valid_small --grid "${GRID:-historical}" --budget 50 "$@"
