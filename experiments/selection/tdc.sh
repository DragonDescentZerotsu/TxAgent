#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?task ID}" "${OUTPUT_ROOT:?fresh selection directory}"
"${PYTHON_BIN:-python}" -m optimization.selection.search "${KIND:-direct}" \
  --tasks "$TASK" --benchmark tdc --subset "${SUBSET:-valid}" \
  --output-root "$OUTPUT_ROOT" "$@"
