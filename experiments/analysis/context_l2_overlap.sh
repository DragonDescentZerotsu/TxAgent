#!/usr/bin/env bash
set -euo pipefail
: "${CACHE_BUNDLE:?cache YAML}" "${CONTROL_ROOT:?control selections}" "${SEMANTIC_ROOT:?semantic selections}" "${OUTPUT_ROOT:?fresh analysis directory}"
"${PYTHON_BIN:-python}" -m analysis.prediction.context_l2_selection_overlap --cache-bundle "$CACHE_BUNDLE" \
  --control-root "$CONTROL_ROOT" --semantic-root "$SEMANTIC_ROOT" --output-root "$OUTPUT_ROOT" "$@"
