#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?task ID}" "${QUERIES:?benchmark JSONL}" "${SELECTION:?UID TSV or optimizer manifest}" "${CACHE_BUNDLE:?named cache-set manifest}" "${RUN_NAME:?readable experiment name}"
prepare=()
if [[ ${RUN:-0} != 1 ]]; then prepare=(--prepare-only); fi
"${PYTHON_BIN:-python}" -m predict.harnesses.branches --task "$TASK" \
  --input-jsonl "$QUERIES" --uids-file "$SELECTION" --retrieval-cache "$CACHE_BUNDLE" \
  --output-root "${OUTPUT_ROOT:-outputs/predictions}" --run-name "$RUN_NAME" \
  --benchmark "${BENCHMARK:-gold}" --evaluation-subset "${SUBSET:-valid}" \
  "${prepare[@]}" "$@"
