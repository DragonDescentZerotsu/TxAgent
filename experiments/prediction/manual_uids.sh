#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?task ID}" "${QUERIES:?benchmark JSONL}" "${SELECTION:?UID TSV or optimizer manifest}" "${CACHE_BUNDLE:?selected cache YAML}" "${BATCH_ROOT:?fresh output directory}"
prepare=()
if [[ ${RUN:-0} != 1 ]]; then prepare=(--prepare-only); fi
"${PYTHON_BIN:-python}" -m predict.harnesses.branches --organization flat --task "$TASK" \
  --input-jsonl "$QUERIES" --uids-file "$SELECTION" --assay-transfer-cache "$CACHE_BUNDLE" \
  --batch-root "$BATCH_ROOT" --benchmark "${BENCHMARK:-gold}" --evaluation-subset "${SUBSET:-valid}" \
  "${prepare[@]}" "$@"
