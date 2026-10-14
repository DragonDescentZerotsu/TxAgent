#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?task slug}" "${RELEASE_ROOT:?pinned V10 evidence release}" "${RETRIEVAL_INDEX:?complete retrieval index}" "${RUN_ROOT:?fresh construction directory}"
"${PYTHON_BIN:-python}" -m optimization.semantic_buckets.build "${STAGE:-prepare}" \
  --task "$TASK" --release-root "$RELEASE_ROOT" --retrieval-index "$RETRIEVAL_INDEX" \
  --run-root "$RUN_ROOT" "$@"
