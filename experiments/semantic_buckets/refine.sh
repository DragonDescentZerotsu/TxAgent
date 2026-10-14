#!/usr/bin/env bash
set -euo pipefail
: "${TASK:?dili or skin_reaction}" "${RUN_ROOT:?fresh refinement directory}"
"${PYTHON_BIN:-python}" -m optimization.semantic_buckets.bucket_refinement "${STAGE:-prepare}" \
  --task "$TASK" --output "$RUN_ROOT" --max-inflight "${MAX_INFLIGHT:-32}" "$@"
