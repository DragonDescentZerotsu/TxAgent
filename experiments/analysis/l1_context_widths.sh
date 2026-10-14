#!/usr/bin/env bash
set -euo pipefail
: "${RESULTS_ROOT:?completed organized prediction runs}" "${OUTPUT_ROOT:?fresh analysis directory}"
"${PYTHON_BIN:-python}" -m analysis.prediction.l1_context_v3 --results-root "$RESULTS_ROOT" --output "$OUTPUT_ROOT" \
  --widths 10 15 25 --k 10 --minimum-contrasts 0 1 2 --prior-directories with_query_prior no_query_prior "$@"
