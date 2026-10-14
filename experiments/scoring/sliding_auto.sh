#!/usr/bin/env bash
set -euo pipefail
: "${INPUT_MANIFEST:?task worlds and cached cards manifest}" "${RUN_ROOT:?fresh scoring directory}"
python_bin=${PYTHON_BIN:-python}
profile=${PROVIDER_CONFIG:-predict/api_client/providers/deepseek_v4_flash_0731_openrouter_auto_128_high.json}
"$python_bin" -m optimization.semantic_buckets.bucket_scoring prepare --passes "${PASSES:-2}" \
  --input-manifest "$INPUT_MANIFEST" --root "$RUN_ROOT" --chain-mode auto --max-chains 8 \
  --overlap 2 --advance-after 2 --pass1-batch-size 12 --pass2-batch-size 12 \
  --pass1-fanout 1 --pass1-average 1 --pass2-fanout 3 --pass2-average 3 \
  --pass1-provider-config "$profile" --pass2-provider-config "$profile" "$@"
# Preparation is offline. Execute only when explicitly requested by the operator.
if [[ ${RUN:-0} == 1 ]]; then
  "$python_bin" -m optimization.semantic_buckets.bucket_scoring run --root "$RUN_ROOT"
  "$python_bin" -m optimization.semantic_buckets.bucket_scoring collect --root "$RUN_ROOT"
fi
