#!/usr/bin/env bash
# Run from the repository root. Data and credentials may live outside this checkout.
set -euo pipefail
PYTHON=${PYTHON:-python}
: "${PAPER_SOURCE_ROOT:?Point PAPER_SOURCE_ROOT to the frozen paper workspace}"
: "${TINKER_ENV_FILE:?Point TINKER_ENV_FILE to an existing .env containing TINKER_API_KEY}"
RUN_NAME=${RUN_NAME:-deepseek_v41_flash_1017_paper_20261031}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/predictions/tinker_sft/txgemma_style_label_v1/$RUN_NAME}
"$PYTHON" -m predict.baselines.tinker_sft prepare \
  --sources experiments/baselines/tinker_sft_paper_sources.json \
  --source-root "$PAPER_SOURCE_ROOT" --output "$OUTPUT_ROOT" \
  --epochs 5 --learning-rates 0.0001 0.0005 --rank 32 --seed 42 --batch-size 32
"$PYTHON" -m predict.baselines.tinker_sft run \
  --output "$OUTPUT_ROOT" --env-file "$TINKER_ENV_FILE" \
  --parallel-tasks 4 --inference-concurrency 32 --max-estimated-usd 1000
