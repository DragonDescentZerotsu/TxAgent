#!/usr/bin/env bash
# Recompute the selected reproduction offline; raw inference runs stay frozen.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-python}
output=${1:?Usage: bash experiments/baselines/molrag.sh NEW_REPORT_DIRECTORY}
base="$(pwd)/outputs/predictions/molrag/molrag_evidence_v1"
suffix=naive_sources_direct_sim_cot_low_20261107_reviewed
"$PYTHON_BIN" -m predict.baselines.molrag score \
  --output-dir "$output" --run-root \
  "$base/deepseek_v4_flash_0731_$suffix" \
  "$base/gpt6_luna_native_temperature_$suffix" \
  "$base/gemini38_flash_$suffix"
