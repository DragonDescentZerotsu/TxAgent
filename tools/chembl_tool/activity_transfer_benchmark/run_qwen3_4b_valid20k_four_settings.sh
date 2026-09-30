#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-qwen3-4b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:9001/v1}"
API_KEY="${API_KEY:-EMPTY}"
PARALLELISM="${PARALLELISM:-64}"
TIMEOUT_S="${TIMEOUT_S:-180}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
MAX_TOKENS_THINK="${MAX_TOKENS_THINK:-20480}"
OUT_ROOT="${OUT_ROOT:-outputs/chembl_tool/activity_transfer_benchmark/llm_runs}"
COMPARISON_DIR="${COMPARISON_DIR:-outputs/chembl_tool/activity_transfer_benchmark/comparisons/hf_transfer_valid20k/qwen3_4b_proper_valid20k}"

INPUT_NO_PROPS="${INPUT_NO_PROPS:-outputs/chembl_tool/activity_transfer_benchmark/hf_transfer_valid20k/proper_assay_transfer_no_prop_no_tanimoto/validation.jsonl}"
INPUT_PROPS="${INPUT_PROPS:-outputs/chembl_tool/activity_transfer_benchmark/hf_transfer_valid20k/proper_assay_transfer_no_tanimoto/validation.jsonl}"

RUN_NO_PROPS_CHOICE="${RUN_NO_PROPS_CHOICE:-qwen3_4b_proper_no_prop_no_tanimoto_valid20k_choice_no_thinking}"
RUN_NO_PROPS_JSON_THINK="${RUN_NO_PROPS_JSON_THINK:-qwen3_4b_proper_no_prop_no_tanimoto_valid20k_json_with_thinking_trace}"
RUN_PROPS_CHOICE="${RUN_PROPS_CHOICE:-qwen3_4b_proper_no_tanimoto_valid20k_choice_no_thinking}"
RUN_PROPS_JSON_THINK="${RUN_PROPS_JSON_THINK:-qwen3_4b_proper_no_tanimoto_valid20k_json_with_thinking_trace}"

run_choice_no_thinking() {
  local input_jsonl="$1"
  local run_id="$2"
  "${PYTHON_BIN}" -m tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark \
    --input-jsonl "${input_jsonl}" \
    --out-root "${OUT_ROOT}" \
    --run-id "${run_id}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --output-mode choice \
    --disable-tools \
    --disable-thinking \
    --parallelism "${PARALLELISM}" \
    --timeout-s "${TIMEOUT_S}" \
    --skip-existing \
    --progress-every "${PROGRESS_EVERY}"
}

run_json_with_thinking() {
  local input_jsonl="$1"
  local run_id="$2"
  "${PYTHON_BIN}" -m tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark \
    --input-jsonl "${input_jsonl}" \
    --out-root "${OUT_ROOT}" \
    --run-id "${run_id}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --output-mode json \
    --disable-tools \
    --enable-thinking \
    --parallelism "${PARALLELISM}" \
    --timeout-s "${TIMEOUT_S}" \
    --skip-existing \
    --max-tokens "${MAX_TOKENS_THINK}" \
    --progress-every "${PROGRESS_EVERY}"
}

echo "[qwen3-4b valid20k] 1/4 no-props choice no-thinking"
run_choice_no_thinking "${INPUT_NO_PROPS}" "${RUN_NO_PROPS_CHOICE}"

echo "[qwen3-4b valid20k] 2/4 no-props json thinking trace"
run_json_with_thinking "${INPUT_NO_PROPS}" "${RUN_NO_PROPS_JSON_THINK}"

echo "[qwen3-4b valid20k] 3/4 props choice no-thinking"
run_choice_no_thinking "${INPUT_PROPS}" "${RUN_PROPS_CHOICE}"

echo "[qwen3-4b valid20k] 4/4 props json thinking trace"
run_json_with_thinking "${INPUT_PROPS}" "${RUN_PROPS_JSON_THINK}"

echo "[qwen3-4b valid20k] plotting comparison"
"${PYTHON_BIN}" -m tools.chembl_tool.activity_transfer_benchmark.plot_llm_multi_run_comparison \
  --input-jsonl "${INPUT_PROPS}" \
  --out-dir "${COMPARISON_DIR}" \
  --title "Qwen3-4B proper assay transfer valid20k comparison" \
  --run "no_props_choice_no_think=${OUT_ROOT}/${RUN_NO_PROPS_CHOICE}" \
  --run "no_props_json_think=${OUT_ROOT}/${RUN_NO_PROPS_JSON_THINK}" \
  --run "props_choice_no_think=${OUT_ROOT}/${RUN_PROPS_CHOICE}" \
  --run "props_json_think=${OUT_ROOT}/${RUN_PROPS_JSON_THINK}"

echo "[qwen3-4b valid20k] done"
