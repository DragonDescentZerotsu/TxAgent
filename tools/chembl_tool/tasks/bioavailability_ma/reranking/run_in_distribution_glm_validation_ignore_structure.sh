#!/usr/bin/env bash
# Run the matched group-instruction ablation for k=3/5/7/10/15.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_profile.sh"
cd "${TXAGENT_PROJECT_ROOT}"

: "${TXAGENT_TOOL_SERVICE_PORT:=8766}"
: "${TXAGENT_GLM_PARALLELISM:=32}"
: "${TXAGENT_GLM_GROUP_WORKERS:=5}"
: "${TXAGENT_GLM_LIMIT:=0}"
: "${TXAGENT_ASSAY_TRANSFER_TOP_K_VALUES:=3,5,7,10,15}"
: "${TXAGENT_ASSAY_TRANSFER_MIN_SCORE:=0.5}"
: "${TXAGENT_NORMAL_BATCH_ROOT:=outputs/paper/molecular_evidence_agent/validation_parent_disjoint/bioavailability_ma/v6_5_no_query_extra_details}"
: "${TXAGENT_GLM_BATCH_ROOT:=outputs/paper/molecular_evidence_agent/validation_parent_disjoint/bioavailability_ma/v6_5_no_query_extra_details_ignore_structure_trust_likelihoods}"
: "${TXAGENT_GROUP_PROMPT_INSTRUCTIONS_FILE:=tools/chembl_tool/tasks/bioavailability_ma/prompt_instructions/assay_transfer_tool_ignore.txt}"

if [[ -z "${LITELLM_API_KEY:-}" ]]; then
  LITELLM_API_KEY="$("${TXAGENT_REASONING_PYTHON}" -c \
    'from keys import LITELLM_API_KEY; print(LITELLM_API_KEY, end="")')"
  export LITELLM_API_KEY
fi
if [[ -z "${LITELLM_API_KEY}" ]]; then
  echo "LITELLM_API_KEY is unavailable; export it or provide ignored keys.py" >&2
  exit 2
fi

ROOT="outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution"
CONDITION="${ROOT}/v6_5_no_query_extra_details"
VERSION_PATH="${CONDITION}/VERSION.json"
TOOL_URL="http://127.0.0.1:${TXAGENT_TOOL_SERVICE_PORT}"
SERVICE_LOG="${CONDITION}/tool-service-ignore-${SLURM_JOB_ID:-local}.log"

VERSION_SCORE_COUNT="$("${TXAGENT_REASONING_PYTHON}" - "${VERSION_PATH}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "complete":
    raise SystemExit(f"cache VERSION status is not complete: {payload.get('status')!r}")
print(int(payload["n_prompt_scores"]))
PY
)"
: "${TXAGENT_RERANK_EXPECTED_SCORE_COUNT:=${VERSION_SCORE_COUNT}}"

LIMIT_ARGS=()
if [[ "${TXAGENT_GLM_LIMIT}" -gt 0 ]]; then
  LIMIT_ARGS=(--limit "${TXAGENT_GLM_LIMIT}")
fi

TXAGENT_TOOL_SERVICE_PORT="${TXAGENT_TOOL_SERVICE_PORT}" \
  "${TXAGENT_REASONING_PYTHON}" -m uvicorn tools.service.app:app \
  --host 127.0.0.1 --port "${TXAGENT_TOOL_SERVICE_PORT}" \
  >"${SERVICE_LOG}" 2>&1 &
service_pid=$!
cleanup() {
  kill "${service_pid}" 2>/dev/null || true
  wait "${service_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${TXAGENT_REASONING_PYTHON}" - "${TOOL_URL}/health" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
last_error = "service did not respond"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
        if (
            payload.get("status") == "ok"
            and payload.get("tools_initialized") == payload.get("tools_total") == 3
            and not payload.get("tools_with_initialization_errors")
        ):
            print("tool service ready", flush=True)
            break
        last_error = f"unhealthy response: {payload}"
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
else:
    raise SystemExit(f"tool service health preflight failed: {last_error}")
PY

IFS=',' read -r -a top_k_values <<<"${TXAGENT_ASSAY_TRANSFER_TOP_K_VALUES}"
for top_k in "${top_k_values[@]}"; do
  normal_batch_id="bioavailability_ma__starling_in_distribution__full_mechanism__assay_transfer_tool_scored__k${top_k}__scoremin050__parent_disjoint__no_query_extra_details__no_group_tools__canonical_endpoint"
  source_batch="${TXAGENT_NORMAL_BATCH_ROOT}/${normal_batch_id}"
  if [[ ! -f "${source_batch}/metrics.json" ]]; then
    echo "Matched normal source batch is incomplete: ${source_batch}" >&2
    exit 2
  fi
  batch_id="${normal_batch_id}__ignore_structure__trust_transfer_likelihoods"
  "${TXAGENT_REASONING_PYTHON}" -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
    --python-executable "${TXAGENT_REASONING_PYTHON}" \
    --input-jsonl data/processed/Bioavailability_Ma/valid.jsonl \
    --batch-root "${TXAGENT_GLM_BATCH_ROOT}" \
    --batch-id "${batch_id}" \
    --index "${ROOT}/starling_in_distribution_neighbor_index.pkl" \
    --retrieval-source starling_in_distribution \
    --experiment-mode full_mechanism \
    --neighbor-identity-policy parent_disjoint \
    --retrieval-strategy assay_transfer_tool \
    --assay-transfer-initial-morgan-filter 100 \
    --rerank-catalog "${ROOT}/starling_in_distribution_catalog.jsonl" \
    --rerank-candidate-manifest "${CONDITION}/manifest.jsonl" \
    --rerank-cache "${CONDITION}/scores.sqlite3" \
    --rerank-cache-mode read_only \
    --assay-transfer-model jiosephlee/assay-transfer-tool \
    --assay-transfer-model-revision 9515603b1a5c4586e41c221dcdbc5e7487c0c3f5 \
    --assay-transfer-template-profile v6_5_query_context_copy_no_extra_details \
    --rerank-expected-score-count "${TXAGENT_RERANK_EXPECTED_SCORE_COUNT}" \
    --rerank-cache-version-manifest "${VERSION_PATH}" \
    --rerank-preflight-source-batch "${source_batch}" \
    --top-k-per-group "${top_k}" \
    --min-similarity 0.0 \
    --assay-transfer-min-score "${TXAGENT_ASSAY_TRANSFER_MIN_SCORE}" \
    --enable-assay-transfer-scores \
    --group-prompt-format assay_transfer_tool \
    --group-prompt-instructions-file "${TXAGENT_GROUP_PROMPT_INSTRUCTIONS_FILE}" \
    --disable-group-tools \
    --single-analysis-source-batch "${source_batch}" \
    --retrieval-replay-source-batch "${source_batch}" \
    --api-key-env LITELLM_API_KEY \
    --base-url https://litellm.parcc.upenn.edu/v1 \
    --model zai-org/GLM-5.2-FP8 \
    --disable-thinking \
    --reasoning-effort "" \
    --tool-service-url "${TOOL_URL}" \
    --parallelism "${TXAGENT_GLM_PARALLELISM}" \
    --group-workers "${TXAGENT_GLM_GROUP_WORKERS}" \
    "${LIMIT_ARGS[@]}" \
    --skip-existing
done
