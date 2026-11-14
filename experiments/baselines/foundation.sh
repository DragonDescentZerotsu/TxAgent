#!/usr/bin/env bash
# Seven encoders; the shared Python runner owns training, selection and auditing.
set -euo pipefail
RUNTIME_ROOT=${RUNTIME_ROOT:-/data1/tianang/Projects/TxAgent-foundation-models}
SOURCE_ROOT=${SOURCE_ROOT:-/data1/tianang/Projects/TxAgent}
RUN_NAME=${RUN_NAME:-seven_models_minimol_fixed_lambda_20261107}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/predictions/foundation/paper_v1/$RUN_NAME}
DEFAULT_PYTHON=$RUNTIME_ROOT/envs/main/bin/python
if [[ -x "$RUNTIME_ROOT/envs/ssd/bin/python" ]]; then DEFAULT_PYTHON=$RUNTIME_ROOT/envs/ssd/bin/python; fi
PYTHON=${PYTHON:-$DEFAULT_PYTHON}
PHASE=${1:-verify}
if (($#)); then shift; fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
args=(--output "$OUTPUT_ROOT" --runtime-root "$RUNTIME_ROOT")
if [[ "$PHASE" == prepare ]]; then
  args+=(--source-root "$SOURCE_ROOT" --run-name "$RUN_NAME")
  if [[ -n "${AUXILIARY_POLICY:-}" ]]; then args+=(--auxiliary-policy "$AUXILIARY_POLICY"); fi
elif [[ "$PHASE" == run ]]; then
  args+=(--workers "${FEATURE_WORKERS:-16}" --conformer-workers "${CONFORMER_WORKERS:-64}"
         --slots-per-gpu "${SLOTS_PER_GPU:-4}" --cpu-slots "${CPU_SLOTS:-16}"
         --repeated-runs --unimol-precision "${UNIMOL_PRECISION:-bf16}")
  if [[ -n "${LEGACY_MANIFEST:-}" ]]; then args+=(--legacy-manifest "$LEGACY_MANIFEST"); fi
  if [[ -n "${CONFORMER_CACHE:-}" ]]; then args+=(--conformer-cache "$CONFORMER_CACHE"); fi
  if [[ -n "${SCRATCH_ROOT:-}" ]]; then args+=(--scratch-root "$SCRATCH_ROOT"); fi
  if [[ "${PREFETCH_FEATURES:-1}" == 1 ]]; then args+=(--prefetch-features); fi
  args+=(--feature-cpu-slots "${FEATURE_CPU_SLOTS:-2}" --audit-interval "${AUDIT_INTERVAL:-120}")
fi
exec "$PYTHON" -u -m predict.baselines.foundation_suite "$PHASE" "${args[@]}" "$@"
