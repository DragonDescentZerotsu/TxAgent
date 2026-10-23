#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?directory containing train and evaluation JSONL splits}"
: "${OUTPUT_ROOT:?fresh output directory}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT already exists; choose a fresh run directory: $OUTPUT_ROOT" >&2
  exit 2
fi
read -r -a methods <<< "${METHODS:-morgan_knn minimol_knn minimol_head}"
if (( ${#methods[@]} == 0 )); then
  echo "METHODS must select at least one baseline" >&2
  exit 2
fi
for method in "${methods[@]}"; do
  case "$method" in
    morgan_knn|minimol_knn|minimol_head) ;;
    *) echo "Unknown baseline method: $method" >&2; exit 2 ;;
  esac
done

python_bin="${PYTHON_BIN:-python}"
if [[ -n "${TXAGENT_MINIMOL_RUNTIME_ROOT:-}" ]]; then
  export PYTHONPATH="$TXAGENT_MINIMOL_RUNTIME_ROOT/pydeps:$TXAGENT_MINIMOL_RUNTIME_ROOT/runtime_41642328${PYTHONPATH:+:$PYTHONPATH}"
fi
evaluation_split="${EVALUATION_SPLIT:-test}"
case "${TRAINING_POOL:-conditioned}" in
  conditioned) condition_field=condition_group ;;
  tdc_only) condition_field="" ;;
  mixed) condition_field=model_condition ;;
  *) echo "TRAINING_POOL must be conditioned, tdc_only or mixed" >&2; exit 2 ;;
esac
condition_policy=same_condition_then_null
if [[ "${TRAINING_POOL:-conditioned}" == conditioned && "${TASK:-}" =~ ^(dili|carcinogens)$ ]]; then
  condition_policy=same_condition_then_null_then_all
fi
condition_policy="${CONDITION_POLICY:-$condition_policy}"
hidden_dim=512
depth=3
lr=3e-4
case "${TASK:-}" in
  bbb_martins) hidden_dim=2048; lr=1e-4 ;;
  ames) lr=1e-4 ;;
  dili) depth=4; lr=5e-4 ;;
  bioavailability_ma|skin_reaction|carcinogens) ;;
  *) echo "TASK must name one of the six paper tasks" >&2; exit 2 ;;
esac
embedding_cache="${EMBEDDING_CACHE_DIR:-$OUTPUT_ROOT/embeddings}"
minimol_args=(
  --data-dir "$DATA_DIR" --output-dir "$OUTPUT_ROOT/minimol_head"
  --embedding-cache-dir "$embedding_cache" --evaluation-split "$evaluation_split"
  --threshold-strategy valid_macro_f1
  --epochs 25 --ensemble-size 5 --seed 1 --hidden-dim "$hidden_dim" --depth "$depth"
  --dropout 0.1 --lr "$lr" --warmup 5 --weight-decay 1e-4
  --train-batch-size 32 --eval-batch-size 128 --embedding-batch-size 100
)
if [[ "${TRAIN_ALL:-0}" == 1 ]]; then minimol_args+=(--train-all --threshold-strategy fixed_0.5); fi
if [[ -n "${CONDITION_FIELD-$condition_field}" ]]; then
  minimol_args+=(--condition-field "${CONDITION_FIELD-$condition_field}")
fi
if [[ -n "${DEVICE:-}" ]]; then minimol_args+=(--device "$DEVICE"); fi
if [[ -n "${MINIMOL_SOURCE:-}" ]]; then minimol_args+=(--minimol-source "$MINIMOL_SOURCE"); fi
if [[ -n "${REUSE_EMBEDDING_CACHE_DIR:-}" ]]; then
  minimol_args+=(--reuse-embedding-cache-dir "$REUSE_EMBEDDING_CACHE_DIR")
fi

embeddings_ready=false
head_overrides=()
for arg in "$@"; do
  if [[ "$arg" != --force-embed ]]; then head_overrides+=("$arg"); fi
done
for method in "${methods[@]}"; do
  if [[ "$method" == morgan_knn ]]; then
    "$python_bin" -m predict.baselines.structure_knn.run \
      --data-dir "$DATA_DIR" --output-dir "$OUTPUT_ROOT/morgan_knn" \
      --evaluation-split "$evaluation_split" --reference-splits train \
      --k "${K:-3}" --condition-policy "$condition_policy"
    continue
  fi
  if [[ "$embeddings_ready" == false ]]; then
    "$python_bin" -m predict.baselines.minimol.run_bioavailability_ma \
      "${minimol_args[@]}" "$@" --embeddings-only
    embeddings_ready=true
  fi
  if [[ "$method" == minimol_knn ]]; then
    "$python_bin" -m predict.baselines.minimol.run_embedding_knn \
      --data-dir "$DATA_DIR" --embedding-cache-dir "$embedding_cache" \
      --output-dir "$OUTPUT_ROOT/minimol_knn" --evaluation-split "$evaluation_split" \
      --reference-splits train --k "${K:-3}" --condition-policy "$condition_policy"
  else
    "$python_bin" -m predict.baselines.minimol.run_bioavailability_ma \
      "${minimol_args[@]}" "${head_overrides[@]}"
  fi
done
