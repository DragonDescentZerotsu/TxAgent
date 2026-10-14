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
evaluation_split="${EVALUATION_SPLIT:-test}"
condition_policy="${CONDITION_POLICY:-same_condition_then_null}"
embedding_cache="${EMBEDDING_CACHE_DIR:-$OUTPUT_ROOT/embeddings}"
minimol_args=(
  --data-dir "$DATA_DIR" --output-dir "$OUTPUT_ROOT/minimol_head"
  --embedding-cache-dir "$embedding_cache" --evaluation-split "$evaluation_split"
  --train-all --threshold-strategy fixed_0.5
  --epochs 25 --ensemble-size 5 --seed 1 --hidden-dim 512 --depth 3
  --dropout 0.1 --lr 3e-4 --warmup 5 --weight-decay 1e-4
  --train-batch-size 32 --eval-batch-size 128 --embedding-batch-size 100
)
if [[ -n "${CONDITION_FIELD-condition_group}" ]]; then
  minimol_args+=(--condition-field "${CONDITION_FIELD-condition_group}")
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
