#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <task_name> <data_dir> <embedding_cache_dir> <output_root> [gpu_csv]" >&2
  exit 2
fi

task_name="$1"
data_dir="$2"
embedding_cache_dir="$3"
output_root="$4"
gpu_csv="${5:-4,5,6,7}"

conda_bin="/data1/tianang/anaconda3/condabin/conda"
IFS=',' read -r -a gpus <<< "$gpu_csv"

configs=(
  "512 3 0.0001"
  "512 3 0.0003"
  "512 4 0.0003"
  "512 4 0.0005"
  "1024 3 0.0003"
  "1024 3 0.0005"
  "1024 4 0.0001"
  "1024 4 0.0005"
  "2048 3 0.0001"
  "2048 4 0.0003"
  "2048 4 0.0005"
)

mkdir -p "${output_root}/${task_name}"

running=0
gpu_idx=0
for cfg in "${configs[@]}"; do
  read -r hidden depth lr <<< "$cfg"
  lr_id="${lr/./p}"
  run_id="h${hidden}_d${depth}_lr${lr_id}"
  out_dir="${output_root}/${task_name}/${run_id}"
  mkdir -p "$out_dir"
  gpu="${gpus[$gpu_idx]}"
  gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))

  echo "[direct_sweep] launch task=${task_name} run=${run_id} gpu=${gpu}"
  env CUDA_VISIBLE_DEVICES="$gpu" "$conda_bin" run -n intern python -m predict.baselines.minimol.run_bioavailability_ma \
    --data-dir "$data_dir" \
    --output-dir "$out_dir" \
    --embedding-cache-dir "$embedding_cache_dir" \
    --hidden-dim "$hidden" \
    --depth "$depth" \
    --lr "$lr" \
    --epochs 25 \
    --ensemble-size 5 \
    > "${out_dir}/run.log" 2>&1 &

  running=$((running + 1))
  if [ "$running" -ge "${#gpus[@]}" ]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "[direct_sweep] complete task=${task_name}"
