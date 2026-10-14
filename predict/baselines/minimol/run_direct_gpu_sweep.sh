#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then
  echo "usage: $0 task_name data_dir embedding_cache_dir output_root [gpu_csv] [trainer arguments...]" >&2
  exit 2
fi
task_name=$1; data_dir=$2; embedding_cache_dir=$3; output_root=$4
 gpu_csv=${5:-4,5,6,7}
if (( $# >= 5 )); then shift 5; else shift 4; fi
python_bin=${PYTHON_BIN:-python}
IFS=',' read -r -a gpus <<< "$gpu_csv"
(( ${#gpus[@]} > 0 )) || { echo "gpu_csv must not be empty" >&2; exit 2; }
for gpu in "${gpus[@]}"; do [[ -n $gpu ]] || exit 2; done
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l) -eq ${#gpus[@]} ]] || { echo "duplicate GPUs" >&2; exit 2; }
configs=(
  "512 3 0.0001" "512 3 0.0003" "512 4 0.0003" "512 4 0.0005"
  "1024 3 0.0003" "1024 3 0.0005" "1024 4 0.0001" "1024 4 0.0005"
  "2048 3 0.0001" "2048 4 0.0003" "2048 4 0.0005"
)
worker() {
  local worker_index=$1 gpu=$2 i hidden depth lr run_id out_dir status code
  shift 2
  for (( i=worker_index; i<${#configs[@]}; i+=${#gpus[@]} )); do
    read -r hidden depth lr <<< "${configs[i]}"
    run_id=h${hidden}_d${depth}_lr${lr/./p}
    out_dir=$output_root/$task_name/$run_id
    command=(env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m predict.baselines.minimol.run_bioavailability_ma
      --data-dir "$data_dir" --output-dir "$out_dir" --embedding-cache-dir "$embedding_cache_dir"
      --hidden-dim "$hidden" --depth "$depth" --lr "$lr" --epochs 25 --ensemble-size 5 "$@")
    if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; continue; fi
    code=0; status=cached
    if [[ ! -f $out_dir/metrics.json || ${FORCE:-0} == 1 ]]; then
      mkdir -p "$out_dir"
      status=ok
      if "${command[@]}" > "$out_dir/run.log" 2>&1; then
        if [[ ! -f $out_dir/metrics.json ]]; then code=1; status=failed; fi
      else code=$?; status=failed; fi
    fi
    printf '%s\t%s\t%s\n' "$run_id" "$status" "$code" >> "$status_root/$worker_index.tsv"
  done
}
if [[ ${DRY_RUN:-0} == 1 ]]; then
  for i in "${!gpus[@]}"; do worker "$i" "${gpus[i]}" "$@"; done
  exit 0
fi
status_root=$(mktemp -d)
trap 'rm -rf "$status_root"' EXIT
pids=()
for i in "${!gpus[@]}"; do worker "$i" "${gpus[i]}" "$@" & pids+=("$!"); done
worker_failed=0
for pid in "${pids[@]}"; do wait "$pid" || worker_failed=1; done
(( worker_failed == 0 )) || exit 1
cat "$status_root"/*.tsv > "$status_root/all.tsv"
"$python_bin" -m predict.baselines.minimol.summarize_sweep --task-name "$task_name" --data-dir "$data_dir" \
  --output-root "$output_root" --status-file "$status_root/all.tsv"
