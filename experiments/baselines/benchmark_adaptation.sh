#!/usr/bin/env bash
# Run on one reserved, idle GPU; CUDA_VISIBLE_DEVICES selects it.
set -u
run=${1:?existing foundation run root}
runtime=${2:?foundation runtime root}
output=${3:?new diagnostic output root}
python=${PYTHON:-python}
if [[ -e "$output" ]]; then
    echo "Use a new output directory; preserve existing measurements and logs." >&2
    exit 1
fi
mkdir -p "$output"
failed=0
for model in unimol chemberta molformer smited grover kermt chemeleon; do
    for mode in full_fp32 full_bf16 lora_bf16; do
        "$python" -u -m predict.baselines.benchmark_adaptation \
            --run "$run" --runtime-root "$runtime" --output "$output/$model.$mode.json" \
            --model "$model" --mode "$mode" > "$output/$model.$mode.log" 2>&1 || failed=1
    done
done
exit "$failed"
