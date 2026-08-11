#!/usr/bin/env bash
set -euo pipefail

attempt="${1:-manual}"
runtime_root=/local/tianang/txagent_rl_lora
nemo_root=/local/tianang/nemo-rl
project_root=/data1/tianang/Projects/TxAgent
policy_worker_root="$runtime_root/worker_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
policy_python_bin="$(dirname "$(readlink -f "$policy_worker_root/bin/python")")"
cuda_toolkit_root="$policy_worker_root/lib/python3.13/site-packages/nvidia/cu13"
loader_shim_source="$project_root/tools/chembl_tool/paper_experiments/rl_lora/runtime/deny_cuda12_dlopen.c"
loader_shim_dir="$runtime_root/runtime_shims"
loader_shim="$loader_shim_dir/libdeny_cuda12_dlopen.so"
config="${2:-$project_root/tools/chembl_tool/paper_experiments/rl_lora/configs/grpo_gpt_oss_120b_lora_smoke.yaml}"
log="$runtime_root/logs/smoke_driver_${attempt}.log"

mkdir -p \
  "$runtime_root/home" \
  "$runtime_root/cache" \
  "$runtime_root/tmp" \
  "$runtime_root/pip_cache" \
  "$runtime_root/uv_cache" \
  "$runtime_root/triton_cache" \
  "$runtime_root/torchinductor_cache" \
  "$runtime_root/vllm_cache" \
  "$runtime_root/torch_extensions" \
  "$runtime_root/cuda_cache" \
  "$loader_shim_dir" \
  "$runtime_root/logs"

if [[ ! -f "$loader_shim" || "$loader_shim_source" -nt "$loader_shim" ]]; then
  gcc -shared -fPIC -O2 -Wall -Wextra -Werror \
    -o "$loader_shim" "$loader_shim_source" -ldl
fi

export HOME="$runtime_root/home"
export PYTHONPATH="$project_root:$project_root/tools/chembl_tool/paper_experiments"
export TORCH_CUDA_ARCH_LIST=8.0
# The long-lived tmux server on these nodes can retain CUDA 12.8 paths.  This
# worker environment is PyTorch/Transformer Engine cu130, so bind its bundled
# CUDA 13 toolkit explicitly instead of inheriting the tmux server's toolkit.
export CUDA_HOME="$cuda_toolkit_root"
export PATH="$policy_worker_root/bin:$policy_python_bin:$CUDA_HOME/bin:${PATH:-/usr/bin:/bin}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib"
export LD_PRELOAD="$loader_shim${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NEMO_RL_ROOT="$nemo_root"
export NEMO_RL_VENV_DIR="$runtime_root/worker_venvs"
export NRL_MEGATRON_CHECKPOINT_DIR="$runtime_root/megatron_cache"
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.05
export NRL_JOB_START_EPOCH="$(date +%s)"
export HF_HOME=/data1/tianang/cache
export XDG_CACHE_HOME="$runtime_root/cache"
export TMPDIR="$runtime_root/tmp"
export PIP_CACHE_DIR="$runtime_root/pip_cache"
export UV_CACHE_DIR="$runtime_root/uv_cache"
export TRITON_CACHE_DIR="$runtime_root/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor_cache"
export VLLM_CACHE_ROOT="$runtime_root/vllm_cache"
export TORCH_EXTENSIONS_DIR="$runtime_root/torch_extensions"
export CUDA_CACHE_PATH="$runtime_root/cuda_cache"

cd "$nemo_root"
"$nemo_root/.venv/bin/python" \
  "$project_root/tools/chembl_tool/paper_experiments/rl_lora/run_grpo.py" \
  --config "$config" 2>&1 | tee "$log"
