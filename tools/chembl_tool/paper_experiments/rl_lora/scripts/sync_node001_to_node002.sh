#!/usr/bin/env bash
set -euo pipefail

source_host="${1:-node001}"
runtime_root=/local/tianang/txagent_rl_lora
nemo_root=/local/tianang/nemo-rl
uv_env_name=nemo-rl-cp3.13.14-a1238f07b8e31921

mkdir -p \
  "$nemo_root" \
  /local/tianang/uv-python \
  "/local/tianang/uv-cache/environments-v2/$uv_env_name" \
  "$runtime_root/worker_venvs" \
  "$runtime_root/megatron_cache" \
  "$runtime_root/data" \
  "$runtime_root/logs"

rsync -a --partial --info=progress2 --exclude=.venv \
  "$source_host:/local/tianang/nemo-rl/" "$nemo_root/"
rsync -a --partial --info=progress2 \
  "$source_host:/local/tianang/uv-python/" /local/tianang/uv-python/
rsync -a --partial --info=progress2 \
  "$source_host:/local/tianang/uv-cache/environments-v2/$uv_env_name/" \
  "/local/tianang/uv-cache/environments-v2/$uv_env_name/"
ln -sfn "/local/tianang/uv-cache/environments-v2/$uv_env_name" "$nemo_root/.venv"

rsync -a --partial --info=progress2 \
  "$source_host:$runtime_root/worker_venvs/" "$runtime_root/worker_venvs/"
rsync -a --partial --info=progress2 \
  "$source_host:$runtime_root/megatron_cache/" "$runtime_root/megatron_cache/"
rsync -a --partial --info=progress2 \
  "$source_host:$runtime_root/data/" "$runtime_root/data/"

"$nemo_root/.venv/bin/python" -c 'import nemo_rl, ray, torch; print(torch.__version__)'
"$runtime_root/worker_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python" \
  -c 'import megatron.core, torch, transformer_engine; print(torch.__version__)'
du -sh "$nemo_root" "$runtime_root/worker_venvs" "$runtime_root/megatron_cache"
