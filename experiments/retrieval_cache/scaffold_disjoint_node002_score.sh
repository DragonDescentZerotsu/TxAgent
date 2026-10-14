#!/usr/bin/env bash
set -euo pipefail

TXAGENT_REMOTE=/data1/joseph/txagent_scaffold_disjoint_parent100_20261013T084030Z
TXAGENT_RUN=outputs/analysis/retrieval_cache/scaffold_disjoint_parent100_20261013T084030Z
TXAGENT_PYTHON=/data1/joseph/miniconda3/envs/openrlhf/bin/python
cd "$TXAGENT_REMOTE/TxAgent"
export HF_HOME="$TXAGENT_REMOTE/hf_home"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ -f "$TXAGENT_REMOTE/logs/scoring.exit" ]]; then
    mv "$TXAGENT_REMOTE/logs/scoring.exit" "$TXAGENT_REMOTE/logs/scoring.exit.$(date -u +%Y%m%dT%H%M%SZ)"
fi
trap 'printf "%s\n" "$?" > "$TXAGENT_REMOTE/logs/scoring.exit"' EXIT

for subset in test valid; do
    for task in bbb_martins dili carcinogens ames bioavailability_ma skin_reaction; do
        [[ ! -f "$TXAGENT_REMOTE/staged/$task.$subset.scores-complete" ]] || continue
        while [[ ! -f "$TXAGENT_REMOTE/staged/$task.ready" ]]; do sleep 20; done
        printf 'Scoring %s %s at %s\n' "$task" "$subset" "$(date -u +%FT%TZ)"
        "$TXAGENT_PYTHON" -u "$TXAGENT_RUN/run.py" score --subset "$subset" \
            --tasks "$task" --gpus 0,1,2,3,4,5,6,7 --batch-size 32 \
            --gpu-memory-fraction 0.6 --defer-finalization
        touch "$TXAGENT_REMOTE/staged/$task.$subset.scores-complete"
    done
done
touch "$TXAGENT_REMOTE/staged/all.scores-complete"
