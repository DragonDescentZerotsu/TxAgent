#!/usr/bin/env bash
set -euo pipefail
cd /vast/projects/myatskar/design-documents/joseph/TxAgent
PYTHON=${TXAGENT_PYTHON:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/bin/python}
STUDY=${TXAGENT_STUDY:-outputs/analysis/retrieval_cache/molecular_policy_20261015T195506Z}
REMOTE=/data1/joseph/txagent_molecular_policy_20261015T195506Z/TxAgent
REMOTE_PYTHON=/data1/joseph/miniconda3/envs/openrlhf/bin/python
export HF_HOME=/vast/projects/myatskar/design-documents/hf_home
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MODULE=predict.retrieval.assay_reranking.rebuild_policy
role=${1:?Use prepare TASK [existing preparation PID], collect TASK, node002, dgx018 or dgx009}
tasks=(skin_reaction bbb_martins ames dili carcinogens)

if [[ "$role" == prepare || "$role" == collect ]]; then
    task=${2:?Task required}
    if [[ "$role" == prepare && -n ${3:-} ]]; then
        while kill -0 "$3" 2>/dev/null; do sleep 10; done
    fi
    jobs=()
    for benchmark in gold_v1 tdc; do
        for policy in parent_disjoint scaffold_disjoint; do
            jobs+=(--job "${benchmark}_${task}_${policy}")
        done
    done
    if [[ "$role" == collect ]]; then
        workset="$STUDY/score_worksets/${task}_weighted"
        for resource in node002 dgx018 dgx009; do
            while [[ ! -f "$workset/dispatch/$resource.done" ]]; do sleep 10; done
        done
        "$PYTHON" -m "$MODULE" reuse --study "$STUDY" --workers 2 "${jobs[@]}" \
            --score-reuse-root "$workset/$task/RELEASE_INDEX.json"
        "$PYTHON" -m "$MODULE" finalize --study "$STUDY" --workers 2 --ready-only "${jobs[@]}"
        exec "$PYTHON" -m "$MODULE" prepare --study "$STUDY" --workers 2 "${jobs[@]}"
    fi
    "$PYTHON" -m "$MODULE" prepare --study "$STUDY" --workers 2 "${jobs[@]}"
    "$PYTHON" -m "$MODULE" reuse --study "$STUDY" --workers 2 "${jobs[@]}"
    exec "$PYTHON" -m "$MODULE" worksets --study "$STUDY" "${jobs[@]}" \
        --output-root "$STUDY/score_worksets/${task}_weighted"
fi

case "$role" in
    node002) shards=(0 1 2 3 4 5 6 7); devices=0,1,2,3,4,5,6,7 ;;
    dgx018) shards=(8 9 10 11); allocation=8994724; fraction=0.17; devices=0 ;;
    dgx009) shards=(12 13 14 15); allocation=8997395; fraction=0.6; devices=0 ;;
    *) exit 2 ;;
esac
if [[ -n ${TXAGENT_PREVIOUS_PID:-} ]]; then
    if [[ "$role" == node002 ]]; then
        while ssh -o BatchMode=yes node002 "kill -0 $TXAGENT_PREVIOUS_PID 2>/dev/null"; do sleep 5; done
    else
        while kill -0 "$TXAGENT_PREVIOUS_PID" 2>/dev/null; do sleep 5; done
    fi
fi
args=(--gpus "$devices" --num-shards 16 --batch-size 32)
for shard in "${shards[@]}"; do args+=(--shard "$shard"); done
while ((${#tasks[@]})); do
    remaining=()
    for task in "${tasks[@]}"; do
        workset="$STUDY/score_worksets/${task}_weighted"
        if [[ ! -f "$workset/study.json" ]]; then
            remaining+=("$task")
            continue
        fi
        mkdir -p "$workset/dispatch"
        [[ ! -f "$workset/dispatch/$role.done" ]] || continue
        date -u +'%FT%TZ' > "$workset/dispatch/$role.started"
        if [[ "$role" == node002 ]]; then
            rg --files --hidden --no-ignore "$workset" | rg '/(study.json|VERSION.json|prompts.parquet)$' > "$workset/dispatch/node002_inputs.txt"
            rsync -a --files-from="$workset/dispatch/node002_inputs.txt" ./ "node002:$REMOTE/"
            ssh -o BatchMode=yes node002 "cd $REMOTE && HF_HOME=/data1/joseph/txagent_scaffold_disjoint_parent100_20261013T084030Z/hf_home OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $REMOTE_PYTHON -m $MODULE score --study $workset ${args[*]} --gpu-memory-fraction 0.6"
            rsync -a --include='*/' --include='*.jsonl' --exclude='*' \
                "node002:$REMOTE/$workset/$task/scaffold/test/" "$workset/$task/scaffold/test/"
        else
            srun --jobid="$allocation" --overlap --exact -N1 -n1 -c4 --mem=32G \
                env HF_HOME="$HF_HOME" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
                "$PYTHON" -m "$MODULE" score --study "$workset" "${args[@]}" --gpu-memory-fraction "$fraction"
        fi
        date -u +'%FT%TZ' > "$workset/dispatch/$role.done"
    done
    tasks=("${remaining[@]}")
    ((${#tasks[@]} == 0)) || sleep 10
done
