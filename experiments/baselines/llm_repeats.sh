#!/usr/bin/env bash
# Execute frozen jobs; concurrency changes do not change scientific settings.
set -euo pipefail
cd "$(dirname "$0")/../.."
study=${1:?Usage: bash experiments/baselines/llm_repeats.sh STUDY_DIRECTORY}
export PYTHON_BIN=${PYTHON_BIN:-python}
export SOURCE_ROOT=${SOURCE_ROOT:-/data1/tianang/Projects/TxAgent}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON_BIN" - "$study" <<'PY' | xargs -0 -r -n5 -P"${LLM_REPEAT_WORKERS:-4}" bash -c '
    kind=$1; root=$2; provider=$3; log=$4; source=$5
    if [[ -f "$log.done" ]]; then exit 0; fi
    : "${TXAGENT_PARCC_REPEAT_KEY:?Export the endpoint credential before launching}"
    if [[ ! -d "$root" ]]; then
        if [[ "$kind" == naive ]]; then
            "$PYTHON_BIN" -m predict.harnesses.branches --repeat-paper-baseline "$source" \
                --source-root "$SOURCE_ROOT" --run-name "${root##*/}" \
                --repeat-reasoning-effort low >>"$log.prepare.log" 2>&1 || { printf "1\n" >"$log.exit"; exit 1; }
        elif [[ "$kind" == molrag ]]; then
            "$PYTHON_BIN" -m predict.baselines.molrag repeat --run-root "$source" \
                --run-name "${root##*/}" --provider-config "$provider" >>"$log.prepare.log" 2>&1 || { printf "1\n" >"$log.exit"; exit 1; }
        else exit 2; fi
    fi
    if [[ "$kind" == naive ]]; then
        "$PYTHON_BIN" -m predict.harnesses.branches --run-paper-baseline "$root" \
            --provider-config "$provider" --parallelism 128 --max-stage-requeues 1 --retry-delay-s 1 >>"$log.log" 2>&1
    elif [[ "$kind" == molrag ]]; then
        "$PYTHON_BIN" -m predict.baselines.molrag run --run-root "$root" \
            --provider-config "$provider" \
            --parallelism "${MOLRAG_PARALLELISM:-128}" \
            --provider-max-inflight "${MOLRAG_PARALLELISM:-128}" >>"$log.log" 2>&1
    else exit 2; fi
    status=$?
    printf "%s\n" "$status" >"$log.exit"
    if [[ "$status" == 0 ]]; then touch "$log.done"; fi
    exit "$status"
' _
import json, sys
from pathlib import Path
study = Path(sys.argv[1]).resolve()
status = study/'aggregation_status.json'
if status.exists() and json.loads(status.read_text()).get('status') == 'complete':
    sys.stderr.write('Study already scored; use analysis.prediction.llm_repeats for offline verification.\n')
    sys.exit(0)
jobs = json.loads((study/'jobs.json').read_text())
if len({str(Path(j['root']).resolve()) for j in jobs}) != len(jobs):
    raise ValueError('Duplicate output roots would run concurrently')
(study/'logs').mkdir(exist_ok=True)
for job in jobs:
    for value in (job['kind'], job['root'], job['provider'],
                  str(study/'logs'/Path(job['root']).name), job['source']):
        sys.stdout.write(value + '\0')
PY
