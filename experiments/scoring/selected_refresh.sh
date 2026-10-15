#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RUN_ROOT:?fresh durable semantic scoring root}"
task=${TASK:-bioavailability_ma}
python_bin=${PYTHON_BIN:-python}
profile=${PROVIDER_CONFIG:-predict/api_client/providers/gpt6_luna_openrouter_standard_medium_64.json}
case ${STAGE:-prepare} in
  prepare)
    if [[ ! -f $RUN_ROOT/inputs/inputs.json ]]; then
      "$python_bin" -m optimization.semantic_buckets.bucket_cards --task "$task" \
        --release CURRENT --output "$RUN_ROOT/inputs"
    fi
    seed=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_bindings"]["world"]["path"])' "$RUN_ROOT/inputs/inputs.json")
    INPUT_MANIFEST="$RUN_ROOT/inputs/inputs.json" RUN_ROOT="$RUN_ROOT/scoring" \
      PROVIDER_CONFIG="$profile" PASSES=2 PYTHON_BIN="$python_bin" \
      bash experiments/scoring/sliding_auto.sh --seed-weights "$seed" \
        --max-chains "${MAX_CHAINS:-4}" --pass2-fanout "${FANOUT:-2}" \
        --pass2-average "${AVERAGE:-2}" \
        --reasoning-effort "${REASONING_EFFORT:-medium}" --max-tokens 20480 "$@"
    ;;
  run)
    "$python_bin" -m optimization.semantic_buckets.bucket_scoring run --root "$RUN_ROOT/scoring"
    "$python_bin" -m optimization.semantic_buckets.bucket_scoring collect --root "$RUN_ROOT/scoring"
    ;;
  publish)
    "$python_bin" -m optimization.semantic_buckets.publication refresh-weights \
      --task "$task" --scoring-root "$RUN_ROOT/scoring" --output "$RUN_ROOT/publication" \
      --generation "$(basename "$RUN_ROOT")" --activate
    ;;
  *) echo 'STAGE must be prepare, run, or publish' >&2; exit 2 ;;
esac
