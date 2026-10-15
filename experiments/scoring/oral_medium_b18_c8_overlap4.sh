#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Four anchors per advancing window: eight with the two-window quorum, four with one chain.
TASK=bioavailability_ma MAX_CHAINS=8 REASONING_EFFORT=medium FANOUT=2 AVERAGE=2 \
  PROVIDER_CONFIG=predict/api_client/providers/gpt6_luna_openrouter_standard_medium_64.json \
  exec bash experiments/scoring/selected_refresh.sh \
    --pass2-batch-size 18 --overlap 4 "$@"
