#!/usr/bin/env bash
set -euo pipefail
: "${CACHE_CONFIG:?cache YAML}" "${OUTPUT_ROOT:?fresh analysis directory}"
"${PYTHON_BIN:-python}" -m analysis.prediction.progressive_pair_buckets --cache-config "$CACHE_CONFIG" --output-dir "$OUTPUT_ROOT" \
  --task-levels bbb_martins:5 bioavailability_ma:6 --subset valid --modes morgan assay-transfer \
  --pools all assay-transfer-trained --molecule-limit 10 --l1-limit 10 --later-limit 25 "$@"
