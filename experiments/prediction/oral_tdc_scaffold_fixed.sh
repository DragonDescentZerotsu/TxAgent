#!/usr/bin/env bash
set -euo pipefail
: "${RUN_NAME:?fresh run identity}" "${SELECTION:?regenerated combined optimizer manifest}"

export TASK=bioavailability_ma BENCHMARK=tdc SUBSET=test
export QUERIES=data/gold_labels/TDC/Bioavailability_Ma/test_molecule_condition_labels.jsonl
export CACHE_BUNDLE=predict/retrieval/assay_reranking/flat_v5_manifests/scaffold_disjoint_parent100_v1/catalog.json
export PRIOR_ROOT=${PRIOR_ROOT:-outputs/predictions/query-prior/f20_evidence_calibrated_v2/tdc_v1_test_upstream_deepseek_openrouter_20260923_r1/tdc/bioavailability_ma/test/query-prior}

bash experiments/prediction/manual_uids.sh \
  --prompt final_full_flat --layout level-grouped \
  --query-prior cached --prior-root "$PRIOR_ROOT" \
  --provider-pool-config "${PROVIDER_POOL_CONFIG:-predict/api_client/providers/deepseek_v4_flash_0731_openrouter_auto_128_high.json}" \
  --reasoning-effort high \
  --parallelism "${PARALLELISM:-128}" --execution-mode throughput --max-tokens "${MAX_TOKENS:-131072}" \
  --temperature "${TEMPERATURE:-0}" --skip-existing "$@"
