#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_profile.sh"

OUT_DIR="outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution"
ELIGIBLE_RECORDS="${TXAGENT_STARLING_ROOT}/datasets/eligible/assay_transfer_soft_evidence_v6_5/records.parquet"
SUPPORT_BASE="${TXAGENT_STARLING_ROOT}/datasets/base/canonical_endpoints_v3"

cd "${TXAGENT_PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

exec "${TXAGENT_PYTHON}" \
  -m tools.chembl_tool.tasks.bioavailability_ma.reranking.build_starling_in_distribution_library \
  --hf-cleaned-dir "${ELIGIBLE_RECORDS}" \
  --support-text-base "${SUPPORT_BASE}" \
  --out-dir "${OUT_DIR}" \
  --workers "${TXAGENT_LIBRARY_WORKERS}" \
  --progress-every 10000

