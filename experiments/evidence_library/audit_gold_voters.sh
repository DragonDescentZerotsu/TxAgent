#!/usr/bin/env bash
# Optional first argument: export of the pinned original Skin Git inputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
args=(--verify-only --report "${REPORT:-outputs/analysis/evidence_library/gold_voter_audit.json}")
if [[ $# -gt 0 ]]; then
    args+=(--skin-replay-root "$1")
fi
python -m data.processing.gold_labels.publish_gold_v1_context_membership "${args[@]}"
