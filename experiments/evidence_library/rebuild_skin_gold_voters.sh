#!/usr/bin/env bash
# Build the evidence successor consumed by the cache publication steps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPLAY_ROOT=${1:?Export of pinned Skin Git inputs required}
RELEASE_ROOT=${RELEASE_ROOT:-$PWD/data/evidence_libraries/skin_reaction/v10_main_universe_v7}
SOURCE_ROOT=${SOURCE_ROOT:-$PWD/data/raw/starling/main_universe_skin_gold_voters_v1}
PRIOR_ROOT=${PRIOR_ROOT:-$PWD/data/evidence_libraries/skin_reaction/v10_main_universe_v6}
WORKERS=${WORKERS:-16}
GOLD_ROOT=$RELEASE_ROOT/inputs/gold
MEMBERS=$GOLD_ROOT/Skin_Reaction/v1/scaffold/voter_membership.parquet
LEVELS=$RELEASE_ROOT/inputs/levels

python -m data.processing.gold_labels.publish_gold_v1_context_membership \
    --tasks skin_reaction --skin-replay-root "$REPLAY_ROOT" --output-root "$GOLD_ROOT"
python -m data.processing.evidence_library.import_main_source_universe \
    --tasks skin_reaction --base-root "$PWD/data/raw/starling/main_universe_v1" \
    --voter-membership "$MEMBERS" --output-root "$SOURCE_ROOT"
python -m data.processing.evidence_library.publish_gold_level_mappings \
    --skin-voter-membership "$MEMBERS" --output-root "$LEVELS"
record_args=(--out-dir "$RELEASE_ROOT" --authoritative-source-root "$SOURCE_ROOT"
    --stage1-protected-voter-contract "$LEVELS/stage1_voter_protection.parquet"
    --workers "$WORKERS" --validation-level full --cache-mode auto)
python -m data.processing.evidence_library.versions.v10.tasks.skin_reaction.build_normalized_starling_evidence_library \
    "${record_args[@]}" --from-stage source --through-stage normalize
python -m data.processing.evidence_library.publish_gold_level_mappings \
    --skin-voter-membership "$MEMBERS" --output-root "$LEVELS" \
    --canonical-records "$RELEASE_ROOT/02_canonicalized/records.parquet"
mkdir -p "$RELEASE_ROOT/reviews/assay_transfer_record_pruning_v4"
cp "$PRIOR_ROOT/reviews/assay_transfer_record_pruning_v4/reviews.jsonl" \
    "$RELEASE_ROOT/reviews/assay_transfer_record_pruning_v4/.review_cache.jsonl"
python -m data.processing.evidence_library.versions.v10.build_current_release \
    --task skin_reaction --normalized-root "$RELEASE_ROOT" \
    --authoritative-source-root "$SOURCE_ROOT" \
    --stage1-protected-voter-contract "$LEVELS/stage1_voter_protection.parquet" \
    --voter-membership "$MEMBERS" --reviewed-level-mapping "$LEVELS/level_mapping" \
    --frozen-prior-pruning-manifest "$PRIOR_ROOT/reviews/assay_transfer_record_pruning_v4/manifest.json" \
    --from-stage normalize --validation-level full --cache-mode auto --workers "$WORKERS"
