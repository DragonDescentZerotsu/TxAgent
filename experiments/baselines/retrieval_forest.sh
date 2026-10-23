#!/usr/bin/env bash
set -euo pipefail

: "${PREPARED:?retrieval feature directory}"
: "${OUTPUT_ROOT:?fresh model output directory}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT already exists; choose a fresh run directory: $OUTPUT_ROOT" >&2
  exit 2
fi
read -r -a policies <<< "${POLICIES:-0 1}"
read -r -a feature_sets <<< "${FEATURE_SETS:-direct direct_indirect}"
for policy in "${policies[@]}"; do
  case "$policy" in 0|1) ;; *) echo "POLICIES must contain 0 or 1" >&2; exit 2 ;; esac
done
for feature_set in "${feature_sets[@]}"; do
  case "$feature_set" in direct|direct_indirect) ;; *) echo "Unknown FEATURE_SETS entry" >&2; exit 2 ;; esac
done
python_bin="${PYTHON_BIN:-python}"
workers="${WORKERS:-2}"
if [[ ! -f "$PREPARED/manifest.json" ]]; then
  : "${DATA_DIR:?directory containing frozen train/valid/test JSONL splits}"
  : "${SOURCE_PREPARED:?prepared MiniMol auxiliary targets}"
  "$python_bin" -m predict.baselines.retrieval_forest prepare \
    --prepared "$PREPARED" --data-dir "$DATA_DIR" --source-prepared "$SOURCE_PREPARED" \
    --condition-policy "${CONDITION_POLICY:-same_condition_then_null}" --workers "$workers"
fi
for feature_set in "${feature_sets[@]}"; do
  for policy in "${policies[@]}"; do
    output="$OUTPUT_ROOT/$feature_set/policy_$policy"
    for max_features in sqrt 0.3; do
      for leaf in 1 5 20; do
        "$python_bin" -m predict.baselines.retrieval_forest fit \
          --prepared "$PREPARED" --feature-set "$feature_set" --policy "$policy" \
          --trees 100 --seeds 1 2 3 4 5 --max-features "$max_features" --min-leaf "$leaf" \
          --workers "$workers" --output "$output/candidates/features_${max_features}_leaf_$leaf"
      done
    done
    "$python_bin" -m predict.baselines.retrieval_forest finalize \
      --prepared "$PREPARED" --feature-set "$feature_set" --policy "$policy" \
      --expected-candidates 6 --workers "$workers" --output "$output"
  done
done
