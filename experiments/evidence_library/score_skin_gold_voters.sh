#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SCORER_PYTHON=${SCORER_PYTHON:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/bin/python}
SCORING_ROOT=${SCORING_ROOT:-data/caches/assay_reranking/active/flat_v5/gold_v1/skin_reaction/l1/assay_transfer/v9_0_2/scaffold_disjoint_parent100_v1/scoring}
GOLD_ROOT=${GOLD_ROOT:-$PWD/data/evidence_libraries/skin_reaction/v10_main_universe_v7/inputs/gold/Skin_Reaction/v1/scaffold}
phase=${1:-score}
for split in valid test; do
    args=(--task skin_reaction --lineage v9_scaffold --subset "$split" --cache-root "$SCORING_ROOT/$split")
    case "$phase" in
        prepare)
            args+=(--train-path "$GOLD_ROOT/train_molecule_condition_labels.jsonl"
                --query-path "$GOLD_ROOT/${split}_molecule_condition_labels.jsonl")
            if [[ -n ${REUSE_ROOT:-} ]]; then args+=(--reuse-root "$REUSE_ROOT"); fi
            ;;
        score) args+=(--shard-index 0 --num-shards 1 --batch-size 4 --device 0) ;;
        finalize) args+=(--num-shards 1) ;;
        validate) ;;
        *) echo "Usage: $0 [prepare|score|finalize|validate]" >&2; exit 2 ;;
    esac
    "$SCORER_PYTHON" -m predict.retrieval.assay_reranking.v9 "$phase" "${args[@]}"
done
