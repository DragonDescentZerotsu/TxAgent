#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TXAGENT_REPO=$PWD
TXAGENT_RUN=${TXAGENT_RUN:-outputs/analysis/retrieval_cache/scaffold_disjoint_parent100_20261013T084030Z}
TXAGENT_REMOTE=${TXAGENT_REMOTE:?Set TXAGENT_REMOTE to the remote staging root}
TXAGENT_REMOTE_HOST=${TXAGENT_REMOTE_HOST:-node002}
TXAGENT_PYTHON=${TXAGENT_PYTHON:-python}
mkdir -p "$TXAGENT_RUN/logs"
trap 'printf "%s\n" "$?" > "$TXAGENT_RUN/logs/node002_collect.exit"' EXIT

while ! ssh -n -o BatchMode=yes "$TXAGENT_REMOTE_HOST" "test -f '$TXAGENT_REMOTE/staged/all.scores-complete'"; do
    if ssh -n -o BatchMode=yes "$TXAGENT_REMOTE_HOST" "test -f '$TXAGENT_REMOTE/logs/scoring.exit'"; then
        printf 'Remote scoring exited before completion; inspect scoring.log\n' >&2
        exit 1
    fi
    sleep 30
done
rsync -a --partial-dir=.rsync-partial --prune-empty-dirs \
    --include='*/' --include='*.jsonl' --exclude='*' -e 'ssh -o BatchMode=yes' \
    "$TXAGENT_REMOTE_HOST:$TXAGENT_REMOTE/TxAgent/data/caches/assay_reranking/active/flat_v5/" \
    "$TXAGENT_REPO/data/caches/assay_reranking/active/flat_v5/"

TXAGENT_FROZEN=$(mktemp -d /tmp/txagent-scaffold-finalize.XXXXXX)
tar -xzf "$TXAGENT_RUN/scorer_source_performance_v2.tar.gz" -C "$TXAGENT_FROZEN"
ln -s "$TXAGENT_REPO/data/caches" "$TXAGENT_FROZEN/data/caches"
ln -s "$TXAGENT_REPO/data/gold_labels" "$TXAGENT_FROZEN/data/gold_labels"
for task in bbb_martins bioavailability_ma ames dili carcinogens skin_reaction; do
    release=v10_main_universe_v3
    [[ "$task" != skin_reaction ]] || release=v10_main_universe_v5
    mkdir -p "$TXAGENT_FROZEN/data/evidence_libraries/$task"
    ln -s "$TXAGENT_REPO/data/evidence_libraries/$task/$release" \
        "$TXAGENT_FROZEN/data/evidence_libraries/$task/$release"
done
mkdir -p "$TXAGENT_FROZEN/$TXAGENT_RUN"
cp "$TXAGENT_RUN/run.py" "$TXAGENT_RUN/manifest.json" "$TXAGENT_FROZEN/$TXAGENT_RUN/"
"$TXAGENT_PYTHON" "$TXAGENT_FROZEN/$TXAGENT_RUN/run.py" finalize --subset both
"$TXAGENT_PYTHON" "$TXAGENT_FROZEN/$TXAGENT_RUN/run.py" publish --publication-root "$TXAGENT_REPO"
"$TXAGENT_PYTHON" "$TXAGENT_RUN/status.py"
touch "$TXAGENT_RUN/cache_build.complete"
