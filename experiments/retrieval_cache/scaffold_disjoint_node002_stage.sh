#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TXAGENT_REPO=$PWD
TXAGENT_RUN=${TXAGENT_RUN:-outputs/analysis/retrieval_cache/scaffold_disjoint_parent100_20261013T084030Z}
TXAGENT_REMOTE=${TXAGENT_REMOTE:?Set TXAGENT_REMOTE to the remote staging root}
TXAGENT_REMOTE_HOST=${TXAGENT_REMOTE_HOST:-node002}
mkdir -p "$TXAGENT_RUN/logs"
trap 'printf "%s\n" "$?" > "$TXAGENT_RUN/logs/node002_stage.exit"' EXIT
ssh -n -o BatchMode=yes "$TXAGENT_REMOTE_HOST" "mkdir -p '$TXAGENT_REMOTE/staged'"

for task in bbb_martins dili carcinogens ames bioavailability_ma skin_reaction; do
    printf 'Staging %s at %s\n' "$task" "$(date -u +%FT%TZ)"
    while IFS=$'\t' read -r row_task kind source destination existing; do
        [[ "$row_task" == "$task" && "$kind" == checkpoint ]] || continue
        target="$TXAGENT_REMOTE/$destination"
        if [[ "$existing" != - ]]; then
            ssh -n -o BatchMode=yes "$TXAGENT_REMOTE_HOST" "mkdir -p '$(dirname "$target")'; if [ ! -e '$target' ]; then ln -s '$existing' '$target'; fi"
        else
            rsync -aL --partial-dir=.rsync-partial --info=stats2 -e 'ssh -o BatchMode=yes' \
                --rsync-path="mkdir -p '$target' && rsync" "$source/" "$TXAGENT_REMOTE_HOST:$target/"
        fi
    done < "$TXAGENT_RUN/remote_stage.tsv"
    rsync -a --partial-dir=.rsync-partial --info=stats2 -e 'ssh -o BatchMode=yes' \
        --files-from=<(awk -F '\t' -v task="$task" '$1 == task && $2 == "repo_file" {print $3}' "$TXAGENT_RUN/remote_stage.tsv") \
        "$TXAGENT_REPO/" "$TXAGENT_REMOTE_HOST:$TXAGENT_REMOTE/TxAgent/"
    ssh -n -o BatchMode=yes "$TXAGENT_REMOTE_HOST" "touch '$TXAGENT_REMOTE/staged/$task.ready'"
    printf 'Ready %s at %s\n' "$task" "$(date -u +%FT%TZ)"
done
ssh -n -o BatchMode=yes "$TXAGENT_REMOTE_HOST" "touch '$TXAGENT_REMOTE/staged/all.ready'"
