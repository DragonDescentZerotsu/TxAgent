#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REGISTRY="$REPO_ROOT/tools/chembl_tool/paper_experiments/current_conditioned_results.json"
PORT="${1:-8776}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [port] [task=progressive-run-root ...]" >&2
  exit 2
fi
shift $(( $# > 0 ? 1 : 0 ))

if [[ ! -f "$REGISTRY" ]]; then
  echo "Missing current-results registry: $REGISTRY" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to resolve current progressive roots." >&2
  exit 1
fi

SERVE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/txagent-trace-viewer.XXXXXX")"
SERVER_PID=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf -- "$SERVE_ROOT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$SERVE_ROOT/runs"
ln -s "$SCRIPT_DIR/viewer.html" "$SERVE_ROOT/.trace_viewer.html"
ln -s "$REGISTRY" "$SERVE_ROOT/.current_conditioned_results.json"
: > "$SERVE_ROOT/.trace_viewer_catalog.tsv"

task_label() {
  case "$1" in
    bbb_martins) echo "BBB Martins" ;;
    bioavailability_ma) echo "Bioavailability Ma" ;;
    skin_reaction) echo "Skin Reaction" ;;
    *) echo "$1" ;;
  esac
}

register_task() {
  local task="$1"
  local run_root="$2"
  local status="$3"
  local receipt="$4"
  local absolute_root="$run_root"

  if [[ "$absolute_root" != /* ]]; then
    absolute_root="$REPO_ROOT/$absolute_root"
  fi
  if [[ ! -f "$absolute_root/experiment_manifest.json" ]]; then
    echo "Skipping $task: missing experiment_manifest.json under $absolute_root" >&2
    return
  fi
  if [[ ! -d "$absolute_root/$task/levels" || ! -f "$absolute_root/$task/none/predictions.jsonl" ]]; then
    echo "Skipping $task: not a progressive task root: $absolute_root/$task" >&2
    return
  fi

  ln -s "$absolute_root" "$SERVE_ROOT/runs/$task"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$task" "$(task_label "$task")" "$status" "runs/$task/$task" \
    "runs/$task/experiment_manifest.json" "$receipt" >> "$SERVE_ROOT/.trace_viewer_catalog.tsv"
  echo "Registered $task [$status] -> $absolute_root"
}

if (( $# > 0 )); then
  for specification in "$@"; do
    if [[ "$specification" != *=* ]]; then
      echo "Invalid override '$specification'; expected task=progressive-run-root" >&2
      exit 2
    fi
    register_task "${specification%%=*}" "${specification#*=}" "manual_override" ""
  done
else
  while IFS=$'\t' read -r task run_root; do
    status="$(jq -r --arg task "$task" '.progressive_tasks[$task].status // "unregistered"' "$REGISTRY")"
    receipt="$(jq -r --arg task "$task" '.progressive_tasks[$task].retrieval_change_receipt // ""' "$REGISTRY")"
    register_task "$task" "$run_root" "$status" "$receipt"
  done < <(
    jq -r '.result_families.progressive_append_only.scaffold | to_entries[] | [.key, .value] | @tsv' "$REGISTRY"
  )
fi

catalog_count="$(wc -l < "$SERVE_ROOT/.trace_viewer_catalog.tsv")"
if (( catalog_count == 0 )); then
  echo "No progressive task roots are available." >&2
  exit 1
fi

echo "Registered progressive tasks: $catalog_count"
echo "Serving curated trace roots from: $SERVE_ROOT"
echo "Open: http://127.0.0.1:$PORT/.trace_viewer.html?v=paper-v3"

cd "$SERVE_ROOT"
python3 -m http.server "$PORT" --bind 127.0.0.1 &
SERVER_PID=$!
wait "$SERVER_PID"
