#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PORT="${1:-8776}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [port] [trace-root ...]" >&2
  exit 2
fi

shift $(( $# > 0 ? 1 : 0 ))

if (( $# > 0 )); then
  TRACE_ROOTS=("$@")
else
  TRACE_ROOTS=(
    "outputs/paper/molecular_evidence_agent_starling_random"
    "outputs/paper/molecular_evidence_agent_starling_scaffold"
    "outputs/paper/molecular_evidence_agent"
  )
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

mkdir -p "$SERVE_ROOT/datasets"
ln -s "$SCRIPT_DIR/viewer.html" "$SERVE_ROOT/.trace_viewer.html"
ln -s "$SCRIPT_DIR/live.html" "$SERVE_ROOT/.live_trace_viewer.html"
if [[ -d "$REPO_ROOT/outputs/paper/live" ]]; then
  ln -s "$REPO_ROOT/outputs/paper/live" "$SERVE_ROOT/live-data"
fi
: > "$SERVE_ROOT/.trace_viewer_sources.tsv"
: > "$SERVE_ROOT/.trace_viewer_catalog.tsv"

CENTRAL_DATASET="$SERVE_ROOT/central_traces"
central_count="$(
  cd "$REPO_ROOT"
  python3 -m predict.traces.viewer.build_dataset \
    --trace-root "${PREDICT_TRACE_ROOT:-outputs/paper/live}" \
    --output-root "$CENTRAL_DATASET"
)"
if (( central_count > 0 )); then
  TRACE_ROOTS=("$CENTRAL_DATASET" "${TRACE_ROOTS[@]}")
fi

source_index=0
for trace_root in "${TRACE_ROOTS[@]}"; do
  if [[ "$trace_root" != /* ]]; then
    trace_root="$PWD/$trace_root"
  fi
  if [[ ! -d "$trace_root" ]]; then
    echo "Skipping missing trace root: $trace_root" >&2
    continue
  fi

  source_name="$(basename "$trace_root")"
  case "$source_name" in
    molecular_evidence_agent_starling_random)
      source_label="Starling random test"
      ;;
    molecular_evidence_agent_starling_scaffold)
      source_label="Starling scaffold test"
      ;;
    molecular_evidence_agent)
      source_label="Historical TDC test"
      ;;
    central_traces)
      source_label="Central standard and progressive traces"
      ;;
    *)
      source_label="$source_name"
      ;;
  esac

  source_id="source_${source_index}"
  ln -s "$trace_root" "$SERVE_ROOT/datasets/$source_id"
  printf '%s\t%s\t%s\n' "$source_id" "$source_label" "datasets/$source_id" \
    >> "$SERVE_ROOT/.trace_viewer_sources.tsv"
  while IFS= read -r predictions_path; do
    relative_path="${predictions_path#"$trace_root"/}"
    regime="${relative_path%%/*}"
    remainder="${relative_path#*/}"
    task="${remainder%%/*}"
    remainder="${remainder#*/}"
    condition="${remainder%%/*}"
    case "$regime" in
      runs|runs_deployment_visible_prefetched|runs_deployment_visible|runs_deployment_visible_parent_disjoint)
        printf '%s\t%s\t%s\t%s\n' "$source_id" "$regime" "$task" "$condition" \
          >> "$SERVE_ROOT/.trace_viewer_catalog.tsv"
        ;;
    esac
  done < <(
    find "$trace_root" -mindepth 4 -maxdepth 4 -type f -name predictions.jsonl -print \
      | sort
  )
  echo "Registered dataset: $source_label -> $trace_root"
  source_index=$((source_index + 1))
done

if (( source_index == 0 )); then
  echo "No trace roots are available." >&2
  exit 1
fi

catalog_count="$(wc -l < "$SERVE_ROOT/.trace_viewer_catalog.tsv")"
echo "Registered conditions: $catalog_count"
echo "Serving curated trace datasets from: $SERVE_ROOT"
echo "Open: http://127.0.0.1:$PORT/.trace_viewer.html?v=paper-v2"
echo "Live: http://127.0.0.1:$PORT/.live_trace_viewer.html?data=live-data"

cd "$SERVE_ROOT"
python3 -m http.server "$PORT" --bind 127.0.0.1 &
SERVER_PID=$!
wait "$SERVER_PID"
