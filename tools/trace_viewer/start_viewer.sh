#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TRACE_ROOT="outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  TRACE_ROOT="$DEFAULT_TRACE_ROOT"
  PORT="$1"
else
  TRACE_ROOT="${1:-$DEFAULT_TRACE_ROOT}"
  PORT="${2:-8776}"
fi

if [[ "$TRACE_ROOT" != /* ]]; then
  TRACE_ROOT="$PWD/$TRACE_ROOT"
fi

if [[ ! -d "$TRACE_ROOT" ]]; then
  echo "Trace root does not exist: $TRACE_ROOT" >&2
  exit 1
fi

ln -sf "$SCRIPT_DIR/viewer.html" "$TRACE_ROOT/.trace_viewer.html"

echo "Serving trace root: $TRACE_ROOT"
echo "Open: http://127.0.0.1:$PORT/.trace_viewer.html"

cd "$TRACE_ROOT"
python3 -m http.server "$PORT" --bind 127.0.0.1
