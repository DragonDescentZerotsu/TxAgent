#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_ROOT="outputs/paper/molecular_evidence_agent"
PORT="${1:-8776}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [port]" >&2
  exit 2
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
echo "Open: http://127.0.0.1:$PORT/.trace_viewer.html?v=paper-v1"

cd "$TRACE_ROOT"
python3 -m http.server "$PORT" --bind 127.0.0.1
