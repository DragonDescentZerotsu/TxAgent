#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-python}"
CACHE_SET="${CACHE_SET:-predict/retrieval/assay_reranking/flat_v5_manifests/scaffold_disjoint_parent100_v2/catalog.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/caches/record_scores/bioavailability_ma/v2}"
release="$(cat data/evidence_libraries/bioavailability_ma/CURRENT)"
for benchmark in gold tdc; do
    bindings="$("$PYTHON" - "$CACHE_SET" "$benchmark" "$release" <<'PY'
from pathlib import Path
import sys
from predict.retrieval.assay_reranking.artifact_bundle import cache_task
from predict.utils.json import read_json
index = Path(cache_task(Path(sys.argv[1]), 'bioavailability_ma', sys.argv[2])['retrieval']['later']).resolve()
projection = read_json(index.parent / read_json(index)['evidence']['manifest'])
if projection['evidence_library_release'] != sys.argv[3]:
    raise ValueError('Retrieval projection differs from Oral CURRENT')
print(index)
print(projection['source_paths']['records'])
PY
)"
    mapfile -t inputs <<< "$bindings"
    "$PYTHON" -m optimization.selection.record_scores \
        --release-index "${inputs[0]}" --canonical-records "${inputs[1]}" \
        --output-root "$OUTPUT_ROOT/$benchmark"
done
