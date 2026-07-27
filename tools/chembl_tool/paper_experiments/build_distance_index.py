"""Merge frozen ChEMBL base and H1/H2 extension indices into one distance index."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from tools.chembl_tool.common.distance_index import merge_neighbor_indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", required=True)
    parser.add_argument("--extension-index", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--index-version", required=True)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args(argv)

    with Path(args.base_index).open("rb") as handle:
        base_index = pickle.load(handle)
    with Path(args.extension_index).open("rb") as handle:
        extension_index = pickle.load(handle)
    merged = merge_neighbor_indices(
        base_index,
        extension_index,
        index_version=args.index_version,
        source_release=args.source_release,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    output_index = Path(args.output_index)
    output_meta = Path(args.output_meta)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with output_index.open("wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    output_meta.write_text(
        json.dumps(
            {
                "index_version": merged["version"],
                "source": merged["source"],
                "distance_index": merged["distance_index"],
                "n_molecules": len(merged["molecules"]),
                "n_groups": len(merged["group_to_molecule_indices"]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
