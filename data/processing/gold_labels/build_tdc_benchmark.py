"""Validate the single frozen six-task TDC benchmark publication.

Historical source construction is preserved in data/legacy/tdc/code/.
"""
from __future__ import annotations

import argparse

def main() -> int:
    """Validate the canonical frozen benchmark without regenerating its cohorts."""
    from data.processing.gold_labels.conditioned_benchmark import TASK_DIRECTORIES, tdc_manifest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_DIRECTORIES), default=list(TASK_DIRECTORIES))
    args = parser.parse_args()
    for task in args.tasks:
        for subset in ("valid", "test"):
            print(tdc_manifest(task, subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
