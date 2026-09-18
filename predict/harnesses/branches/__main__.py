"""Run one branch evidence organization through the shared prediction runner.

The command selects direct, flat, or full evidence organization, then delegates
query preparation, ready-stage scheduling, model calls, checkpoints, traces,
and metrics to :mod:`predict.harnesses.branches.runner`.
"""

from __future__ import annotations

import argparse
import sys


ORGANIZATIONS = ("direct", "flat", "full")


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--organization" not in raw:
        parser = argparse.ArgumentParser(
            description="Run direct, flat, or full branch prediction.",
            allow_abbrev=False,
        )
        parser.add_argument("--organization", choices=ORGANIZATIONS, required=True)
        parser.parse_args(raw)
        raise AssertionError("argparse must exit when --organization is absent")
    parser = argparse.ArgumentParser(
        description="Run direct, flat, or full branch prediction.",
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument("--organization", choices=ORGANIZATIONS, required=True)
    selected, remaining = parser.parse_known_args(raw)

    if selected.organization == "flat":
        from predict.harnesses.branches.flat import run

        return run(remaining)

    from predict.harnesses.branches.runner import mode_main

    mode = "direct" if selected.organization == "direct" else "full_mechanism"
    return mode_main(mode, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
