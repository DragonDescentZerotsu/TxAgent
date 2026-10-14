"""Run FINAL flat prediction through the shared scheduler."""

from __future__ import annotations

import argparse
import sys


ORGANIZATIONS = ("flat",)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--organization" not in raw:
        parser = argparse.ArgumentParser(
            description="Run FINAL flat prediction.",
            allow_abbrev=False,
        )
        parser.add_argument("--organization", choices=ORGANIZATIONS, required=True)
        parser.parse_args(raw)
        raise AssertionError("argparse must exit when --organization is absent")
    parser = argparse.ArgumentParser(
        description="Run FINAL flat prediction.",
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument("--organization", choices=ORGANIZATIONS, required=True)
    selected, remaining = parser.parse_known_args(raw)

    from predict.harnesses.branches.flat import run

    return run(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
