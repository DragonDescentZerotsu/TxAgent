"""Compatibility entry point for the progressive inference runner."""

from predict.harnesses.progressive import runner as _impl

main = _impl.main
run = _impl.run


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(main())
