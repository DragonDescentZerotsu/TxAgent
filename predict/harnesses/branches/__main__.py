"""Predict from explicit cache, selection and prompt artifacts."""

from predict.harnesses.branches.flat import run

def main(argv=None):
    return run(argv)

if __name__ == "__main__":
    raise SystemExit(main())
