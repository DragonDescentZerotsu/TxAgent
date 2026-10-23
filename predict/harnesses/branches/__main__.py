"""Predict from explicit cache, selection and prompt artifacts."""

def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('--prepare-paper-baseline', '--run-paper-baseline'):
        from predict.harnesses.branches.paper_baseline import run
        return run(argv)
    if argv and argv[0] == "--verify-paper-baseline":
        import argparse
        import json
        from predict.prompts.paper_baseline import verify_batch

        parser = argparse.ArgumentParser(description="Verify frozen paper baseline requests and saved responses offline.")
        parser.add_argument("--verify-paper-baseline", required=True, metavar="BATCH_ROOT")
        args = parser.parse_args(argv)
        print(json.dumps(verify_batch(args.verify_paper_baseline), indent=2))
        return 0

    from predict.harnesses.branches.flat import run
    return run(argv)

if __name__ == "__main__":
    raise SystemExit(main())
