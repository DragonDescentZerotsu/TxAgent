"""Progressive append-only evidence-card inference."""


def main(argv=None):
    from predict.harnesses.progressive.runner import main as runner_main

    return runner_main(argv)


def run(*args, **kwargs):
    from predict.harnesses.progressive.runner import run as runner_run

    return runner_run(*args, **kwargs)


__all__ = ["main", "run"]
