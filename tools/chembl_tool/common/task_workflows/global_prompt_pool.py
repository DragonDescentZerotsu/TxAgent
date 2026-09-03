"""Compatibility alias for :mod:`predict.harnesses.branches.pool`."""

import sys

from predict.harnesses.branches import pool as _implementation


# Return the canonical module itself so historical monkeypatching and private
# helper imports still affect the code executed by its public functions.
sys.modules[__name__] = _implementation
