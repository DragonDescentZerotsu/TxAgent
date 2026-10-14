import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.processing.evidence_library.evidence_library import starling_molecule_id
from predict.utils.json import sha256_file
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.canonical_starling_source import (
    CANONICAL_VERSION,
)






def test_starling_molecule_id_is_cross_profile_and_stable():
    assert starling_molecule_id("CCO") == "STARLING_89B394FD02E5E5E6"
    assert starling_molecule_id("CCO") == starling_molecule_id("CCO")






