"""Bioavailability_Ma binding for the shared auxiliary-metadata attacher.

The lookup and audit live in ``common/starling/auxiliary_metadata.py``.  This
module pins the mapping this task uses and the three mechanism sources it
applies to.
"""

from __future__ import annotations

from pathlib import Path

from tools.chembl_tool.common.starling.auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    OUTPUT_FIELDS,
)
from tools.chembl_tool.common.starling.auxiliary_metadata import (
    AuxiliaryMetadataAttacher as _AuxiliaryMetadataAttacher,
)
from tools.chembl_tool.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
    NULL_LIKE,
)


DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parent
    / "data_processing"
    / "globally_reconciled_auxiliary_value_mapping.json"
)
APPLICABLE_SOURCES = ("oral_exposure", "fa", "fg", "fh")
OUTPUT_FIELDS_BY_SOURCE = {
    "oral_exposure": (
        "global_species_context",
        "global_biological_matrix",
    ),
    "fa": OUTPUT_FIELDS,
    "fg": OUTPUT_FIELDS,
    "fh": OUTPUT_FIELDS,
}


class AuxiliaryMetadataAttacher(_AuxiliaryMetadataAttacher):
    def __init__(self, path: str | Path = DEFAULT_MAPPING_PATH):
        super().__init__(
            path,
            mapping_version=MAPPING_VERSION,
            applicable_sources=APPLICABLE_SOURCES,
            null_like=NULL_LIKE,
            output_fields=OUTPUT_FIELDS_BY_SOURCE,
        )


__all__ = [
    "APPLICABLE_SOURCES",
    "AUXILIARY_ATTACHMENT_VERSION",
    "AuxiliaryMetadataAttacher",
    "DEFAULT_MAPPING_PATH",
    "OUTPUT_FIELDS",
    "OUTPUT_FIELDS_BY_SOURCE",
]
