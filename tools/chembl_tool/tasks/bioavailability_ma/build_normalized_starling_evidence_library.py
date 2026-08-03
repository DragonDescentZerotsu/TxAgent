"""Build the policy-decoupled layered v6 Bioavailability evidence library.

The staged builder is shared (``common/starling/build_normalized_evidence_library``);
this entry point only binds the Bioavailability_Ma policy so the historical
command line keeps working unchanged.
"""

from __future__ import annotations

from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    AUXILIARY_MAPPING_MANIFEST_FILENAME,
    CLEANED_FILENAME,
    DISTRIBUTION_AUDIT_FILENAME,
    DUPLICATES_FILENAME,
    ENDPOINT_INVENTORY_FILENAME,
    ENDPOINT_REGISTRY_FILENAME,
    EVIDENCE_BRIDGE_FILENAME,
    EVIDENCE_FAMILIES_FILENAME,
    EVIDENCE_MANIFEST_FILENAME,
    INDEX_FINGERPRINTS_FILENAME,
    INDEX_MEMBERSHIP_FILENAME,
    INDEX_META_FILENAME,
    INDEX_MOLECULES_FILENAME,
    MANIFEST_FILENAME,
    NORMALIZATION_STAGE_VERSION,
    NORMALIZED_RECORDS_FILENAME,
    RECORD_DEPENDENT_DIRECTORIES,
    RECORD_DEPENDENT_FILES,
    RECORDS_FILENAME,
    REJECTIONS_FILENAME,
    SOURCE_COLUMN_CONTRACT_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    STAGE_ARTIFACTS,
    STAGE_OUTPUT_FILENAMES,
    STAGES,
    VALIDITY_POLICY_FILENAME,
    build,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import (
    DEFAULT_OUT_DIR,
    DEFAULT_SMILES_MAPPING,
    DEFAULT_STARLING_DATA_DIR,
    EXPECTED_SMILES_MAPPING_SHA256,
    POLICY,
)


def main(argv: list[str] | None = None) -> int:
    return build(POLICY, argv)


if __name__ == "__main__":
    raise SystemExit(main())
