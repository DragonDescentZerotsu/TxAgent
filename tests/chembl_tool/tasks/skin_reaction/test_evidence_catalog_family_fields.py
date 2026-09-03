"""Family labels must survive the compaction round-trip.

``compact_persisted_records`` strips ``assay_tier``, ``endpoint_group``,
``evidence_role`` and ``target_pref_name`` from the organize-stage artifact
because they are derivable.  Resuming the index stage reloads records from that
artifact, so without a family resolver the catalog silently emits empty labels
and a resumed build disagrees with a full one.
"""

import pytest

from data.processing.evidence_library.compact_artifacts import (
    build_relational_evidence_catalog,
    compact_persisted_records,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_policy import POLICY

FAMILY_FIELDS = ("assay_tier", "endpoint_group", "evidence_role", "target_pref_name")


def _record():
    assignment = POLICY.family_resolver("skin_exposure", "permeability_coefficient")
    assert assignment is not None
    return {
        "normalized_record_id": "rec-1",
        "source_id": "skin_exposure",
        "source_name": "starling-labs/skin_reaction/Skin_Exposure",
        "source_row_number": 1,
        "endpoint_name": "permeability_coefficient",
        "canonical_endpoint": "permeability_coefficient",
        "canonical_smiles": "CCO",
        "group_id": assignment.group_id,
        "assay_tier": assignment.assay_tier,
        "endpoint_group": assignment.endpoint_group,
        "evidence_role": assignment.evidence_role,
        "target_pref_name": assignment.target_pref_name,
        "retrieval_eligible": True,
        "finite_scalar_value": 1.5,
        "confidence": 0.9,
    }


def test_full_run_catalog_carries_family_labels():
    families, _ = build_relational_evidence_catalog(
        [_record()], family_resolver=POLICY.family_resolver
    )
    assert len(families) == 1
    for field in FAMILY_FIELDS:
        assert families[0][field], field


def test_resumed_catalog_matches_a_full_run():
    full, _ = build_relational_evidence_catalog(
        [_record()], family_resolver=POLICY.family_resolver
    )
    # A resumed build reads back exactly what the organize stage persisted.
    persisted = compact_persisted_records([_record()])
    assert all(field not in persisted[0] for field in FAMILY_FIELDS)
    resumed, _ = build_relational_evidence_catalog(
        persisted, family_resolver=POLICY.family_resolver
    )
    assert resumed == full


def test_without_a_resolver_the_labels_are_lost():
    """Documents the old behaviour the resolver exists to prevent."""
    persisted = compact_persisted_records([_record()])
    families, _ = build_relational_evidence_catalog(persisted)
    assert all(families[0][field] == "" for field in FAMILY_FIELDS)


@pytest.mark.parametrize("source_id", ["skin_exposure", "sensitization_aop"])
def test_resolver_covers_every_scalar_source(source_id):
    assignment = POLICY.family_resolver(source_id, "anything")
    assert assignment is not None
    assert assignment.group_id
