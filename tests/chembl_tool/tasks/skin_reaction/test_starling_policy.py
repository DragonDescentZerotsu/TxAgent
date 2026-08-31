"""The Skin_Reaction policy must satisfy the shared builder's plug-in contract."""

import json

import pytest

from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    load_task_policy,
    parse_args,
)
from tools.chembl_tool.common.starling.normalization.task_policy import (
    StarlingTaskPolicy,
)
from tools.chembl_tool.tasks.skin_reaction.starling_auxiliary_metadata import (
    APPLICABLE_SOURCES,
    PendingAuxiliaryAttacher,
)
from tools.chembl_tool.tasks.skin_reaction.starling_pair_buckets import (
    ENDPOINT_FIELD_BY_SOURCE,
    SOURCE_PAIR_FIELDS,
    V7_ENDPOINT_FIELD_BY_SOURCE,
)
from tools.chembl_tool.tasks.skin_reaction.starling_policy import POLICY
from tools.chembl_tool.tasks.skin_reaction.starling_spacing_and_spelling import (
    ENDPOINT_CONCEPT_PATHS,
    endpoint_concept,
)
from tools.chembl_tool.tasks.skin_reaction.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
)


def test_endpoint_concept_maps_cover_every_frozen_source_pair():
    for path in ENDPOINT_CONCEPT_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["mappings"]:
            assert endpoint_concept(
                payload["source_id"],
                row["endpoint_name"],
                row["canonical_endpoint_name"],
            ) == row["canonical_endpoint_concept"]

    assert endpoint_concept(
        "direct_skin_reaction",
        "allergic_contact_dermatitis_contact_ally",
        "allergic_contact_dermatitis_contact_allergy",
    ) == "allergic_contact_dermatitis_contact_allergy"
    with pytest.raises(ValueError, match="unreviewed endpoint concept"):
        endpoint_concept("skin_exposure", "new endpoint", "new_endpoint")


def test_policy_is_discoverable_by_the_shared_builder():
    assert load_task_policy("skin_reaction") is POLICY
    assert isinstance(POLICY, StarlingTaskPolicy)
    assert POLICY.task_id == "skin_reaction"


def test_compact_profile_is_namespaced_to_this_task():
    # A shared index loader resolves the owning task from these strings, so a
    # collision with another task's index would silently load the wrong policy.
    assert POLICY.compact.task_id == "skin_reaction"
    assert POLICY.compact.index_version.startswith("skin_reaction.")
    assert POLICY.compact.artifact_version.startswith("skin_reaction.")
    assert "skin" in POLICY.compact.evidence_source_label.casefold()


def test_each_source_is_stratified_by_what_actually_makes_it_comparable():
    """Measured sources stratify on reconciled context; encoded ones on encoder."""
    assert set(SOURCE_PAIR_FIELDS) == set(EXPECTED_SOURCE_ROWS)
    assert SOURCE_PAIR_FIELDS["direct_skin_reaction"] == (
        "global_context",
        "global_species_context",
        "categorical_encoder_id",
    )
    assert SOURCE_PAIR_FIELDS["sensitization_aop"] == (
        "aop_event",
        "global_context",
        "global_species_context",
    )
    assert SOURCE_PAIR_FIELDS["phototoxicity_irritation_local_damage"] == (
        "global_context",
        "global_species_context",
        "categorical_encoder_id",
    )
    assert SOURCE_PAIR_FIELDS["skin_exposure"] == (
        "global_context",
        "global_species_context",
    )
    assert ENDPOINT_FIELD_BY_SOURCE == {
        "sensitization_aop": "global_endpoint_context"
    }
    assert V7_ENDPOINT_FIELD_BY_SOURCE == {
        source: "canonical_endpoint_concept" for source in SOURCE_PAIR_FIELDS
    }
    # Every source declares something; none is left without a stratum.
    assert all(fields for fields in SOURCE_PAIR_FIELDS.values())


def test_bounded_runs_require_an_unfrozen_endpoint_inventory():
    with pytest.raises(SystemExit):
        parse_args(POLICY, ["--max-rows-per-source", "10"])


def test_missing_auxiliary_mapping_must_be_opted_into():
    with pytest.raises(SystemExit):
        parse_args(POLICY, ["--auxiliary-mapping", "/nonexistent/mapping.json"])
    args = parse_args(
        POLICY,
        [
            "--auxiliary-mapping",
            "/nonexistent/mapping.json",
            "--allow-missing-auxiliary-mapping",
            "--allow-missing-reference-semantics",
        ],
    )
    assert args.allow_missing_auxiliary_mapping is True


def test_stage01_auxiliary_dependency_must_be_present_or_explicitly_pending():
    with pytest.raises(SystemExit):
        parse_args(
            POLICY,
            [
                "--through-stage",
                "clean",
                "--auxiliary-mapping",
                "/nonexistent/mapping.json",
            ],
        )
    args = parse_args(
        POLICY,
        [
            "--through-stage",
            "clean",
            "--auxiliary-mapping",
            "/nonexistent/mapping.json",
            "--allow-missing-auxiliary-mapping",
        ],
    )
    assert args.through_stage == "clean"


def test_pending_attacher_marks_applicable_sources_unavailable_not_mapped():
    attacher = PendingAuxiliaryAttacher()
    scalar = attacher.attach({"source_id": "skin_exposure"})
    qualitative = attacher.attach({"source_id": "direct_skin_reaction"})
    assert scalar["auxiliary_mapping_status"] == "not_available"
    assert qualitative["auxiliary_mapping_status"] == "not_available"
    # An incomplete build must fail its coverage validation rather than look done.
    audit = attacher.coverage_audit([scalar, qualitative])
    assert audit["validations"]["all_applicable_records_mapped"] is False


def test_manifest_versions_are_task_scoped():
    versions = POLICY.manifest_versions()
    assert versions["spacing_and_spelling_version"].startswith("skin_reaction")
    assert versions["endpoint_policy_version"].startswith("skin_reaction")
    assert versions["normalization_domain_rules_version"].startswith("skin_reaction")
    # Bioavailability-only keys must not leak into another task's manifest.
    assert "fg_scalar_rule_version" not in versions
    assert "report_type_normalization_version" not in versions


def test_both_canonical_partition_sources_require_endpoint_identity():
    required = set(POLICY.endpoint_identity_required_sources)
    assert {"direct_skin_reaction", "sensitization_aop"} <= required
