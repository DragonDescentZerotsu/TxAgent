"""Declarative Starling v7 schema for Skin_Reaction.

The names in this module are intentionally source-faithful.  For example,
``assay_type`` remains a cleaned source field even though its text supplies
both an assay concept and an extracted species context.
"""

from __future__ import annotations

from data.processing.evidence_library.shared.v1.canonicalization_v7 import (
    CanonicalDimensionSpec,
    CanonicalProducerSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from data.processing.evidence_library.shared.v1.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v8.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v8.tasks.skin_reaction.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    MEASUREMENT_SCALES,
)
from data.processing.evidence_library.versions.v8.tasks.skin_reaction.starling_measurement_semantics import (
    MEASUREMENT_SEMANTICS_VERSION,
)
from data.processing.evidence_library.versions.v8.tasks.skin_reaction.starling_source_column_contracts import (
    SOURCE_COLUMNS,
)
from data.processing.evidence_library.versions.v8.tasks.skin_reaction.starling_reference_semantics import (
    MAPPING_VERSION as REFERENCE_SEMANTICS_VERSION,
    REFERENCE_SEMANTICS_CONFIG,
)
from data.processing.evidence_library.versions.v8.tasks.skin_reaction.starling_spacing_and_spelling import (
    ENDPOINT_CONCEPT_VERSION,
)


TASK_ID = "skin_reaction"
ENDPOINT_RULE_VERSION = "skin_reaction_endpoint_name.v7"
MEASUREMENT_UNIT_VERSION = MEASUREMENT_SEMANTICS_VERSION
SMILES_VERSION = "rdkit_standardized_source_smiles.v1"
MEASUREMENT_ATOMIC_GROUP = "canonical_measurement_unit_pair"
PAIR_PRODUCER_FIELD = "canonical_pair_producer_id"
SOURCE_PAIR_PRODUCER_IDS = {
    source_id: f"skin_reaction.{source_id}.legacy_measurement_pair.v1"
    for source_id in (
        "direct_skin_reaction",
        "sensitization_aop",
        "phototoxicity_irritation_local_damage",
        "skin_exposure",
    )
}
SOURCE_RULE_PAIR_PRODUCER_IDS = {
    source_id: f"skin_reaction.{source_id}.exact_source_pair.v1"
    for source_id in SOURCE_PAIR_PRODUCER_IDS
}
SOURCE_EXTRACTION_PAIR_PRODUCER_IDS = {
    source_id: f"skin_reaction.{source_id}.frozen_extraction_pair.v1"
    for source_id in SOURCE_RULE_PAIR_PRODUCER_IDS
}


def _base_dimensions(
    *,
    source_id: str,
    endpoint_method: str = "deterministic_rule",
    endpoint_legacy: str = "canonical_endpoint",
    endpoint_inputs: tuple[str, ...] = ("endpoint_name",),
    endpoint_version: str = ENDPOINT_RULE_VERSION,
    categorical_inputs: tuple[str, ...] = (),
) -> tuple[CanonicalDimensionSpec, ...]:
    measurement_inputs = ("endpoint_name", "measurement_text", "unit_text")
    categorical_producers = tuple(
        CanonicalProducerSpec(
            scale.scale_id,
            categorical_inputs,
            "controlled_encoder",
            CATEGORICAL_RESPONSE_VERSION,
        )
        for scale in MEASUREMENT_SCALES.values()
        if scale.source_id == source_id
    )
    pair_producers = categorical_producers + (
        CanonicalProducerSpec(
            SOURCE_RULE_PAIR_PRODUCER_IDS[source_id],
            measurement_inputs,
            "deterministic_rule",
            EXACT_UNIT_MAPPING_VERSION,
        ),
        CanonicalProducerSpec(
            SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id],
            measurement_inputs,
            "frozen_extraction",
            EXACT_UNIT_MAPPING_VERSION,
        ),
    )
    output = [
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint_name",
            endpoint_inputs,
            endpoint_method,
            MAPPING_VERSION if endpoint_method == "frozen_mapping" else endpoint_version,
            legacy_value_field=endpoint_legacy,
        ),
        CanonicalDimensionSpec(
            "canonical_endpoint_concept",
            "endpoint_concept",
            ("endpoint_name",),
            "frozen_mapping",
            ENDPOINT_CONCEPT_VERSION,
            depends_on=("canonical_endpoint_name",),
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_text",
            "measurement",
            measurement_inputs,
            "deterministic_rule",
            MEASUREMENT_UNIT_VERSION,
            atomic_group=MEASUREMENT_ATOMIC_GROUP,
            depends_on=("canonical_endpoint_name",),
            producer_id=SOURCE_PAIR_PRODUCER_IDS[source_id],
            producer_id_field=PAIR_PRODUCER_FIELD,
            producer_variants=pair_producers,
            legacy_value_field="canonical_measurement",
        ),
        CanonicalDimensionSpec(
            "canonical_unit_text",
            "unit",
            measurement_inputs,
            "deterministic_rule",
            MEASUREMENT_UNIT_VERSION,
            atomic_group=MEASUREMENT_ATOMIC_GROUP,
            depends_on=("canonical_endpoint_name",),
            producer_id=SOURCE_PAIR_PRODUCER_IDS[source_id],
            producer_id_field=PAIR_PRODUCER_FIELD,
            producer_variants=pair_producers,
            legacy_value_field="canonical_unit",
        ),
        CanonicalDimensionSpec(
            "canonical_smiles",
            "molecular_structure",
            ("smiles",),
            "deterministic_rule",
            SMILES_VERSION,
            legacy_value_field="canonical_smiles",
        ),
    ]
    if categorical_inputs:
        output.append(
            CanonicalDimensionSpec(
                "canonical_measurement_scale_id",
                "measurement_scale",
                categorical_inputs,
                "controlled_encoder",
                CATEGORICAL_RESPONSE_VERSION,
                legacy_value_field="categorical_encoder_id",
            )
        )
    reference_inputs = tuple(
        dict.fromkeys(
            (
                "endpoint_name",
                "measurement_text",
                "unit_text",
                "support_text",
                *REFERENCE_SEMANTICS_CONFIG.source_specs[source_id].extra_fields,
            )
        )
    )
    output.extend(
        (
            CanonicalDimensionSpec(
                "canonical_reference_scope",
                "measurement_reference_scope",
                reference_inputs,
                "frozen_mapping",
                REFERENCE_SEMANTICS_VERSION,
                missing_policy="explicit_unknown",
                atomic_group="canonical_reference_semantics_pair",
                classification_evidence=True,
                legacy_value_field="canonical_reference_scope",
            ),
            CanonicalDimensionSpec(
                "canonical_reference_basis",
                "measurement_reference_basis",
                reference_inputs,
                "frozen_mapping",
                REFERENCE_SEMANTICS_VERSION,
                missing_policy="explicit_unknown",
                atomic_group="canonical_reference_semantics_pair",
                classification_evidence=True,
                legacy_value_field="canonical_reference_basis",
            ),
        )
    )
    return tuple(output)


def _mapped(
    output_field: str,
    semantic_dimension: str,
    inputs: tuple[str, ...],
    legacy: str,
) -> CanonicalDimensionSpec:
    return CanonicalDimensionSpec(
        output_field,
        semantic_dimension,
        inputs,
        "frozen_mapping",
        MAPPING_VERSION,
        missing_policy="explicit_unknown",
        legacy_value_field=legacy,
    )


SOURCES = {
    "direct_skin_reaction": SourceProfile(
        source_id="direct_skin_reaction",
        source_columns=SOURCE_COLUMNS["direct_skin_reaction"],
        endpoint_field="reaction_type",
        measurement_field="effect_metric",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="direct_skin_reaction",
                categorical_inputs=(
                    "positive_count",
                    "total_tested",
                    "measurement_text",
                    "outcome_label",
                )
            ),
            _mapped(
                "canonical_assay_or_test",
                "assay_or_test",
                ("assay_or_test",),
                "global_context",
            ),
            _mapped(
                "canonical_species_or_population",
                "species_or_population",
                ("species_or_population",),
                "global_species_context",
            ),
            _mapped(
                "canonical_severity_grade",
                "severity_grade",
                ("measurement_text",),
                "global_severity_grade",
            ),
        ),
    ),
    "sensitization_aop": SourceProfile(
        source_id="sensitization_aop",
        source_columns=SOURCE_COLUMNS["sensitization_aop"],
        endpoint_field="endpoint_or_target",
        measurement_field="result_value",
        unit_field="result_unit",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="sensitization_aop",
                endpoint_legacy="canonical_endpoint",
                endpoint_inputs=("endpoint_name", "measurement_text", "unit_text"),
                endpoint_version=MEASUREMENT_SEMANTICS_VERSION,
            ),
            _mapped(
                "canonical_assay_type",
                "assay_type",
                ("assay_type",),
                "global_context",
            ),
            _mapped(
                "canonical_species_context",
                "species_context",
                ("assay_type", "experimental_conditions", "support_text"),
                "global_species_context",
            ),
            CanonicalDimensionSpec(
                "canonical_aop_event",
                "aop_event",
                ("aop_event",),
                "deterministic_rule",
                "skin_reaction_aop_event.v1",
                legacy_value_field="aop_event",
            ),
        ),
    ),
    "phototoxicity_irritation_local_damage": SourceProfile(
        source_id="phototoxicity_irritation_local_damage",
        source_columns=SOURCE_COLUMNS["phototoxicity_irritation_local_damage"],
        endpoint_field="evidence_endpoint",
        measurement_field="observed_effect",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="phototoxicity_irritation_local_damage",
                categorical_inputs=("result_label",),
            ),
            _mapped(
                "canonical_assay_method",
                "assay_method",
                ("assay_method",),
                "global_context",
            ),
            _mapped(
                "canonical_evidence_system",
                "evidence_system",
                ("evidence_system",),
                "global_species_context",
            ),
        ),
    ),
    "skin_exposure": SourceProfile(
        source_id="skin_exposure",
        source_columns=SOURCE_COLUMNS["skin_exposure"],
        endpoint_field="evidence_type",
        measurement_field="result_value",
        unit_field="result_unit",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="skin_exposure",
                endpoint_inputs=("endpoint_name", "measurement_text", "unit_text"),
                endpoint_version=MEASUREMENT_SEMANTICS_VERSION,
            ),
            _mapped(
                "canonical_study_design",
                "study_design",
                ("study_design",),
                "global_context",
            ),
            _mapped(
                "canonical_species_context",
                "species_context",
                ("skin_source",),
                "global_species_context",
            ),
        ),
    ),
}


PAIR_BUCKETS = {
    "direct_skin_reaction": PairBucketSpec(
        source_id="direct_skin_reaction",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_assay_or_test",
            "canonical_species_or_population",
            "canonical_measurement_scale_id",
        ),
    ),
    "sensitization_aop": PairBucketSpec(
        source_id="sensitization_aop",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_aop_event",
            "canonical_assay_type",
            "canonical_species_context",
        ),
    ),
    "phototoxicity_irritation_local_damage": PairBucketSpec(
        source_id="phototoxicity_irritation_local_damage",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_assay_method",
            "canonical_evidence_system",
            "canonical_measurement_scale_id",
        ),
    ),
    "skin_exposure": PairBucketSpec(
        source_id="skin_exposure",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_study_design",
            "canonical_species_context",
        ),
    ),
}


RECORD_CONTRACT = StarlingRecordContract(
    TASK_ID,
    SOURCES,
    PAIR_BUCKETS,
    measurement_scales=MEASUREMENT_SCALES,
)


__all__ = [
    "PAIR_BUCKETS",
    "RECORD_CONTRACT",
    "SOURCE_EXTRACTION_PAIR_PRODUCER_IDS",
    "SOURCE_PAIR_PRODUCER_IDS",
    "SOURCE_RULE_PAIR_PRODUCER_IDS",
    "SOURCES",
    "TASK_ID",
]
