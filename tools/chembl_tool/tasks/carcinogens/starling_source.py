"""Source-field adapters; animal, human and mechanistic claims stay distinct."""

def fields(endpoint, value, unit, assay, species):
    return dict(zip(("canonical_endpoint_name", "canonical_measurement_text", "canonical_unit_text", "canonical_assay_context", "canonical_species_context"), (tuple(x.split()) for x in (endpoint, value, unit, assay, species))))

FIELDS = {
    "base": fields("cancer_or_tumor carcinogenic_role", "carcinogenicity_conclusion classification_authority classification_label", "", "evidence_basis exposure_route", "evidence_scope evidence_population_or_model"),
    "v1": fields("assay_family endpoint_measure", "effect_direction result_value", "measurement_unit", "assay_method biological_test_system", ""),
    "v2": fields("assay_family endpoint_detail", "result_interpretation response_value", "response_unit", "test_system metabolic_activation", ""),
    "v3": fields("endpoint_category endpoint", "effect_direction reported_result", "measurement_unit", "assay_and_detection_method biological_system", "species"),
    "v4": fields("assay_domain endpoint", "effect_direction endpoint_result", "measurement_unit", "assay_type biological_test_system", ""),
    "v5": fields("phenotype_domain endpoint_and_method", "effect_direction endpoint_result", "endpoint_unit", "assay_type assay_format biological_model", ""),
}
