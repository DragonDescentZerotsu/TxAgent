"""Source-field adapters; no gold threshold or retrieval exclusion is inferred."""

def fields(endpoint, value, unit, assay, species):
    return dict(zip(("canonical_endpoint_name", "canonical_measurement_text", "canonical_unit_text", "canonical_assay_context", "canonical_species_context"), (tuple(x.split()) for x in (endpoint, value, unit, assay, species))))

FIELDS = {
    "base": fields("clinical_phenotype biochemical_pattern", "causal_status maximum_reported_severity", "", "human_evidence_basis reaction_type exposure_context", ""),
    "v1": fields("endpoint_category assay_and_readout", "result_value", "result_unit", "biological_model_context culture_format", "species"),
    "v2": fields("assay_category assay_detail", "reported_result", "", "biological_system", "species"),
    "v3": fields("assay_category target_or_process endpoint_metric", "effect_direction result_value", "result_unit", "assay_format biological_system_context", ""),
    "v4": fields("assay_family endpoint_and_comparator", "effect_direction result_value", "result_unit", "experimental_system analytical_platform_and_normalization", ""),
    "v5": fields("endpoint_class endpoint_name", "effect_direction quantitative_measure_type quantitative_value", "quantitative_unit", "biological_model_context assay_and_platform", ""),
}
