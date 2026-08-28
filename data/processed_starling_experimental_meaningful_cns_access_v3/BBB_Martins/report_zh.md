# BBB experimental meaningful-CNS-access gold build

- lineage: `experimental_meaningful_cns_access_v3`
- gold contract: `bbb_experimental_meaningful_cns_access_gold.v3`
- frozen source rows: 304,845
- accepted experimental source rows: 8,268
- binary parent molecules: 3,666
- parent labels Y=0/Y=1: 966/2,700
- scaffold train/valid/test: 2,934/366/366
- valid/test singleton parents: 22/22
- parent identity overlap and scaffold overlap: 0 by construction

## Gold scope

Gold contains experimentally observed brain, unbound-brain, brain/systemic-ratio, CSF, PET/autoradiography, or explicit in-vivo BBB outcomes after systemic administration. The target is meaningful or adequate CNS access versus restricted or poor access, independent of entry mechanism. Passive/in-vitro permeability and transporter results remain mechanism evidence and do not vote.

Low but nonzero exposure can remain negative when the experiment supports restricted or poor access; mere detectability is not the positive-label rule.
No endpoint or mechanism quota is used to balance the gold. Mechanism-family coverage is a separate downstream retrieval-index audit.

CSF is retained as an explicitly tagged proxy outcome; it is not represented as a brain-parenchyma measurement.

## Included endpoint-family distribution

| family | accepted records | parent molecules | Y=0 parents | Y=1 parents |
|---|---:|---:|---:|---:|
| brain_tissue | 3,383 | 2,159 | 558 | 1,601 |
| csf | 1,327 | 562 | 164 | 398 |
| brain_systemic_ratio | 1,217 | 936 | 258 | 678 |
| pet_or_autoradiography | 1,044 | 661 | 162 | 499 |
| brain_unbound | 186 | 136 | 33 | 103 |
| experimental_bbb_outcome | 161 | 146 | 19 | 127 |

## Major source-row exclusions

- `no_direct_cns_outcome`: 141,522
- `interpretation_altering_qualifying_conditions`: 98,295
- `no_explicit_experimental_basis`: 15,227
- `computational_or_predicted_result`: 12,500
- `unsupported_metal_complex_identity`: 10,440
- `no_explicit_binary_permeability_label`: 7,226
- `in_vitro_or_passive_permeability_result`: 7,217
- `non_systemic_or_altered_barrier_context`: 3,785
- `indirect_outcome_inference`: 249
- `invalid_or_unresolved_smiles`: 47
- `ex_vivo_only_result`: 20
- `within_record_direction_conflict`: 4
- `manual_source_exclusion:access_inferred_from_absent_brain_biomarker_effect`: 1
- `manual_source_exclusion:access_inferred_from_absent_neurochemical_effect`: 1
- `manual_source_exclusion:access_inferred_from_absent_recombination`: 1
- `manual_source_exclusion:access_inferred_from_comparative_target_effect`: 1
- `manual_source_exclusion:access_inferred_from_intracerebroventricular_vs_iv_effect`: 1
- `manual_source_exclusion:access_inferred_from_pharmacodynamic_id50`: 1
- `manual_source_exclusion:access_inferred_from_unspecified_in_vivo_activity`: 1
- `manual_source_exclusion:active_metabolite_without_query_dosing`: 1
- `manual_source_exclusion:autoradiography_claim_pmid_model_mismatch`: 1
- `manual_source_exclusion:background_comparator_as_query_outcome`: 1
- `manual_source_exclusion:bbb_access_inferred_only_from_in_vivo_activity`: 1
- `manual_source_exclusion:cns_infection_altered_barrier_context`: 1
- `manual_source_exclusion:combination_treatment_disease_context`: 1
- `manual_source_exclusion:compound_class_to_query_attribution_ambiguity`: 1
- `manual_source_exclusion:csf_compartment_ratio_not_systemic_access_endpoint`: 1
- `manual_source_exclusion:efflux_ratio_proxy_without_cns_exposure`: 1
- `manual_source_exclusion:endogenous_distribution_without_query_dosing`: 1
- `manual_source_exclusion:endogenous_distribution_without_systemic_query_dosing`: 1
- `manual_source_exclusion:endogenous_local_synthesis_without_systemic_query_outcome`: 1
- `manual_source_exclusion:ex_vivo_human_brain_autoradiography`: 1
- `manual_source_exclusion:ex_vivo_trace_without_systemic_query_dosing`: 1
- `manual_source_exclusion:focal_ischemia_altered_barrier_context`: 1
- `manual_source_exclusion:generic_radiotracer_capability_without_measured_uptake`: 1
- `manual_source_exclusion:in_silico_logbb_source`: 1
- `manual_source_exclusion:in_situ_perfusion_without_systemic_administration`: 1
- `manual_source_exclusion:in_vivo_efflux_ratio_without_absolute_cns_access_outcome`: 1
- `manual_source_exclusion:intracerebral_hemorrhage_altered_barrier_context`: 1
- `manual_source_exclusion:intracranial_tumor_efficacy_inference`: 1
- `manual_source_exclusion:irradiation_altered_barrier_context`: 1
- `manual_source_exclusion:ischemic_altered_barrier_relative_pet_only`: 1
- `manual_source_exclusion:lesion_model_with_bbb_leakage`: 1
- `manual_source_exclusion:lovastatin_metabolite_and_in_vitro_pmid_mismatch`: 1
- `manual_source_exclusion:low_brain_to_blood_signal_conflicts_with_meaningful_access_label`: 1
- `manual_source_exclusion:malignant_glioma_altered_barrier_context`: 1
- `manual_source_exclusion:meningitis_brain_graft_without_direct_query_pk`: 1
- `manual_source_exclusion:metabolite_and_chronic_ischemia_context`: 1
- `manual_source_exclusion:mixed_drug_clinical_relapse_inference`: 1
- `manual_source_exclusion:mixed_extract_dosing_and_mptp_altered_context`: 1
- `manual_source_exclusion:mixed_parent_metabolite_and_extrapolated_brain_level`: 1
- `manual_source_exclusion:multi_compound_attribution_ambiguity`: 1
- `manual_source_exclusion:multi_radioligand_attribution_ambiguity`: 1
- `manual_source_exclusion:nanoformulation_route_species_pmid_mismatch`: 1
- `manual_source_exclusion:parent_and_metabolite_brain_signal_unresolved`: 1
- `manual_source_exclusion:parent_metabolite_quantitative_attribution_mismatch`: 1
- `manual_source_exclusion:penetration_inferred_from_relative_ed50`: 1
- `manual_source_exclusion:pet_use_and_lipophilicity_without_uptake_measurement`: 1
- `manual_source_exclusion:pmid_mismatch_ceritinib_crizotinib_review`: 1
- `manual_source_exclusion:pmid_mismatch_shp099_pharmacokinetics`: 1
- `manual_source_exclusion:pmid_model_mismatch_autoradiography_claim`: 1
- `manual_source_exclusion:pmid_query_claim_mismatch`: 1
- `manual_source_exclusion:pmid_route_and_primary_experiment_mismatch_propanediol`: 1
- `manual_source_exclusion:predicted_in_vitro_bbb_model`: 1
- `manual_source_exclusion:qa_uncertain_alzheimer_surgical_barrier_context`: 1
- `manual_source_exclusion:qa_uncertain_csf_measurement_context`: 1
- `manual_source_exclusion:qa_uncertain_efflux_ratio_proxy_without_exposure`: 1
- `manual_source_exclusion:qa_uncertain_formulation_aggregation`: 1
- `manual_source_exclusion:qa_uncertain_logbb_experimental_provenance`: 1
- `manual_source_exclusion:qa_uncertain_meaningful_access_label_direction`: 1
- `manual_source_exclusion:qa_uncertain_measurement_provenance_pharmacodynamic_pmid`: 1
- `manual_source_exclusion:qa_uncertain_neurodegeneration_model_context`: 1
- `manual_source_exclusion:qa_uncertain_pet_metric_text_consistency`: 1
- `manual_source_exclusion:qa_uncertain_preclinical_measurement_context`: 1
- `manual_source_exclusion:qa_uncertain_prediction_paper_logbb_provenance`: 1
- `manual_source_exclusion:qa_uncertain_prediction_paper_reference_value`: 1
- `manual_source_exclusion:qa_uncertain_prodrug_generated_query_analyte`: 1
- `manual_source_exclusion:qa_uncertain_secondary_background_provenance`: 1
- `manual_source_exclusion:qa_uncertain_secondary_clinical_bbb_claim`: 1
- `manual_source_exclusion:qa_uncertain_secondary_csf_claim_without_primary_measurement`: 1
- `manual_source_exclusion:qa_uncertain_secondary_measurement_context`: 1
- `manual_source_exclusion:qa_uncertain_secondary_or_predicted_logbb_metric`: 1
- `manual_source_exclusion:qa_uncertain_transgenic_disease_model_context`: 1
- `manual_source_exclusion:qa_uncertain_unbound_ratio_context`: 1
- `manual_source_exclusion:qa_uncertain_zebrafish_systemic_route`: 1
- `manual_source_exclusion:qualitative_label_conflicts_with_near_equal_brain_blood_concentrations`: 1
- `manual_source_exclusion:query_analyte_mismatch_human_albumin`: 1
- `manual_source_exclusion:query_analyte_pmid_mismatch_thyroxine_growth_hormone`: 1
- `manual_source_exclusion:query_identity_mismatch_dexpramipexole_vs_dexamethasone`: 1
- `manual_source_exclusion:query_identity_mismatch_selegiline`: 1
- `manual_source_exclusion:query_not_systemically_dosed_brain_generated_metabolite`: 1
- `manual_source_exclusion:query_pmid_evidence_mismatch_dopamine`: 1
- `manual_source_exclusion:query_pmid_identity_mismatch_antisense`: 1
- `manual_source_exclusion:query_pmid_mismatch_dopamine`: 1
- `manual_source_exclusion:query_pmid_mismatch_jnj54175446`: 1
- `manual_source_exclusion:query_radioligand_mismatch`: 1
- `manual_source_exclusion:query_text_molecule_mismatch_mirex`: 1
- `manual_source_exclusion:species_endpoint_pmid_mismatch_tolcapone`: 1
- `manual_source_exclusion:systemic_claim_attached_to_in_vitro_slice_pmid`: 1
- `manual_source_exclusion:transporter_efflux_proxy_without_cns_exposure`: 1
- `manual_source_exclusion:transporter_inhibitor_cotreatment_context`: 1
- `manual_source_exclusion:tuberculous_meningitis_altered_barrier_context`: 1
- `manual_source_exclusion:uncontrolled_disease_and_indirect_serotonin_context`: 1
- `manual_source_exclusion:unresolved_parent_and_metabolite_total_radioactivity`: 1
