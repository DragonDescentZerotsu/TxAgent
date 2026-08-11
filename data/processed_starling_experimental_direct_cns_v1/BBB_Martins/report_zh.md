# BBB experimental direct-CNS gold build

- lineage: `experimental_direct_cns_v1`
- gold contract: `bbb_experimental_direct_cns_gold.v1`
- frozen source rows: 304,845
- accepted experimental source rows: 11,406
- binary parent molecules: 4,195
- parent labels Y=0/Y=1: 1,120/3,075
- scaffold train/valid/test: 3,357/419/419
- valid/test singleton parents: 14/15
- parent identity overlap and scaffold overlap: 0 by construction

## Gold scope

Gold contains experimentally observed brain, unbound-brain, brain/systemic-ratio, CSF, PET/autoradiography, or explicit in-vivo BBB outcomes. Passive/in-vitro permeability and transporter results remain mechanism evidence and do not vote.

CSF is retained as an explicitly tagged proxy outcome; it is not represented as a brain-parenchyma measurement.

## Included endpoint-family distribution

| family | accepted records | parent molecules | Y=0 parents | Y=1 parents |
|---|---:|---:|---:|---:|
| brain_tissue | 3,740 | 2,304 | 605 | 1,699 |
| csf | 2,571 | 711 | 217 | 494 |
| brain_systemic_ratio | 2,080 | 1,326 | 363 | 963 |
| pet_or_autoradiography | 1,055 | 673 | 158 | 515 |
| brain_unbound | 329 | 214 | 64 | 150 |
| experimental_bbb_outcome | 181 | 165 | 22 | 143 |

## Major source-row exclusions

- `no_direct_cns_outcome`: 146,879
- `interpretation_altering_qualifying_conditions`: 103,340
- `no_explicit_binary_permeability_label`: 12,983
- `computational_or_predicted_result`: 12,463
- `in_vitro_or_passive_permeability_result`: 7,284
- `no_explicit_experimental_basis`: 7,034
- `non_systemic_or_altered_barrier_context`: 3,314
- `indirect_outcome_inference`: 135
- `invalid_or_unresolved_smiles`: 56
- `within_record_direction_conflict`: 7
