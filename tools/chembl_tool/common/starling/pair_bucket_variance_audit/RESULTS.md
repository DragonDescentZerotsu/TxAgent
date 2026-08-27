# Pair-bucket variance audit

> Historical snapshot: this report predates promotion of canonical exact oral dose into the V7 pair key. Regenerate it after the pending exact-unit-map and Bioavailability Stage-02/03 rebuild; `analyze.py` now reads the V7 dose fields directly.

This is a read-only diagnostic of how cumulative pair-bucket dimensions change within-molecule and between-molecule variation. It does not modify pair buckets.

## Method

- Fixed cohort: retrieval-eligible records with resolved SMILES and positive, linear continuous measurements.
- Excluded categorical encodings, logits, nonpositive values, and units already expressed as `log10(...)`.
- Each source contributes its two largest endpoints having at least 100 records, 30 molecules, and 20 repeatedly measured molecules.
- Values are analyzed as `log10(value)`. A standard deviation of 0.301 dex is displayed as a twofold spread because `10^0.301 = 2`.
- Dimensions are added in the exact order declared by each source's `PairBucketSpec`. Missing values are an explicit `__unknown__` level.
- Repeat-pair retention is the fraction of same-molecule record pairs that remain comparable after adding dimensions. The `>=20` gate counts unique collapsed molecule units per final context.

## Inputs

| Artifact | SHA-256 |
|---|---|
| `outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7/03_records/records.parquet` | `680063eb2719d67e7176cd11785796181925d69dfe472191d568318195e7c6af` |
| `outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v7/03_records/records.parquet` | `c6806d5eee8e65c714b4fb73d20e86f4c1f4cd0dd512635d559a349549b81da9` |
| `outputs/chembl_tool/tasks/skin_reaction/evidence_library/starling_normalized_v7/03_records/records.parquet` | `843c9821d44b9d727bf63237b5e97c89566621e4738b12ec572d6aec1a29b93a` |
| `outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v5/records.parquet` | `2320e61ea83a07ab1bd17ccb6f1641a802e1b14e33a8f92aae57cabb53f94df0` |

## Cross-source results

### BBB

| Source | Endpoint | Records / molecules | Base intra / inter | Cumulative intra-fold path | Full intra / inter | Final repeat pairs | `>=20` final |
|---|---|---:|---:|---|---:|---:|---:|
| `direct_bbb` | `brain_concentration` | 2,803 / 980 | 0.367 dex (2.33x) / 0.891 dex (7.78x) | unit 2.33x -> scale 2.33x -> assay 1.91x -> species 1.83x -> scope 1.79x -> basis 1.79x | 0.252 dex (1.79x) / 0.486 dex (3.06x) | 53% | 0 groups / 0 units / 0% rows |
| `direct_bbb` | `csf_concentration` | 2,107 / 533 | 0.412 dex (2.58x) / 0.759 dex (5.74x) | unit 2.58x -> scale 2.58x -> assay 1.92x -> species 1.81x -> scope 1.74x -> basis 1.74x | 0.242 dex (1.74x) / 0.517 dex (3.29x) | 15% | 4 groups / 122 units / 13% rows |
| `passive_permeability` | `apparent_permeability` | 414 / 150 | 0.295 dex (1.97x) / 0.784 dex (6.08x) | unit 1.97x -> scale 1.97x -> assay type 1.80x -> assay 1.67x -> species 1.64x -> scope 1.64x -> basis 1.64x | 0.216 dex (1.64x) / 0.676 dex (4.74x) | 48% | 2 groups / 56 units / 24% rows |
| `passive_permeability` | `effective_permeability` | 392 / 300 | 0.206 dex (1.61x) / 0.962 dex (9.17x) | unit 1.61x -> scale 1.61x -> assay type 1.57x -> assay 1.47x -> species 1.47x -> scope 1.47x -> basis 1.47x | 0.168 dex (1.47x) / 0.833 dex (6.80x) | 59% | 4 groups / 269 units / 77% rows |
| `efflux_transport` | `efflux_ratio` | 378 / 319 | 0.076 dex (1.19x) / 0.380 dex (2.40x) | unit 1.19x -> scale 1.19x -> transporter 1.14x -> evidence type 1.14x -> assay 1.07x -> species 1.07x -> scope 1.07x -> basis 1.07x | 0.030 dex (1.07x) / 0.237 dex (1.73x) | 8% | 0 groups / 0 units / 0% rows |
| `efflux_transport` | `brain_exposure_fold_change` | 131 / 78 | 0.257 dex (1.81x) / 0.497 dex (3.14x) | unit 1.81x -> scale 1.81x -> transporter 1.65x -> evidence type 1.64x -> assay 1.35x -> species 1.35x -> scope 1.35x -> basis 1.35x | 0.130 dex (1.35x) / 0.177 dex (1.50x) | 6% | 0 groups / 0 units / 0% rows |
| `influx_transport` | `blood_to_brain_transport` | 4,398 / 427 | 0.420 dex (2.63x) / 0.732 dex (5.39x) | unit 2.63x -> mechanism 2.61x -> kinetic 2.61x -> scope 2.61x -> basis 2.61x | 0.416 dex (2.61x) / 0.706 dex (5.08x) | 97% | 15 groups / 717 units / 71% rows |
| `influx_transport` | `brain_endothelial_cell_uptake` | 1,496 / 206 | 0.413 dex (2.59x) / 0.749 dex (5.61x) | unit 2.59x -> mechanism 2.44x -> kinetic 2.44x -> scope 2.44x -> basis 2.44x | 0.387 dex (2.44x) / 0.742 dex (5.52x) | 82% | 6 groups / 311 units / 57% rows |

### Oral bioavailability

| Source | Endpoint | Records / molecules | Base intra / inter | Cumulative intra-fold path | Full intra / inter | Final repeat pairs | `>=20` final |
|---|---|---:|---:|---|---:|---:|---:|
| `hf_bioavailability` | `oral_bioavailability` | 70,446 / 13,214 | 0.289 dex (1.94x) / 0.470 dex (2.95x) | unit 1.94x -> scale 1.94x -> report type 1.85x -> evidence scope 1.85x -> scope 1.85x | 0.266 dex (1.85x) / 0.483 dex (3.04x) | 42% | 6 groups / 17,841 units / 100% rows |
| `oral_exposure` | `cmax` | 31,659 / 2,796 | 0.436 dex (2.73x) / 0.812 dex (6.48x) | unit 2.73x -> species 2.48x -> matrix 2.40x -> scope 2.39x | 0.378 dex (2.39x) / 0.813 dex (6.51x) | 54% | 50 groups / 5,658 units / 90% rows |
| `oral_exposure` | `tmax` | 21,356 / 2,520 | 0.230 dex (1.70x) / 0.316 dex (2.07x) | unit 1.70x -> species 1.59x -> matrix 1.55x -> scope 1.55x | 0.190 dex (1.55x) / 0.328 dex (2.13x) | 49% | 27 groups / 4,334 units / 97% rows |
| `fa` | `fraction_absorbed` | 5,857 / 1,077 | 0.354 dex (2.26x) / 0.527 dex (3.36x) | unit 2.26x -> assay 1.76x -> species 1.63x -> scope 1.59x | 0.201 dex (1.59x) / 0.499 dex (3.15x) | 21% | 21 groups / 1,205 units / 51% rows |
| `fa` | `solubility` | 5,220 / 983 | 0.757 dex (5.71x) / 0.979 dex (9.52x) | unit 5.71x -> assay 4.72x -> species 4.67x -> scope 4.62x | 0.665 dex (4.62x) / 0.944 dex (8.78x) | 56% | 10 groups / 930 units / 65% rows |
| `fg` | `bidirectional_permeability` | 3,684 / 1,107 | 0.365 dex (2.32x) / 0.710 dex (5.13x) | unit 2.32x -> scale 2.32x -> target 2.32x -> assay 2.01x -> species 1.98x -> scope 1.95x | 0.289 dex (1.95x) / 0.679 dex (4.77x) | 56% | 9 groups / 1,707 units / 83% rows |
| `fg` | `uptake_or_absorptive_transport` | 1,400 / 359 | 0.362 dex (2.30x) / 0.807 dex (6.41x) | unit 2.30x -> scale 2.30x -> target 2.30x -> assay 2.02x -> species 1.99x -> scope 1.87x | 0.271 dex (1.87x) / 0.557 dex (3.60x) | 61% | 5 groups / 220 units / 29% rows |
| `fh` | `intrinsic_clearance` | 14,445 / 3,310 | 0.570 dex (3.71x) / 1.010 dex (10.22x) | unit 3.71x -> assay 2.99x -> species 2.50x -> scope 2.45x | 0.390 dex (2.45x) / 0.852 dex (7.11x) | 38% | 42 groups / 4,424 units / 54% rows |
| `fh` | `cyp_metabolism` | 3,547 / 686 | 0.466 dex (2.92x) / 0.566 dex (3.68x) | unit 2.92x -> assay 2.68x -> species 2.60x -> scope 2.45x | 0.390 dex (2.45x) / 0.437 dex (2.73x) | 39% | 12 groups / 534 units / 30% rows |

### Skin reaction

| Source | Endpoint | Records / molecules | Base intra / inter | Cumulative intra-fold path | Full intra / inter | Final repeat pairs | `>=20` final |
|---|---|---:|---:|---|---:|---:|---:|
| `direct_skin_reaction` | `allergic_contact_dermatitis_contact_allergy` | 1,669 / 302 | 0.437 dex (2.73x) / 0.411 dex (2.57x) | unit 2.73x -> test 2.61x -> population 2.61x -> scale 2.61x -> scope 2.61x -> basis 2.61x | 0.417 dex (2.61x) / 0.410 dex (2.57x) | 81% | 3 groups / 326 units / 94% rows |
| `direct_skin_reaction` | `sensitization` | 648 / 279 | 0.431 dex (2.70x) / 0.755 dex (5.69x) | unit 2.70x -> test 1.61x -> population 1.56x -> scale 1.56x -> scope 1.56x -> basis 1.56x | 0.194 dex (1.56x) / 0.561 dex (3.64x) | 24% | 5 groups / 198 units / 43% rows |
| `sensitization_aop` | `llna_ec3` | 661 / 267 | 0.252 dex (1.79x) / 0.999 dex (9.99x) | unit 1.79x -> AOP event 1.79x -> assay type 1.79x -> species 1.66x -> scope 1.57x -> basis 1.57x | 0.197 dex (1.57x) / 0.998 dex (9.96x) | 47% | 4 groups / 301 units / 79% rows |
| `sensitization_aop` | `clinical_sensitization_incidence` | 444 / 114 | 0.459 dex (2.87x) / 0.634 dex (4.30x) | unit 2.87x -> AOP event 2.87x -> assay type 2.53x -> species 2.52x -> scope 2.05x -> basis 2.05x | 0.312 dex (2.05x) / 0.419 dex (2.62x) | 52% | 1 groups / 68 units / 45% rows |
| `phototoxicity_irritation_local_damage` | `light_dependent_cytotoxicity` | 434 / 172 | 0.290 dex (1.95x) / 0.597 dex (3.96x) | unit 1.95x -> method 1.70x -> system 1.70x -> scale 1.70x -> scope 1.70x -> basis 1.70x | 0.230 dex (1.70x) / 0.449 dex (2.81x) | 50% | 2 groups / 55 units / 28% rows |
| `phototoxicity_irritation_local_damage` | `skin_irritation` | 393 / 164 | 0.343 dex (2.20x) / 0.539 dex (3.46x) | unit 2.20x -> method 2.01x -> system 1.83x -> scale 1.83x -> scope 1.83x -> basis 1.83x | 0.262 dex (1.83x) / 0.422 dex (2.64x) | 47% | 2 groups / 45 units / 22% rows |
| `skin_exposure` | `dermal_absorption` | 26,186 / 2,069 | 0.645 dex (4.42x) / 0.746 dex (5.57x) | unit 4.42x -> design 3.60x -> species 3.03x -> scope 2.87x -> basis 2.83x | 0.452 dex (2.83x) / 0.783 dex (6.06x) | 19% | 76 groups / 5,723 units / 78% rows |
| `skin_exposure` | `skin_retention` | 21,703 / 1,675 | 0.644 dex (4.41x) / 0.825 dex (6.68x) | unit 4.41x -> design 3.74x -> species 3.34x -> scope 3.20x -> basis 3.18x | 0.503 dex (3.18x) / 0.771 dex (5.90x) | 36% | 48 groups / 2,819 units / 55% rows |

## Oral exact-dose extension

Normalized exact dose is `quantity kind | basis | canonical unit | exact value`. Unresolved values remain in one explicit `__unknown__` level. The decade comparison uses the same normalized components with `floor(log10(value))`.

| Endpoint | Rows / molecules | Exact dose known | Current pair key within / inter | + normalized exact dose | Exact pair retention | `>=20` after exact | + decade dose |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmax` | 31,659 / 2,796 | 23,977 (76%) | 0.378 dex (2.39x) / 0.813 dex (6.51x) | 0.226 dex (1.68x) | 33% | 99 groups / 5,686 units / 52% rows | 0.277 dex (1.89x), pairs 54% |
| `tmax` | 21,356 / 2,520 | 16,215 (76%) | 0.190 dex (1.55x) / 0.328 dex (2.13x) | 0.143 dex (1.39x) | 31% | 74 groups / 5,360 units / 65% rows | 0.164 dex (1.46x), pairs 52% |
| `auc0_inf` | 12,469 / 1,709 | 9,664 (78%) | 0.366 dex (2.32x) / 0.856 dex (7.18x) | 0.214 dex (1.64x) | 37% | 44 groups / 2,135 units / 44% rows | 0.261 dex (1.82x), pairs 57% |
| `auc0_t` | 10,809 / 1,456 | 7,970 (74%) | 0.357 dex (2.27x) / 0.862 dex (7.28x) | 0.201 dex (1.59x) | 39% | 32 groups / 1,509 units / 38% rows | 0.251 dex (1.78x), pairs 61% |
| `auc` | 11,168 / 1,539 | 8,176 (73%) | 0.352 dex (2.25x) / 0.795 dex (6.23x) | 0.228 dex (1.69x) | 44% | 24 groups / 995 units / 23% rows | 0.275 dex (1.89x), pairs 67% |

The V7 library does not persist a reviewed exact-dose field. This diagnostic reuses the historical V5 structured dose parse, joined one-to-one to V7 by `cleaned_record_id`; it does not use raw dose strings as keys.
