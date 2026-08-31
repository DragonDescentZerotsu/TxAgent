# Audit of the raw-versus-log10 normality flag

Date: 2026-08-31

## Selected contract

A scientifically justified positive multiplicative axis uses log10. Endpoint meaning and unit semantics, not sample normality, determine the transform. Explicitly logged units remain raw to prevent a second logarithm.

For audit, an axis with at least 20 records and three distinct positive values receives a raw-like flag when raw normality is not rejected (`p_raw >= 0.05`) and the raw D'Agostino-Pearson statistic is smaller than the log10 statistic (`K2_raw < K2_log10`). Otherwise it receives a log10-like flag. The flag is advisory and cannot change the transform.

| Task | Raw-like flag | Log10-like flag | Evaluated |
|---|---:|---:|---:|
| bbb_martins | 11 | 6 | 17 |
| bioavailability_ma | 16 | 8 | 24 |
| skin_reaction | 22 | 12 | 34 |
| **Total** | **49** | **26** | **75** |

All 75 evaluated axes retain their scientific log10 mapping. The two separately corrected already-log Skin units are outside this screen and remain raw unconditionally.

## What the numerical flag means

This rule answers only which representation is closer to normal under the same test. Comparing p-values is valid here only as a ranking because both p-values come from the same K-squared test on the same observations; the flag is recorded using the test statistics. Non-rejection is not proof of normality or evidence that a raw axis is scientifically preferable.

The scientific and unit rules run first: only positive multiplicative quantities are candidates, and explicit log10, negative-log, natural-log, p-scale, logit, or base-ambiguous log units can never receive another log. Fewer than 20 records are not numerically flagged.

## Density-fit sensitivity analysis

As a sensitivity check, the audit also compared a Gaussian on raw values with a zero-location lognormal on the same raw-value density, using delta AIC, skew, and 1,000 bootstrap resamples. That comparison classified 14 axes as raw-favored, 38 as log10-favored, and 23 as ambiguous. It also remains diagnostic evidence rather than transform authority.

| Task | Raw favored | Log10 favored | Ambiguous |
|---|---:|---:|---:|
| bbb_martins | 5 | 6 | 6 |
| bioavailability_ma | 4 | 15 | 5 |
| skin_reaction | 5 | 17 | 12 |
| **Total** | **14** | **38** | **23** |

## Current pair-bucket applicability

The test is fit across broad exact axes, while assay transfer operates on more specific pair buckets. In the current Stage-03 artifacts, 49 of the 75 axes have no bucket-eligible records and only 10 axes contain an eligible pair bucket with at least 20 records. Twenty-one axes split across multiple eligible pair buckets. Thus this remains a transform-policy prior, not evidence that each condition-specific bucket is Gaussian.

Sixteen raw-like flagged axes currently affect 444 bucket-eligible rows; ten log10-like flagged axes affect 250 bucket-eligible rows. The remaining flags have no present calibration impact but remain relevant to future eligibility repairs.

## Complete axis audit

`K2` is the D'Agostino-Pearson statistic; smaller is closer to normal under this test. `delta AIC` is positive when the raw-normal density fits better. `Bootstrap raw` is the fraction of resamples where raw-normal has lower AIC. Ranges show `minimum / median / maximum`.

### Numerical flag: log10-like

| Task | Exact axis | n / unique | K2 raw / log | p raw / log | Skew raw / log | Decades | Density sensitivity | delta AIC | Bootstrap raw | Range |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| bbb_martins | `direct_bbb / apical_permeability / cm/s / unknown` | 24 / 19 | 3.2 / 3.18 | 0.2023 / 0.2035 | 0.845 / -0.695 | 2.09 | Log10 favored | -11.85 | 0.15 | `2e-07 / 6.8e-06 / 2.46e-05` |
| bbb_martins | `direct_bbb / unidirectional_glucose_influx / µmol/g·min / unknown` | 24 / 22 | 3.34 / 0.151 | 0.1882 / 0.9271 | 0.67 / 0.0208 | 0.36 | Ambiguous | -1.58 | 0.23 | `1.18 / 1.78 / 2.73` |
| bbb_martins | `direct_bbb / bbb_calculated_transport / nmol/g·min / unknown` | 33 / 33 | 3.65 / 2.88 | 0.1611 / 0.237 | 0.725 / -0.685 | 1.42 | Log10 favored | -6.19 | 0.16 | `0.0049 / 0.0424 / 0.1285` |
| bbb_martins | `direct_bbb / csf_to_blood_ratio / ratio / unknown` | 41 / 34 | 5.39 / 5.18 | 0.06743 / 0.0751 | 0.846 / -0.804 | 2.12 | Log10 favored | -16.19 | 0.05 | `0.009 / 0.27 / 1.19` |
| bbb_martins | `direct_bbb / extraction_ratio / ratio / endpoint_defined_ratio` | 32 / 27 | 5.65 / 2.91 | 0.05925 / 0.2333 | 1 / -0.704 | 1.75 | Log10 favored | -12.28 | 0.08 | `0.017 / 0.23 / 0.96` |
| bbb_martins | `direct_bbb / ki / nL/g·s / unknown` | 24 / 24 | 5.79 / 1.01 | 0.05538 / 0.6044 | 1.19 / 0.182 | 1.77 | Log10 favored | -25.70 | 0.00 | `1.75 / 14.21 / 102.9` |
| bioavailability_ma | `fa / intestinal_absorption / µg/cm^2·min / absolute` | 20 / 20 | 2.88 / 1.66 | 0.2374 / 0.4358 | 0.859 / -0.58 | 2.55 | Log10 favored | -19.20 | 0.05 | `0.0621 / 2.765 / 21.96` |
| bioavailability_ma | `oral_exposure / other_oral_exposure / µg/L / absolute` | 22 / 21 | 4.06 / 3.9 | 0.1316 / 0.1419 | 0.687 / -0.626 | 2.91 | Log10 favored | -28.06 | 0.05 | `0.44 / 77.5 / 359.2` |
| bioavailability_ma | `fh / intrinsic_clearance / CYP2C8·µL/min·pmol / absolute` | 21 / 20 | 4.55 / 3.38 | 0.1029 / 0.1845 | 1.09 / -0.0972 | 2.36 | Log10 favored | -27.03 | 0.01 | `0.04 / 0.59 / 9.18` |
| bioavailability_ma | `fh / intrinsic_clearance / mL/min/gprot / unknown` | 21 / 18 | 5.49 / 0.563 | 0.0643 / 0.7545 | 1.18 / 0.26 | 1.01 | Log10 favored | -10.29 | 0.01 | `55 / 155 / 567` |
| bioavailability_ma | `fa / dissolution / mg / absolute` | 22 / 22 | 5.68 / 3.07 | 0.05845 / 0.2153 | 0.772 / -0.162 | 2.43 | Log10 favored | -29.02 | 0.01 | `0.9 / 24.1 / 242.2` |
| bioavailability_ma | `oral_exposure / dose_normalized_exposure / /L / absolute` | 20 / 19 | 5.73 / 0.21 | 0.05695 / 0.9002 | 1.2 / 0.0763 | 1.22 | Log10 favored | -10.77 | 0.02 | `0.0044 / 0.0164 / 0.073` |
| bioavailability_ma | `fa / dissolution / %/h / absolute` | 20 / 20 | 5.88 / 3.55 | 0.05296 / 0.1694 | 1.12 / -0.94 | 1.66 | Ambiguous | -3.47 | 0.31 | `0.75 / 9.675 / 34.09` |
| bioavailability_ma | `fh / intrinsic_clearance / liver·mL/g·min / absolute` | 25 / 18 | 5.88 / 3.29 | 0.05279 / 0.1927 | 1.19 / -0.282 | 2.00 | Log10 favored | -28.96 | 0.00 | `0.68 / 9.9 / 67.8` |
| skin_reaction | `skin_exposure / skin_retention / nmol/mg·tissue / unknown` | 26 / 26 | 2.54 / 2.54 | 0.2808 / 0.2815 | 0.698 / -0.692 | 1.52 | Log10 favored | -4.70 | 0.24 | `0.009 / 0.099 / 0.301` |
| skin_reaction | `skin_exposure / skin_retention / mmol/cm^2 / absolute` | 20 / 18 | 3.34 / 0.699 | 0.188 / 0.705 | 0.853 / -0.227 | 1.29 | Log10 favored | -8.44 | 0.04 | `2.85e-05 / 0.0001695 / 0.000562` |
| skin_reaction | `skin_exposure / relative_penetration / enhancement ratio / unknown` | 21 / 20 | 3.6 / 1.5 | 0.165 / 0.4732 | 0.96 / -0.159 | 1.22 | Log10 favored | -9.66 | 0.04 | `0.8 / 3.4 / 13.2` |
| skin_reaction | `skin_exposure / skin_retention / DPM / unknown` | 22 / 22 | 4.08 / 0.792 | 0.1298 / 0.673 | 0.963 / -0.141 | 1.13 | Log10 favored | -7.29 | 0.08 | `395 / 1616 / 5329` |
| skin_reaction | `skin_exposure / flux / cpm·cm^-2·min^-1 / unknown` | 30 / 30 | 4.21 / 1.47 | 0.122 / 0.4793 | 0.391 / 0.0202 | 0.39 | Ambiguous | -1.81 | 0.16 | `123.3 / 201.1 / 304.2` |
| skin_reaction | `phototoxicity_irritation_local_damage / keratinocyte_damage / mU / unknown` | 28 / 21 | 4.4 / 3.39 | 0.1109 / 0.1836 | 0.908 / 0.617 | 0.37 | Log10 favored | -5.81 | 0.00 | `125 / 168 / 294` |
| skin_reaction | `skin_exposure / tape_strip_result / cpm / unknown` | 20 / 10 | 4.82 / 4.2 | 0.08992 / 0.1224 | 0.188 / -0.122 | 0.39 | Ambiguous | -0.57 | 0.40 | `970 / 1565 / 2360` |
| skin_reaction | `skin_exposure / tape_strip_result / Eq·nmol/cm^2 / absolute` | 48 / 42 | 5.12 / 2.15 | 0.07714 / 0.3414 | 0.734 / -0.487 | 0.87 | Ambiguous | -3.21 | 0.28 | `4.5 / 14.5 / 33.2` |
| skin_reaction | `skin_exposure / skin_exposure_measurement / nM/cm^2 / unknown` | 35 / 35 | 5.31 / 0.809 | 0.0704 / 0.6672 | 0.866 / -0.334 | 0.98 | Log10 favored | -5.33 | 0.15 | `3.6 / 13.6 / 34.2` |
| skin_reaction | `skin_exposure / skin_retention / µg/mg·protein / unknown` | 42 / 31 | 5.47 / 2.48 | 0.06489 / 0.2888 | 0.761 / -0.524 | 1.41 | Log10 favored | -12.32 | 0.05 | `2.38 / 18.04 / 60.58` |
| skin_reaction | `skin_exposure / skin_exposure_measurement / ng/m^2 / unknown` | 27 / 16 | 5.59 / 4.3 | 0.06097 / 0.1167 | 1.09 / -0.941 | 3.75 | Log10 favored | -48.99 | 0.00 | `1 / 483 / 5600` |
| skin_reaction | `skin_exposure / flux / µequiv cm-2 hr-1 / unknown` | 28 / 18 | 5.73 / 1.75 | 0.05705 / 0.4159 | 0.563 / -0.176 | 1.00 | Log10 favored | -6.34 | 0.11 | `0.05 / 0.15 / 0.5` |

### Numerical flag: raw-like

| Task | Exact axis | n / unique | K2 raw / log | p raw / log | Skew raw / log | Decades | Density sensitivity | delta AIC | Bootstrap raw | Range |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| bbb_martins | `direct_bbb / kp_csf_plasma / ratio / unknown` | 23 / 20 | 0.986 / 10.7 | 0.6107 / 0.004811 | 0.389 / -1.48 | 2.33 | Ambiguous | 6.45 | 0.75 | `0.01 / 0.95 / 2.15` |
| bbb_martins | `direct_bbb / k1_star / mL/g·min / unknown` | 37 / 31 | 1.03 / 47.1 | 0.5985 / 5.921e-11 | 0.343 / -3.17 | 4.24 | Raw favored | 44.86 | 0.98 | `2e-05 / 0.127 / 0.35` |
| bbb_martins | `direct_bbb / experimental_permeability / cm/s / unknown` | 47 / 42 | 1.47 / 29.2 | 0.4788 / 4.499e-07 | -0.308 / -1.94 | 1.15 | Raw favored | 23.83 | 0.99 | `7.2e-07 / 5.643e-06 / 1.011e-05` |
| bbb_martins | `direct_bbb / flux / cm/s / unknown` | 22 / 15 | 1.64 / 3.58 | 0.4395 / 0.167 | 0.612 / -0.693 | 0.83 | Ambiguous | -0.34 | 0.41 | `5e-06 / 1.6e-05 / 3.4e-05` |
| bbb_martins | `direct_bbb / initial_brain_uptake / %·ID/g / unknown` | 34 / 28 | 1.65 / 7.18 | 0.4377 / 0.02756 | 0.419 / -1.07 | 1.56 | Ambiguous | 3.80 | 0.70 | `0.33 / 4.475 / 12` |
| bbb_martins | `direct_bbb / kt / mM / unknown` | 102 / 71 | 3.39 / 34.6 | 0.1835 / 3.144e-08 | 0.434 / -1.53 | 2.47 | Raw favored | 31.14 | 0.94 | `0.05 / 4.55 / 14.8` |
| bbb_martins | `direct_bbb / vmax / pmol/mg·min·protein / unknown` | 26 / 25 | 3.53 / 27 | 0.1715 / 1.371e-06 | 0.805 / -2.38 | 2.60 | Raw favored | 13.74 | 0.83 | `2.25 / 345 / 900` |
| bbb_martins | `direct_bbb / vt / mL/cm^3 / unknown` | 36 / 32 | 4.11 / 5.6 | 0.1283 / 0.06082 | 0.779 / -0.524 | 1.80 | Log10 favored | -15.15 | 0.07 | `0.22 / 3.9 / 14` |
| bbb_martins | `direct_bbb / efflux_clearance / µL/brain·g·min / unknown` | 27 / 21 | 4.45 / 5.52 | 0.1079 / 0.06326 | 0.733 / -0.995 | 2.76 | Ambiguous | -17.16 | 0.07 | `1.15 / 102 / 666` |
| bbb_martins | `direct_bbb / brain_to_blood_uptake_ratio / ratio / endpoint_defined_ratio` | 20 / 9 | 5.33 / 6.3 | 0.06971 / 0.04284 | 0.373 / -0.393 | 1.37 | Ambiguous | -5.71 | 0.23 | `0.04 / 0.34 / 0.948` |
| bbb_martins | `direct_bbb / vmax / µmol/g·min / unknown` | 81 / 56 | 5.51 / 26.8 | 0.06369 / 1.511e-06 | 0.638 / -1.54 | 2.29 | Raw favored | 21.79 | 0.91 | `0.02 / 1.33 / 3.93` |
| bioavailability_ma | `fh / intrinsic_clearance / /liver·min / absolute` | 29 / 29 | 0.356 / 22.1 | 0.837 / 1.615e-05 | 0.228 / -2.12 | 3.15 | Raw favored | 23.41 | 0.92 | `0.0078 / 4.44 / 11.05` |
| bioavailability_ma | `fa / caco2_mdck_pampa_permeability / nmol/cm^2·h / absolute` | 24 / 20 | 0.503 / 29.4 | 0.7776 / 4.039e-07 | 0.306 / -2.44 | 1.61 | Raw favored | 12.89 | 0.75 | `0.21 / 3.97 / 8.61` |
| bioavailability_ma | `fa / intestinal_absorption / µM/g dry-weight / unknown` | 20 / 20 | 0.786 / 4.3 | 0.675 / 0.1165 | -0.418 / -0.962 | 0.41 | Ambiguous | 2.88 | 0.91 | `218.9 / 437.4 / 562.7` |
| bioavailability_ma | `oral_exposure / dose_normalized_exposure / h·µg/mL / absolute` | 26 / 21 | 1.37 / 19.6 | 0.5042 / 5.624e-05 | 0.172 / -1.91 | 1.58 | Raw favored | 12.77 | 0.88 | `4.47 / 74.9 / 170.5` |
| bioavailability_ma | `fa / intestinal_absorption / nmol/mg·min / absolute` | 25 / 21 | 2.12 / 2.14 | 0.3464 / 0.3436 | 0.253 / -0.509 | 0.39 | Ambiguous | 0.74 | 0.63 | `0.171 / 0.267 / 0.424` |
| bioavailability_ma | `fh / intrinsic_clearance / CYP2C19·µL/min·pmol / absolute` | 21 / 19 | 2.87 / 4.15 | 0.2384 / 0.1255 | 0.692 / -0.399 | 2.20 | Log10 favored | -15.47 | 0.09 | `0.016 / 0.48 / 2.52` |
| bioavailability_ma | `fh / ugt_metabolism / nmol/h·mg / absolute` | 22 / 22 | 3.53 / 3.7 | 0.1711 / 0.1572 | 0.583 / -0.144 | 0.93 | Log10 favored | -5.36 | 0.13 | `48.9 / 139.5 / 415` |
| bioavailability_ma | `fa / solubility / g/100 g / unknown` | 24 / 24 | 3.61 / 7.54 | 0.1648 / 0.02303 | 0.752 / -1.34 | 5.71 | Ambiguous | -49.74 | 0.06 | `0.00024 / 25.8 / 123` |
| bioavailability_ma | `fa / caco2_mdck_pampa_permeability / %/h / absolute` | 27 / 24 | 3.78 / 4.21 | 0.1512 / 0.1218 | 0.85 / -0.536 | 1.96 | Log10 favored | -13.02 | 0.14 | `0.66 / 18.1 / 60` |
| bioavailability_ma | `oral_exposure / c24 / nM / absolute` | 21 / 18 | 4.01 / 9.39 | 0.1345 / 0.009141 | 1 / -0.186 | 2.05 | Log10 favored | -28.46 | 0.01 | `6 / 83.6 / 680` |
| bioavailability_ma | `fa / intestinal_absorption / µg/min / absolute` | 37 / 34 | 4.43 / 5.52 | 0.1094 / 0.06337 | 0.743 / -0.718 | 3.14 | Log10 favored | -42.02 | 0.03 | `0.03 / 4.78 / 41.7` |
| bioavailability_ma | `fa / dissolution / %·dose·of/h / absolute` | 32 / 24 | 5.05 / 14.8 | 0.08005 / 0.0006016 | -0.567 / -1.64 | 1.51 | Raw favored | 21.57 | 0.99 | `1.405 / 31.25 / 45.85` |
| bioavailability_ma | `fh / intrinsic_clearance / L/h / unknown` | 21 / 20 | 5.24 / 35.7 | 0.07287 / 1.74e-08 | 1.18 / 0.184 | 5.14 | Log10 favored | -156.11 | 0.00 | `0.032 / 0.096 / 4441` |
| bioavailability_ma | `fa / dissolution / ppm / absolute` | 23 / 20 | 5.81 / 9.32 | 0.05478 / 0.009448 | 1.15 / -0.456 | 2.92 | Log10 favored | -37.99 | 0.02 | `1 / 134 / 833` |
| bioavailability_ma | `fa / intestinal_absorption / µmol/min / absolute` | 51 / 41 | 5.88 / 29.8 | 0.05289 / 3.456e-07 | 0.489 / -1.82 | 3.60 | Ambiguous | 2.75 | 0.51 | `0.0065 / 7.1 / 25.7` |
| bioavailability_ma | `fh / intrinsic_clearance / µL/mL·min / absolute` | 34 / 31 | 5.94 / 9.66 | 0.05126 / 0.007985 | 0.9 / -0.21 | 1.89 | Log10 favored | -31.41 | 0.00 | `1.1 / 12.5 / 85` |
| skin_reaction | `skin_exposure / dermal_absorption / mg/dL / unknown` | 27 / 24 | 0.318 / 35.6 | 0.8531 / 1.891e-08 | -0.124 / -2.91 | 2.35 | Raw favored | 33.17 | 0.94 | `0.4 / 45 / 90` |
| skin_reaction | `skin_exposure / cumulative_permeated_amount / µg permeated Cr cm-2 skin / unknown` | 30 / 13 | 0.851 / 1.08 | 0.6536 / 0.5829 | 0.123 / -0.372 | 0.40 | Ambiguous | 0.63 | 0.63 | `0.1 / 0.17 / 0.25` |
| skin_reaction | `skin_exposure / cumulative_permeated_amount / mg/h / unknown` | 20 / 8 | 1.34 / 1.41 | 0.5106 / 0.4949 | 0.412 / -0.435 | 0.78 | Ambiguous | -0.99 | 0.39 | `0.1042 / 0.3125 / 0.625` |
| skin_reaction | `skin_exposure / flux / µmol × 10^-5/day / unknown` | 24 / 20 | 1.39 / 2.04 | 0.4991 / 0.3608 | 0.243 / -0.648 | 0.95 | Ambiguous | 1.07 | 0.60 | `5 / 19.5 / 45` |
| skin_reaction | `skin_exposure / flux / µmoles·10^-5/day / unknown` | 24 / 20 | 1.39 / 2.04 | 0.4991 / 0.3608 | 0.243 / -0.648 | 0.95 | Ambiguous | 1.07 | 0.60 | `5 / 19.5 / 45` |
| skin_reaction | `skin_exposure / flux / µmol 100 g-1 h-1 / unknown` | 23 / 22 | 1.61 / 15.4 | 0.4468 / 0.0004592 | 0.589 / -1.69 | 1.71 | Raw favored | 6.58 | 0.78 | `0.6 / 12.1 / 31` |
| skin_reaction | `skin_exposure / permeability_coefficient / g/m^2h / unknown` | 28 / 27 | 2.05 / 13.4 | 0.3581 / 0.00122 | 0.583 / -1.62 | 2.79 | Ambiguous | 5.54 | 0.65 | `0.7 / 137 / 427` |
| skin_reaction | `phototoxicity_irritation_local_damage / inflammatory_response / per 10 HPF / unknown` | 23 / 17 | 2.09 / 4.5 | 0.3524 / 0.1053 | 0.676 / -0.993 | 1.89 | Ambiguous | -1.60 | 0.39 | `6 / 140.1 / 462` |
| skin_reaction | `skin_exposure / skin_retention / dpm / unknown` | 23 / 22 | 2.16 / 10.7 | 0.3391 / 0.004754 | 0.135 / -1.56 | 2.40 | Raw favored | 9.56 | 0.80 | `288.8 / 3.356e+04 / 7.256e+04` |
| skin_reaction | `skin_exposure / dermal_absorption / % dose/cm^2 / unknown` | 21 / 20 | 2.71 / 3.21 | 0.2574 / 0.2004 | 0.705 / -0.755 | 3.06 | Ambiguous | -18.15 | 0.11 | `0.002 / 0.75 / 2.28` |
| skin_reaction | `phototoxicity_irritation_local_damage / inflammatory_response / mL / unknown` | 29 / 20 | 2.89 / 15.4 | 0.2353 / 0.0004542 | 0.0203 / -1.56 | 1.20 | Raw favored | 10.28 | 0.93 | `0.5 / 3.7 / 7.9` |
| skin_reaction | `skin_exposure / flux / µmol.h^-1 / unknown` | 24 / 11 | 2.93 / 7.43 | 0.2316 / 0.02433 | 0.794 / -1.06 | 1.39 | Ambiguous | -1.75 | 0.32 | `0.05 / 0.345 / 1.23` |
| skin_reaction | `skin_exposure / flux / µl/cm^2/h / absolute` | 25 / 24 | 3.29 / 21.3 | 0.1931 / 2.39e-05 | 0.653 / -0.358 | 2.45 | Log10 favored | -29.32 | 0.05 | `0.06 / 4.8 / 16.9` |
| skin_reaction | `skin_exposure / skin_retention / µmol/mg / unknown` | 22 / 18 | 3.64 / 4.95 | 0.1621 / 0.08421 | 0.935 / 0.0655 | 1.27 | Log10 favored | -12.99 | 0.02 | `0.017 / 0.0715 / 0.319` |
| skin_reaction | `skin_exposure / flux / µmole cm^-2 hr^-1 / absolute` | 27 / 25 | 3.83 / 9.6 | 0.1474 / 0.008239 | 0.624 / -0.0555 | 1.08 | Log10 favored | -8.96 | 0.08 | `1.9 / 6.8 / 22.9` |
| skin_reaction | `phototoxicity_irritation_local_damage / phototoxicity_or_photosensitivity / J/cm2 / unknown` | 25 / 11 | 3.94 / 20.4 | 0.1396 / 3.792e-05 | 0.931 / 0.194 | 1.28 | Log10 favored | -18.22 | 0.02 | `4 / 10 / 77` |
| skin_reaction | `skin_exposure / flux / µL/h / unknown` | 25 / 18 | 4.15 / 10.3 | 0.1256 / 0.005887 | -0.946 / -1.55 | 1.40 | Raw favored | 19.32 | 0.98 | `0.5 / 9 / 12.5` |
| skin_reaction | `skin_exposure / dermal_absorption / nmol / unknown` | 24 / 24 | 4.29 / 8.53 | 0.1169 / 0.01406 | 1 / -0.325 | 2.81 | Log10 favored | -36.63 | 0.01 | `2.2 / 190.5 / 1421` |
| skin_reaction | `phototoxicity_irritation_local_damage / phototoxicity_or_photosensitivity / h / unknown` | 26 / 15 | 4.31 / 5.38 | 0.1157 / 0.06797 | 0.955 / -0.52 | 1.98 | Log10 favored | -15.66 | 0.07 | `1 / 24 / 96` |
| skin_reaction | `skin_exposure / skin_retention / µg/mg dry SC / unknown` | 24 / 18 | 4.55 / 5.98 | 0.1026 / 0.05025 | 0.808 / 0.179 | 1.18 | Log10 favored | -13.91 | 0.02 | `0.414 / 1.4 / 6.3` |
| skin_reaction | `skin_exposure / skin_exposure_measurement / ppb / unknown` | 24 / 21 | 5.27 / 14.1 | 0.07182 / 0.0008845 | 1.13 / 0.0541 | 3.26 | Log10 favored | -70.01 | 0.00 | `0.6 / 25 / 1100` |
| skin_reaction | `skin_exposure / cumulative_permeated_amount / µg/cm^3 / unknown` | 36 / 33 | 5.74 / 10.7 | 0.0566 / 0.004754 | 0.984 / -1.15 | 3.52 | Ambiguous | -26.76 | 0.03 | `0.11 / 90.47 / 363.2` |

## Input lineage

| Task | Stage-02 pretransform records | SHA-256 | Policy | SHA-256 |
|---|---|---|---|---|
| bbb_martins | `outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7/02_canonicalized/records.parquet` | `4bc7812ef3db3681716d5e544a2d06dda794fe7f07d3cd8311d07c4bfcd20b82` | `tools/chembl_tool/tasks/bbb_martins/data_processing/assay_transfer_measurements_v2/policy.json` | `2da16523c837c79efecfadbbc894cefdb328b136d27adca7b475d021e14ee64e` |
| bioavailability_ma | `outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v7/02_canonicalized/records.parquet` | `50a340db529671e6a90ea12599ea6641f866bdf01bb24b0dccd4699a9af80ab0` | `tools/chembl_tool/tasks/bioavailability_ma/data_processing/assay_transfer_measurements_v2/policy.json` | `df7283b946f3ad6b0b6af615513a91301e45fe1249be250726bd29c8c3bfda2d` |
| skin_reaction | `outputs/chembl_tool/tasks/skin_reaction/evidence_library/starling_normalized_v7/02_canonicalized/records.parquet` | `97de37a544eca86fe639ad4c5681072ed9491dd60c9a497b4010e62e15032dcd` | `tools/chembl_tool/tasks/skin_reaction/data_processing/assay_transfer_measurements_v2/policy.json` | `c19739c16f1a1b5cca0080104eecececd0737654ea11acdcf192a36c566e3cae` |

The Stage-02 artifacts still contain the pre-rebuild canonical tuple, but the audit reads only `assay_transfer_pretransform_scalar_value`; those values are unaffected by the pending transform-policy rebuild.
