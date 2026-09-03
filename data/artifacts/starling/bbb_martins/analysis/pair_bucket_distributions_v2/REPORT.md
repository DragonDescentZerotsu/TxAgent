# BBB v2 pair-bucket distribution review

## Scope

This audit uses an isolated rebuild of BBB Stages 02-04 with
`exact_measurement_unit_map.v2.json` (SHA-256
`5531dde4ce475f199fe5c0b7c7b90025293424d67d24413763d23a5d40dbcd5c`).
The production Stage 04 still points to the older v1 map and was not modified.

The figure contains all 9,322 continuous pair buckets with finite values
(35,768 records). The 59,169 binary and 1,381 non-scalar buckets are counted in
the figure footer but omitted because they have no numeric magnitude
distribution.

For each positive-valued bucket, values are transformed with `log10`; buckets
containing zero or negative values use the raw value. Each bucket is then
min-max normalized to `[0, 1]`. Singletons and constant buckets appear at 0.5.
Rows are ordered by their largest supported internal gap, requiring at least
`max(2, 5%)` observations on both sides. The CSV maps every plot row back to the
complete pair-bucket key and its source-unit composition.

## Initial findings

- 1,029 buckets have enough observations to define a supported internal gap.
- 100 buckets have a positive-value gap of at least one decade (10-fold).
- 10 buckets have a gap of at least two decades (100-fold).
- 862 buckets combine more than one extracted input-unit spelling or scale.
- 325 buckets have input-unit-specific medians at least one decade apart. This
  is a screening signal, not proof of a mapping error.

Three records are confirmed double applications of an exponent. In each case,
the extracted measurement is already the true scaled value, while the extracted
unit repeats the scientific factor and v2 applies it again:

| canonical record | extracted pair | published scalar | expected scalar |
|---|---:|---:|---:|
| `833364160d884ab425fa4686f56179305fffab882cf086df5b1eb55d6372c12d` | `1.9e-06`, `10^-6 cm/s` | `1.9e-12 cm/s` | `1.9e-06 cm/s` |
| `df94bf0b7909534ecbb6e826535d5f6dafedca27f5caaa9011d55cdcb481121e` | `8.7e-07`, `10^-6 cm/s` | `8.7e-13 cm/s` | `8.7e-07 cm/s` |
| `429090f9c4510d86df3695bbf5c7f1345cfd74a833cbcd1664b1d642203c9286` | `1e-05`, `10 x 10^-6 cm/s` | `1e-10 cm/s` | `1e-05 cm/s` |

The first two create the largest plotted gap: `1.9e-12` to `1.3e-7 cm/s`
inside one MDCK-MDR apparent-permeability bucket.

The other nine two-decade gaps are not explained by an obvious scale
double-application. The most important non-unit warning is a `/min`
`blood_to_brain_transport` bucket that combines `k1`, `k2`, and `k3` kinetic
parameters; its 0.001-0.45/min records and 66-150/min records are dimensionally
compatible but not clearly the same measurement concept. A
`plasma_to_brain_ratio` record at 0.00825 also has source wording whose direction
looks internally inconsistent. These need endpoint/reference review rather than
unit-scale correction.

## Artifacts

- `bbb_pair_bucket_distributions.png`: all numeric buckets in one heatmap.
- `bbb_pair_bucket_distribution_rows.csv`: one row per plotted bucket.
- `bbb_pair_bucket_gap_ge_2_decades_rows.csv`: all 126 source records in the ten
  buckets with at least a two-decade gap.
- `summary.json`: input/output hashes and coverage counts.
