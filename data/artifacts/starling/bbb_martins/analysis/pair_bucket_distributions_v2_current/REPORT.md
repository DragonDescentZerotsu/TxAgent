# BBB v2 pair-bucket distribution review

## Scope

This audit reads the current BBB Stage 03 records and the rebuilt Stage 04 pair
buckets produced with `exact_measurement_unit_map.v2.json`. The plot contains
all 9,330 continuous pair buckets with finite values (35,768 records). Each
bucket is normalized to the same `[0, 1]` plotting range only for visualization;
the artifact values and units are not changed by the plotter.

The gap screen requires at least `max(2, 5%)` observations on both sides. A
gap of two decades means adjacent supported values differ by at least 100-fold.

## Repairs and prevention

- Stage 01 now fails closed when a nonzero measurement already expresses the
  same scientific factor that remains in its unit. The full BBB scan found four
  such rows; all four have exact, source-hash-pinned reviewed repairs.
- The four repaired passive-permeability coefficients are `10`, `1.9`, `0.87`,
  and `17` with their retained `10^-6 cm/s` units. Their Stage 02 values are
  therefore `1e-5`, `1.9e-6`, `8.7e-7`, and a categorical caffeine outcome,
  rather than applying `10^-6` twice.
- The gap audit found one additional source transcription error: biotin
  `Km = 100 M`. The cited primary study reports approximately `100 µM`
  ([PMID 3098919](https://pubmed.ncbi.nlm.nih.gov/3098919/)). The cleaned source
  view and frozen measurement-resolution row are both corrected; the original
  model response is retained in the resolution manifest.
- BBB influx rows now expose one literal `canonical_kinetic_symbol` column.
  It is a pair-bucket dimension only for `/min` `blood_to_brain_transport`
  records. The 45 recognized rows form 10 keyed buckets, and no keyed bucket
  contains more than one symbol. This removes the former mixed k1/k2/k3 gap
  without reinterpreting the parameters.

## Remaining gaps of at least 100-fold

The original screen found 10 buckets. After the repairs and kinetic split,
eight remain. None is explained by an unresolved unit-map scale error.

| Source and endpoint | Unit | Gap | Classification | Evidence and next action |
|---|---:|---:|---|---|
| influx `brain_endothelial_cell_uptake` | `pmol/cells·min` | `8e-8` to `3.85e-4` | Quantity/context mixing | The low side is observed GSH uptake over 30 minutes; the high side is transporter `Vmax` for other compounds. Conversions per cell and minute are correct. A future split should use observed uptake versus `Vmax`, not change the unit map. |
| influx `blood_to_brain_transport` | `M` | `5e-4` to `1` | Endpoint/quantity mixing | The bucket combines low-affinity glucose `Km` near 1 M with lower concentration/Km values, including a tyrosine tissue concentration. The erroneous biotin `100 M` row is now a separate `100 µM` bucket. |
| direct `influx_rate_constant` | `mL/g·min` | `0.0032` to `1.1` | Plausible assay/compound range | The low side is Zn/spermidine `Kin`; the high side is antipyrine in `mL/100 g/min`, correctly divided by 100. The spermidine source typography loses an exponent sign, but the reviewed `10^-3` interpretation is physiologically coherent; this is not a v2 mapping-direction error. |
| direct `csf_to_blood_ratio` | `ratio` | `0.001` to `0.17` | Plausible biological range | The low values are crizotinib patient measurements; percentages on the upper side are correctly converted to ratios. |
| direct `tissue_to_blood_ratio` | `ratio` | `0.2` to `32.02` | Plausible biological range | The source explicitly spans 11% for etoposide to 32.02-fold for mitoxantrone ([PMID 21400119](https://pubmed.ncbi.nlm.nih.gov/21400119/)). |
| direct `plasma_to_brain_ratio` | `ratio` | `0.00825` to `1.1` | Endpoint-direction warning | Two A-317491 rows call `0.00825` plasma-to-brain even though their interpretation and the cited pharmacology describe limited brain penetration; the underlying study describes a brain-to-plasma ratio. This needs endpoint-direction review, not unit conversion. |
| direct `brain_to_plasma_cmax_ratio` | `ratio` | `0.085` to `9.3` | Plausible compound range | Low indomethacin/BTK-inhibitor exposure and high tricyclic-antidepressant accumulation use the same ratio direction; percent conversion is correct. |
| direct `permeability` | `cm/s` | `1e-7` to `1.08e-5` | Plausible permeant range | Sucrose/mannitol hydrophilic markers occupy the low side and phenytoin the high side. Scientific factors are applied once. |

## Artifacts

- `bbb_pair_bucket_distributions.png`: all numeric pair buckets in one heatmap.
- `bbb_pair_bucket_distribution_rows.csv`: one row per plotted bucket.
- `bbb_pair_bucket_gap_ge_2_decades_rows.csv`: all 70 source records in the
  eight remaining buckets with a gap of at least two decades.
- `summary.json`: exact input/output paths, hashes, and coverage counts.
