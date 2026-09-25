# TDC-v2 Carcinogens Luna positive-gate test

Both validation-frozen mixed profiles completed all 56 test queries with no failures. The leading validation profile scored 0.878261 macro-F1 (44 TN, 1 FP, 3 FN, 8 TP); the second profile scored 0.714770. Direct-only Luna scored 0.841539, and the earlier strong-direct mixed prompt scored 0.754679. The positive-gate prompt was revised after viewing test outcomes, so this comparison is exploratory.

The first profile is published as a distinct successor arm. `results.tsv` contains both results; `provenance.json` pins the input run and copied prompt.
