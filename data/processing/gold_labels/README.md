# Gold-label processing

This package owns benchmark labels, conditioned split contracts, condition-review
logic, and benchmark publication. It consumes evidence-library outputs but is not
part of evidence-library normalization or pair-bucket construction.

The current general entry point is `build_conditioned_benchmark.py`. Historical
record-supported publication remains isolated in
`build_record_supported_benchmark.py` and does not participate in V7 or V8 builds.
