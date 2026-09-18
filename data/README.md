# Data

This directory is the canonical boundary for acquired data, gold labels,
evidence libraries, their construction code, and provenance artifacts.

- `raw/` contains immutable acquired inputs.
- `gold_labels/` contains active, versioned benchmark labels and retired label
  constructions under `gold_labels/legacy/`.
- `evidence_libraries/` contains active, versioned built libraries.
- `caches/` contains active, reusable cross-cutting caches.
- `artifacts/` contains compact audit records and receipts that are not required
  build or runtime inputs.
- `processing/` contains the code that constructs those products.
- `legacy/` preserves retired lineages without making them active defaults.

The `data/` root intentionally contains only these categories. Historical
top-level aliases were removed; new builders must use the canonical paths
exposed by `data.processing.paths`.
