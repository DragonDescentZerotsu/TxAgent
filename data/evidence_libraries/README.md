# Evidence libraries

Each task stores live payloads under a simple release name such as `v7` or
`v8`. The tracked `CURRENT` file selects the default release. Large payloads
are ignored by Git; portable compressed releases and their archive manifests
live under `data/artifacts/evidence_library_compressed/`. Releases without a
compressed archive do not get a directory there.

BBB and Bioavailability runtime level assignment is owned by each active
evidence-library release under `<task>/<release>/level_mapping/`. The compact
`level_mappings.v1.json` index identifies the current reviewed mappings and
their manifests. Gold-owned mapping publications remain preserved for lineage
and explicit gold-version workflows, but they are not the runtime authority for
these two tasks.
