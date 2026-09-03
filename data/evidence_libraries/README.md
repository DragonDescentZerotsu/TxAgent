# Evidence libraries

Each task stores live payloads under a simple release name such as `v7` or
`v8`. The tracked `CURRENT` file selects the default release. Large payloads
are ignored by Git; portable compressed releases and their archive manifests
live under `data/artifacts/evidence_library_compressed/`. Releases without a
compressed archive do not get a directory there.
