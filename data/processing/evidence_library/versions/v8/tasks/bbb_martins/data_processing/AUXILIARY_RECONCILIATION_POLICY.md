# BBB Martins Auxiliary Reconciliation Policy

This policy governs the reviewed second pass over BBB Martins auxiliary labels. Its goal is a compact ontology that preserves experimental mechanism. There is no fixed vocabulary-size target: labels are combined only when the source evidence supports the same scientific context.

The machine-readable authority is `auxiliary_reconciliation_policy.json` in this directory.

## Independent namespaces

Review these six namespaces independently:

- `direct_bbb/global_context`
- `direct_bbb/global_species_context`
- `passive_permeability/global_context`
- `passive_permeability/global_species_context`
- `efflux_transport/global_context`
- `efflux_transport/global_species_context`

A decision must belong to exactly one namespace. A matching label in another source or output does not authorize a cross-namespace merge; it requires a separate decision. Context and species labels must never be reconciled together.

## Ontology rules

Context labels retain explicit, decision-relevant distinctions:

- Assay or model type.
- In vivo versus in vitro setting.
- Transport mechanism and direction.
- Barrier-disruption state.
- Major BBB modality, such as PET, MRI, autoradiography, microdialysis, brain perfusion, PAMPA-BBB, or an endothelial monolayer.

Remove compound, target or transporter, dose, time, pathogen, clone, and protocol detail unless that detail is essential to one of the preserved dimensions. Prefer the shortest label that retains those dimensions. Do not collapse mechanistically different evidence merely to make the vocabulary smaller.

Species labels use concise lowercase common names. Deduplicate and alphabetize multiple explicit species, joined with ` + `. Use a role-qualified form such as `insect host; human gene` only when the raw value explicitly states both the expression host species and the introduced gene species. If both roles are not explicit, do not invent a role from a cell line, assay abbreviation, gene symbol, transporter name, or outside knowledge.

## Review actions and evidence

Each reviewed item receives one of four actions:

- `retain`: keep one provisional label unchanged.
- `merge`: map two or more equivalent labels in one namespace to one canonical label.
- `rename`: replace one label with a clearer label without changing its raw-value membership.
- `split`: divide one provisional label using explicit, exhaustive, deterministic, and non-overlapping raw-value assignments.

Every decision records its ID, namespace, action, input and final labels, affected raw-value count, representative raw examples, evidence summary, rationale, preserved dimensions, removed details, and reviewer, checker, and adjudicator identities. Merge and rename evidence must cover every input label. A split must cover every affected raw value exactly once, show evidence for every child label, and avoid inferring absent information.

## Approval workflow

The workflow is `reviewer → checker → adjudicator → human publication checkpoint`.

The reviewer proposes a fully supported decision. The checker independently verifies source support, namespace scope, ontology preservation, and coverage. The adjudicator resolves any disagreement and accepts or rejects the proposed change; an adjudicator rewrite requires a new rationale. Before publication, a human must approve the final manifest, unresolved-item report, and complete change summary.

Publication is blocked if any input lacks a disposition, an accepted change lacks checker or adjudicator provenance, a split fails coverage validation, a decision crosses namespaces, a disagreement remains unresolved, or human approval is absent. Automated suggestions may create review candidates, but they may not silently change the published mapping.
