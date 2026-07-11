# Paper Pipeline and Data Flow

## Separation of responsibilities

There is one shared molecular evidence agent pipeline and thin task configurations. There is no separate Bioavailability execution engine.

Shared modules own source ingestion, the minimal evidence contract, molecule-level indexing, experiment views, identity-blind tool prefetch, validated LLM calls, batch execution, metrics, and statistical analysis. Each task owns only endpoint-to-mechanism mappings, task prompts, and the final prediction schema.

Bioavailability is special only in data availability and its declared evidence families: Observed direct F, Observed oral exposure, Fa, Fg, and Fh. BBB, Skin_Reaction, and ClinTox use the same execution path with their own mechanism families.

## Offline evidence construction

```text
ChEMBL assay/activity rows or Starling parquet/HF rows
    -> source profile parser
    -> source rows with endpoint, value, units, conditions, text, quality, provenance
    -> minimal_evidence.v1 attachment
    -> aggregate duplicate rows by source molecule and endpoint group
    -> standardize source molecule structure
    -> Morgan fingerprint
    -> molecule-level neighbor index
```

The minimal contract is descriptive, not predictive. It preserves:

- source and record provenance
- source molecule identity for harness-side indexing
- endpoint and scalar measurement when available
- evidence and condition text
- evidence role, scope, uncertainty, and source quality
- representative source examples

It does not assign the benchmark label or decide whether evidence transfers to a query.

## Query-time retrieval

```text
query SMILES (harness only)
    -> standardize structure and compute Morgan fingerprint
    -> select task/source experiment view
    -> rank source molecules independently within each selected group
    -> exclude exact query molecule by canonical structure/InChIKey
    -> retain top 3 neighbors with similarity >= 0.30
```

The experiment views are:

- `none`: no analog evidence
- `direct`: direct task-outcome groups only
- `full_flat`: all selected mechanism evidence merged into one group
- `full_mechanism`: the identical evidence union separated by mechanism family

Flat is generated from the already selected mechanism view, so flat and mechanism receive the same evidence rows. This isolates organization/decomposition from evidence availability.

## Identity-blind harness

The LLM never receives query or neighbor structures or identifiers in reported runs.

```text
query and retrieved neighbor structures
    -> harness calls molecule_properties for query
    -> harness calls mmp_structure_compare and properties_compare for each pair
    -> harness removes structures, InChIKeys, molecule IDs, known molecule names,
       and identities embedded in evidence text
    -> LLM receives aliases, similarities, source evidence, and prefetched tool text
```

Raw `retrieval.json` remains available for provenance and debugging, but only the redacted copy is used to construct LLM messages. The report audits the actual saved request histories and checks that query/neighbor structures, molecule identifiers, and known source names never appear in system, user, or tool inputs. Model-generated assistant text is excluded from this upstream-leak test.

## Reasoning and synthesis

```text
prefetched query properties
    -> single-molecule physicochemical prior (one frozen result per task/query)

each evidence group + analog comparisons
    -> parallel group transferability reasoning
    -> preserve raw group output for audit
    -> remove any source identities inferred by the group model

frozen single prior + sanitized group outputs + retrieval coverage
    -> final task-specific binary prediction
    -> JSON schema validation
    -> up to four total attempts for invalid structured output
       (the final two only reserialize the same JSON evidence)
    -> if a cleaned group payload exceeds 750 KB, deterministically sample at most
       100 evenly spaced evidence rows per oversized neighbor and record the original
       row count, retained row count, byte count, and sampling method in the prompt
```

The frozen single prior is reused by every retrieval condition for the same task/query. Thus paired ablations change only retrieved evidence and evidence grouping.

Identity control is enforced at both LLM boundaries. A preflight gate rejects any known structure, identifier, or source molecule synonym left in a group prompt. Raw group responses are saved separately because a model can infer an analog name from distinctive evidence even when the prompt is redacted; the version passed to final synthesis is sanitized again against the retrieval-wide synonym set.

## Task configurations

### Bioavailability_Ma

- Direct: absolute oral bioavailability F
- Observed exposure: oral AUC/Cmax and food/formulation context
- Fa: absorption, intestinal permeability, solubility, dissolution, GI stability
- Fg: intestinal efflux and transporter evidence
- Fh: first-pass extraction, hepatic/intrinsic clearance, metabolic stability

Starling has all five families. ChEMBL endpoint groups are projected into the same families.

### BBB_Martins

- Direct brain exposure/BBB outcome
- Passive permeability
- Efflux transport
- Influx transport

ChEMBL supports all families. Starling is evaluated for direct evidence only.

### Skin_Reaction

- Direct skin reaction
- Sensitisation AOP
- Phototoxicity, irritation, and local damage
- Skin exposure
- Context/background

### ClinTox

- Clinical human safety
- In vivo toxicology
- Organ-specific toxicity
- Genotoxicity/carcinogenicity
- Cellular stress pathways
- General cytotoxicity
- Off-target, DDI, and exposure context

## Evaluation outputs

Each batch writes per-query retrieval, single/group/final outputs, trace messages, predictions, metrics, and a manifest. The paper summarizer adds:

- accuracy, macro-F1, confusion matrix, and 95% bootstrap interval
- overall and per-group retrieval coverage
- prompt/completion tokens, actual served model, retries, and reused-prior count
- query-SMILES trace audit and prompt-boundary structure/identifier/name audit
- paired macro-F1 deltas and 95% paired bootstrap intervals
- exact two-sided McNemar tests

The scalar Bioavailability KNN control uses only numeric direct-F values, the same top-3/minimum-similarity retrieval rule, a 20% threshold per neighbor, and majority voting. It has no access to conditions or qualitative literature text.
