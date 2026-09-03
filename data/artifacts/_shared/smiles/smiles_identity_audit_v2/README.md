# SMILES identity audit v2

This directory retains the reusable and non-reconstructable products of the shared
BBB, Bioavailability, and Skin name-to-SMILES audit.

Retained artifacts:

- `pubchem_cache.sqlite`: reusable external lookup cache.
- `name_extraction/full/`: parsed full-run LLM extraction tables and manifest.
- `deepseek_v4_flash_xhigh_context_gate/name_extraction/pilot/`: the compact manual
  quality gate used by the full extraction.
- `name_smiles_comparison/v1/review/v1/`: consolidated primary decisions and their
  quality audit.
- `name_smiles_comparison/v1/review/v2/checker_reviews/` and
  `adjudicator_reviews/`: compact independent decisions.
- `name_smiles_comparison/v1/review/v2/proposal/`: the approved Stage 1 ledger,
  manifest, and impact summary.

Generated packets, raw request checkpoints, deterministic comparison tables,
candidate-discovery exports, superseded pilots, and their one-off review code are
intentionally not retained. The approved Stage 1 ledger and reusable PubChem cache
remain the durable inputs; a future audit should introduce a new versioned workflow
instead of reviving the completed review implementation.
