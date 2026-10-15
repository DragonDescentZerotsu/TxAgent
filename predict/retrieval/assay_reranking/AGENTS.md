# Assay reranking

This owner builds and reads frozen candidate rankings and assay-transfer scores. Resolve cache set → bundle manifest/receipt → release index → level database → evidence projection.

- `runtime.py` resolves cache profiles; `*.yaml` binds task/level sources; `flat_v5_manifests/` pins bundles.
- `artifact_bundle.py` checks bundle receipts stored at repository `data/caches/assay_reranking/active/flat_v5/receipts/`.
- `ranked_uid_retrieval.py` selects ranked panels and hydrates physical UIDs from evidence-owned projections; builders publish their indexes and per-level payloads.
- `prompts/` contains assay-transfer scoring templates; prediction templates live in `../../prompts/`.
- Payloads live at repository `data/caches/assay_reranking/active/flat_v5/<benchmark>/<task>/<role>/<method>/<release>/<variant>/`; compact direct ranking sources also live in `../cache/assay_reranking/active/`.
- `../policies.py` defines parent/scaffold disjointness. DILI example below the payload root: `tdc/dili/l2plus/RELEASE_INDEX.json`.
- Keep missing cached scores and partial coverage explicit; select exact releases through manifests, not directory-name guesses.

- Follow [CACHE_LAYOUT.md](CACHE_LAYOUT.md) and the selected bundle manifest and receipt.
- Reuse shared retrieval readers for optimized and manual physical UID selections. Preserve L1 context labels and ordered voter membership; reject invalid candidate membership or disjointness. Specific retrieval comparisons belong in Bash recipes calling the appropriate capability CLI.
- Readers use indexed lookups and existing receipts; publication or explicit verification owns full hashes and integrity checks.
- Reject absent UIDs, incompatible query identities and score/model/template drift before selection or inference.
- Direct L1 requires scaffold-disjoint candidates; only indirect L2+ supports both disjointness policies. Select them through named cache sets and retain each release's explicit score availability. Assay-transfer contrastive selection is retired.
- Skin Gold L1 lives under `flat_v5/gold_v1/skin_reaction/l1/assay_transfer/v9_0_2/scaffold_disjoint_parent100_v1/`, including its corrected `scoring/` source. Cache relocations use the shared path resolver and `data/caches/assay_reranking/relocations.json` at the repository root, never symlinks or substitution of scientifically different releases.
- FA2 remains the scoring default. Explicit FA3 preparation and scoring use their own contract and prompt keys, validated frozen tokens and a separate release. Reject backend/contract mismatch before accepting journals, loading models or writing scores; never relabel an existing preparation.
- Never rescore or rewrite a published cache in place; specialized workflows retain their pinned request contracts.

Active prediction and indirect selection read named cache sets under `flat_v5_manifests/`; numerical YAML recipes are historical inputs. The canonical TDC dataset manifest owns query cohorts and direct-card bindings. Keep frozen retrieval L1 and optimizer `l1_cards/` separate; their scores have different scientific identities.

TDC has one active `tdc` namespace, including scaffold-disjoint successors. Historical headers are preserved through the relocation map; metadata renames preserve payload content IDs and database bytes.

Gold L1 contexts hydrate their ordered canonical voter UIDs from the membership artifact; never synthesize member IDs or distribute aggregate examples across members. Membership identity includes the context: one physical row may appear in different condition cards. Keep card labels and conditions on contexts, and source measurements on physical rows. TDC L1 members remain native training-label records. Historical Skin synthetic-member caches are blocked for new hydration; use the validated canonical-voter successor described in [Gold voter provenance](../../../docs/look/gold_voter_provenance.md). Count corrections require rescoring changed rendered prompts with the original pinned model; reuse scores only under an exact compatible prompt contract.
