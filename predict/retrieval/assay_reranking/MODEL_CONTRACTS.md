# Assay-reranker model and prompt contracts

The executable model revisions live in `runtime.py`, `v27_safety.py`, and
`v27_skin.py`; completed cache `VERSION.json` files pin the selected model,
projection, template, and source hashes. This index groups every active family
without replacing those immutable receipts. Historical releases stay unchanged.

| Family | Dataset / adapter | Template | Candidate universe | Current validation |
|---|---|---|---|---|
| BBB/Oral/Skin TDC direct | Pinned context-conditioned TDC datasets; `score_tdc_ranked_retrieval.DirectL1PromptRenderer` | `prompts/direct_l1_task_best_v1/tdc.jinja` | Existing L1 Morgan-100 parents/cards | Existing published caches and fixtures |
| Ames/DILI/Carcinogens Gold direct | Pinned conditioned Gold-v1 datasets; same direct renderer | `prompts/direct_l1_task_best_v1/gold.jinja` | Existing L1 Morgan-100 parents/cards | Existing published caches and fixtures |
| Ames/DILI/Carcinogens TDC direct successors | Starling V10.3 TDC mixed-canonical datasets; same direct renderer, TDC binary labels | `prompts/direct_l1_task_best_v1/tdc.jinja` | Copy `ranked_level_retrieval_tdc_v1` L1 Morgan-100 rows | 384 real prompt comparisons per model |
| BBB V24.1 L2-L4 | Per-level V24.1 datasets; `v24_1_levels.V241PromptRenderer` | `prompts/v24_1_bbb/prompt.jinja` | Published level-specific cache contract | Historical; not re-audited here |
| Oral V25 L2/L3/L4/L6 | Per-level V25 datasets; `v25_oral_levels.V25OralPromptRenderer` | `prompts/v25_oral/prompt.jinja` | Published level-specific cache contract | Historical; not re-audited here |
| Skin V27 predecessor | V27 combined dataset; `v27_skin.SkinV27PromptRenderer` legacy mode | `prompts/v27_skin/prompt.jinja` | Published Skin L2/L3 Morgan-100 UID rows | Preserved, known prompt mismatch |
| Skin V27 successor | Same V27 combined dataset; training-mode Skin renderer with canonical endpoint and V19 field binding, plus candidate-context inference query copy | `prompts/v27_skin/prompt_training.jinja` | One top-100 Morgan parent list per query from the combined L2+ pool, expanded to unique physical UIDs | 384 real prompt comparisons |
| Ames/DILI/Carcinogens V27 general | Task-specific V27-general datasets; `v27_safety.SafetyV27PromptRenderer`, Starling V10.3 safety visibility and Ames semantic display, plus candidate-context inference query copy | Same V20 training template as Skin successor | One shared top-100 Morgan parent list per query by default; Gold Carcinogens and TDC AMES have separate top-40-prefix successors; levels only partition storage | 384 real prompt comparisons per model |
| Earlier V9/V10.3/V10.4 direct and V19.1/V21 record modes | Pinned role mappings in `runtime.MODEL_PROFILES` and the corresponding `v9.py`, `v19_1.py`, `v21_bbb.py` builders | Their versioned `prompts/` assets | Their published cache contracts | Historical; not re-audited here |

The seven successor checkpoint identities are:

| Role | Model revision | Dataset identity |
|---|---|---|
| TDC Ames direct | `jiosephlee/intern-s1-mini-ames-v10-3-tdc-mixed-canonical-best@392dd01912090040cfc0427f9f680d16d47eb27b` | `context-conditioned-molecule-transfer-v10.3-tdc-ames-mixed-canonical-intern@3bd32bf7c6eba916cdcc358425cf61ebb5e6ae7e` |
| TDC DILI direct | `jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-tdc-pinned-dili-xnkoqrux@bb5248609a729ac7dc6d6c9a3cccba1566a3d19e` | `context-conditioned-molecule-transfer-v10.3-tdc-dili-mixed-canonical-intern@401d41b27d87d608ce03a45fed7d43d081229007` |
| TDC Carcinogens direct | `jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-tdc-pinned-carcinogens-2xcwcpul@8be3376cd1d84e46c18b61617f63ad597aeaff52` | `context-conditioned-molecule-transfer-v10.3-tdc-carcinogens-mixed-canonical-intern@c21748b1dcb036082014ac51f3bbcbd20872e5c2` |
| Ames V27 general | `jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-ames-general-best@e731fd48ebebe95aa621b3500e59f28d4c27b036` | `assay-transfer-record-level-v27-ames-general-intern`; local release manifest SHA `6f2d2a3d80869e4c604d193568e3b9373a75200e9e5a9f9736f0f593e3be2c75` |
| DILI V27 general | `jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-dili-general-best@e34d922504cc67290ce5289607f54e5d74f6868e` | `assay-transfer-record-level-v27-dili-general-intern`; local release manifest SHA `8bd8cba9c059a441138a093d02d9d94d5298b77203c70fce0cfd204a35e01e1f` |
| Carcinogens V27 general | `jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-carcinogens-general-best@13a182fbc9f8b2a49c5286514dd5b6110393544b` | `assay-transfer-record-level-v27-carcinogens-general-intern`; local release manifest SHA `408548b285f32523ac190f9d0340c39049a8e8d7af83d895b59775f80478476c` |
| Skin V27 combined | `jiosephlee/intern-s1-mini-assay-transfer-record-level-v27-skin-reaction-combined-best@398a41ccb38cadd120ee26b989bdbce5cab6c021` | `assay-transfer-record-level-v27-skin-reaction-combined-intern@1fa33cf3a7df7b6bd07a3b959f37502f951fc8ed` |

Run `python -m predict.retrieval.assay_reranking.audit_training_prompts --report
outputs/analysis/assay_reranking/v27_candidate_copy_v1/prompt_audit.json`
before scoring. It checks sampled published prompts byte-for-byte and separately
checks candidate-context copying on the indirect hidden-query side. Training
pair rendering retains the exact Starling query fields; SMILES-only inference
copies the candidate's query-visible fields using the same Starling visibility
filter. This keeps the field surface closer to training but does not make the
copied context the query's own assay record. The `candidate_copy_v1` identity
records that distinction; the proposed structured-only variant was not
published. A mismatch blocks scoring; the audit emits a compact JSON receipt,
TSV, and report.

Only scoring runs on DGX GPUs. Prepare and validate once, score four disjoint
shards with `srun --jobid=<free-dgx027-allocation> --overlap --ntasks=4
--gpus-per-task=1`; each Slurm rank sees its assigned GPU as device `0`, so pass
`--device 0 --num-shards 4 --shard-index "$SLURM_LOCALID"`. Use a high measured
device batch size and let the scorer halve batches for long padded prompts.
For V27 safety, `build_safety_v27_ranked_retrieval tokenize` can run on EPYC
with `--output-root` pointing to a closed copy of the prepared cache and
`--tokenized-root` pointing to a separate sidecar. It batch-tokenizes the exact
`(A)`/`(B)` scoring prefixes, checks samples against live tokenization, and
atomically writes hash-pinned per-shard Parquet. Pass that same sidecar through
`--tokenized-root` on later `score` commands; omit it to retain the original
scorer. Existing running scorers and journals are unaffected.
Stage resumable journals on that node's `/local`, copy only closed trees to
`/vast`, validate there, then publish under the semantic `flat_v5/...` identity.
Pass a closed prior stage with `--score-reuse-root` during preparation to reuse
matching score keys from a finalized SQLite cache or a validated journal prefix;
the prompt and model key must match, and the reused stage is hash-pinned in the
new cache manifest. The old per-level candidate stages are reuse sources only,
never valid shared-parent releases.
Do not activate a bundle or revise a tracked artifact manifest until all its
levels and the combined inference read are validated.

The opt-in harness configurations are
`ranked_level_retrieval_gold_v1_v27_successors_v1.yaml` and
`ranked_level_retrieval_tdc_v1_v27_successors_v1.yaml`. They combine the
preserved BBB/Oral and Gold direct releases with the successor Skin and safety
indexes; TDC safety L1 uses the new V10.3 direct indexes. For successor
Skin/safety L2+ caches, the full-flat reader displays every physical UID under
the query's shared selected parents, partitioned by level but with no record quota.
The Gold Carcinogens and TDC AMES top-40 successors use `--parent-capacity 40`
and a hash-pinned `--source-parent-universe` from their frozen top-100 stages;
their new release paths do not replace the top-100 stages or activate a bundle
before validation and publication.
For an optional top-50 successor, use `--supplement-41-50` to prepare only
parent ranks 41-50 from the frozen top-50 `PARENT_UNIVERSE.json` on each split.
The supplement is score-only: `prepare-universe`, `prepare-level`, and `tokenize`
can run on EPYC without touching top-40 journals, while `write-index` and
`validate` are deliberately unavailable. Tokenize with the same shard count
intended for later GPU scoring. Do not queue that scoring automatically. After
both score sources close, prepare a fresh `--parent-capacity 50` release with
`--score-reuse-root` pointing to the top-40 and supplement roots; exact
prompt/model keys supply reuse, and only that full release may be validated
and published for inference.
Prompt-size checks are not cache-generation or ranking-validation gates. They deliberately
fail preflight while any required successor artifact is absent. The historical
configurations and Flat-V5 artifact manifests remain unchanged until the full
successor bundle has been verified.
