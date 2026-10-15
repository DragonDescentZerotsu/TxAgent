#!/usr/bin/env bash
set -euo pipefail
cd /vast/projects/myatskar/design-documents/joseph/TxAgent
PYTHON=${TXAGENT_PYTHON:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/bin/python}
STUDY=${TXAGENT_STUDY:-outputs/analysis/retrieval_cache/molecular_policy_20261015T195506Z}
export HF_HOME=${HF_HOME:-/vast/projects/myatskar/design-documents/hf_home}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
phase=${1:?Use inventory, evidence, prepare, reuse, worksets, tokenize, score, finalize or publish}
shift
if [[ "$phase" == inventory ]]; then
    exec "$PYTHON" -m predict.retrieval.assay_reranking.rebuild_policy inventory --study "$STUDY" \
      --source-catalog predict/retrieval/assay_reranking/flat_v5_manifests/parent_disjoint_parent100_v1/catalog.json \
      --source-catalog predict/retrieval/assay_reranking/flat_v5_manifests/scaffold_disjoint_parent100_v2/catalog.json "$@"
elif [[ "$phase" == evidence ]]; then
    for task in skin_reaction bioavailability_ma bbb_martins ames dili carcinogens; do
        release=v10_main_universe_v3
        [[ "$task" != skin_reaction ]] || release=v10_main_universe_v5
        manifest="data/evidence_libraries/$task/$release/retrieval_projection/ranked_evidence_source_v1/VERSION.json"
        [[ ! -f "$manifest" ]] || continue
        "$PYTHON" -m predict.retrieval.assay_reranking.build_ranked_uid_retrieval build-evidence \
          --task "$task" --evidence-release "$release" --evidence-manifest "$manifest" \
          --profile "flat_v5/gold_v1/$task/l2plus/assay_transfer/source_v1/parent_disjoint_parent100_v2" \
          --molecular-representation-policy query_parent_evidence_source.v1 "$@"
    done
else
    exec "$PYTHON" -m predict.retrieval.assay_reranking.rebuild_policy "$phase" --study "$STUDY" "$@"
fi
