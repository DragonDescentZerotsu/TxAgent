#!/usr/bin/env bash
set -euo pipefail

project_root=/data1/tianang/Projects/TxAgent
python=/data1/tianang/anaconda3/envs/vllm/bin/python
receipt_root=outputs/paper/rl_lora_gpt_oss_120b/feasibility_20260810
requested_task="${1:-all}"

export GPT_OSS_LOCAL_API_KEY=EMPTY
cd "$project_root"

run_batch() {
  local task="$1"
  local module="$2"
  local input="$3"
  local index="$4"
  local indices="$5"
  local retrieval_name="$6"
  local profile_option="$7"
  local profile="$8"

  "$python" -m "$module" \
    --input-jsonl "$input" \
    --indices $indices \
    --index "$index" \
    --experiment-mode full_flat \
    --retrieval-source starling \
    --neighbor-identity-policy parent_disjoint \
    --identity-blind \
    --retrieval-replay-source-batch "$receipt_root/retrieval/$retrieval_name" \
    --batch-root "$receipt_root/base_context/$task" \
    --batch-id "${retrieval_name}_context" \
    --python-executable "$python" \
    --parallelism 4 \
    --max-stage-requeues 1 \
    --api-key-env GPT_OSS_LOCAL_API_KEY \
    --base-url http://127.0.0.1:9001/v1 \
    --tool-service-url http://127.0.0.1:8765 \
    --model gpt-oss-120b \
    --reasoning-effort "" \
    --disable-thinking \
    --max-tokens 4096 \
    "$profile_option" "$profile"
}

if [[ "$requested_task" == all || "$requested_task" == bbb_martins ]]; then
  run_batch \
    bbb_martins \
    tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
    data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold/train.jsonl \
    outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2/evidence/bbb_starling_full/starling_bbb_neighbor_index.pkl \
    "0 1 2 6" \
    bbb_full_flat_mixed \
    --bbb-prompt-profile \
    meaningful_cns_access_v1
fi
if [[ "$requested_task" == all || "$requested_task" == bioavailability_ma ]]; then
  run_batch \
    bioavailability_ma \
    tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
    data/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold/train.jsonl \
    outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/evidence/bioavailability_starling_full/starling_factor_neighbor_index.pkl \
    "0 1 3 4" \
    bio_full_flat_mixed \
    --bioavailability-prompt-profile \
    f20_evidence_calibrated_v2
fi

if [[ "$requested_task" == all || "$requested_task" == skin_reaction ]]; then
  run_batch \
    skin_reaction \
    tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch \
    data/processed_starling_record_supported_v2/Skin_Reaction/scaffold/train.jsonl \
    outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/evidence/skin_reaction_starling_full/starling_skin_reaction_neighbor_index.pkl \
    "0 1 5 12" \
    skin_full_flat_mixed \
    --skin-prompt-profile \
    sensitization_aligned_v2
fi
