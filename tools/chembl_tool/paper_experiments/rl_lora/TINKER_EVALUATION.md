# Tinker backend evaluation

Status: account credit, supported model, client environment, renderer parity,
current pricing, and bounded hosted RL runs were verified on 2026-08-10. The
visible-prefetched 3-step rank-16 attention-only run, loss export, W&B sync,
checkpoint export, and sampler reload all passed. The user subsequently changed
the formal candidate to rank-32 all-module LoRA. The matched three-step hosted
smoke, output-length audit, W&B sync, checkpoint export, and valid-3 sampler
reload now pass. A formal hosted 120B run was subsequently launched and then
intentionally stopped by the user. Preserve its partial receipts, but do not
resume it automatically. The local NeMo 20B comparison was also subsequently
started and intentionally stopped; there is currently no active RL training
path.

Tinker is a backend comparison, not a second scientific contract. Historical
final-only sections follow E17; one-pass sections follow E18. Each must consume
the same materialized rows, private labels, deterministic reward implementation,
task sampling, generation count, token limits, and checkpoint-selection gates
as local NeMo RL. Provider-specific code belongs in one thin adapter and one
config file.

The comparison receipt must record:

- exact Tinker base-model identifier and API/client version;
- LoRA rank/alpha/target-module support and effective trainable parameters;
- whether rollout generation and optimization are billed separately;
- prompt/completion/training token counts and all charged units;
- startup latency, warm step throughput, checkpoint/export latency;
- projected wall time and price for BBB, Bioavailability, Skin, and all three;
- credit balance before and after the bounded test, without storing secrets.

## Verified account and client setup

The grant email confirms `$5,000.00` in Tinker Research Grant credits.  The
credential is stored only in the node-local file below, mode `0600`; its value
must never be copied into the repository, shell history, logs, or receipts.

```text
/local/tianang/txagent_tinker/credentials.env
```

The isolated client environment is `/local/tianang/txagent_tinker/.venv` and
contains Tinker SDK `0.25.0`, the official cookbook checkout at commit
`3d3e97670`, PyTorch `2.10.0+cu128`, and Transformers `5.5.4`.  The authenticated
server capability response lists both `openai/gpt-oss-120b` and
`openai/gpt-oss-120b:peft:131072`; the normal 32K model is sufficient for the
historical E17 final-only sequences. E18 one-pass prompts require the 128K
endpoint, as documented below.

`run_tinker.py` is the sole hosted adapter.  It uses rank 32, eight rollouts per
prompt, temperature 1, importance-sampling loss, and a conservative `1e-4`
learning rate. Rank 32 is the raw SDK default and the user-selected E18 capacity
candidate. The earlier rank 16 choice was not a service limit: it was inherited
from the cookbook's published GPT-OSS-120B SFT sweep and used as a conservative
first RL smoke. The cookbook has no calibrated GPT-OSS RL rank or learning-rate
sweep, so neither the SFT rank-16 result nor the SDK rank-32 default establishes
which rank is optimal for this RL dataset.

The 2026-08-10 cookbook `Config` forwards only `rank` to
`create_lora_training_client_async`; the raw SDK defaults train MLP, attention,
and unembedding.  `run_tinker.py` freezes these component flags within the
current process, now defaults E18 one-pass rows to `--lora-target all` and
`--lora-rank 32`, restores the SDK method after training, and writes
`backend_contract.json`. Historical E17 final-only rows retain their frozen
rank-16 attention-only default, so reproducing that lineage does not silently
change. The pinned SDK
signature independently confirms `rank=32`, `train_mlp=true`,
`train_attn=true`, and `train_unembed=true` as its defaults.

Capacity under the pinned cookbook counter is:

| rank / target | trainable parameters | relative to rank-16 attention-only |
|---|---:|---:|
| rank-16 attention-only | 11,943,936 | 1.0x |
| rank-32 attention-only | 23,887,872 | 2.0x |
| rank-16 all-module | 657,193,984 | 55.0x |
| rank-32 all-module | 1,314,387,968 | 110.0x |

Moving from attention-only to all modules is therefore the dominant capacity
change; doubling rank then doubles that larger adapter again. This may help the
model change evidence integration in MLP blocks, but it also increases optimizer
work and overfitting/instability risk on 1,674 Bio train molecules. Tinker bills
token units rather than trainable-parameter count, so the token-price formula is
unchanged, but wall time and queue behavior must be remeasured rather than copied
from the rank-16 attention-only smoke.

The no-hosted-call E18 config preflight is stored at:

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_visible_prefetched_v1/
  config_preflight/bio_rank32_all_128k_6144/
```

It passed against all 1,674 audited Bio train rows and records rank 32,
all-module flags, 1,314,387,968 trainable parameters, the 128K endpoint,
6,144 output tokens, 419 expected batches, and
`hosted_training_client_created=false` / `hosted_training_calls_made=false`.

## Prompt and generation parity

Use renderer `gpt_oss_medium_reasoning`, not the cookbook's first generic
recommendation `gpt_oss_no_sysprompt`.  The former exactly reproduces the
current valid Hugging Face Harmony surface, including the dated OpenAI system
header and `Reasoning: medium`; the checked BBB sample is 3,135 tokens in both
paths.  The model-visible message list itself is read unchanged from the
materialized trace.

Current GPT-OSS-120B `full_flat` scaffold-valid traces have these token
distributions:

| task | n | prompt mean / p95 / max | completion mean / p95 / max | max total |
|---|---:|---:|---:|---:|---:|
| BBB | 366 | 3037 / 3647 / 4216 | 662 / 787 / 870 | 4923 |
| Bioavailability | 209 | 2855 / 3615 / 4147 | 668 / 783 / 856 | 4832 |
| Skin | 245 | 2943 / 3671 / 4120 | 714 / 909 / 977 | 4980 |

The hosted adapter therefore preserves the valid runner's 20,480 generated
token cap and relies on the GPT-OSS renderer stop sequences (`return` and
`call`) for normal completion.  It must not shorten prompts or use the earlier
768-token local diagnostic cap.

## Current public pricing

For `openai/gpt-oss-120b` at 32K, current public rates per million tokens are
`$0.33` prefill, `$0.066` cached prefill, `$0.84` sampling, and `$0.737`
training.  The final receipt must use service billing events and actual token
counts; a forecast may show uncached and seven-of-eight-cached prefill bounds.

## Historical unmatched receipt: BBB rank-16 step 0

The interrupted Codex thread did not terminate the provider job.  It completed
successfully and wrote the local receipt to:

```text
outputs/paper/rl_lora_gpt_oss_120b/feasibility_20260810/tinker/bbb_step1_rank16/
```

The run used four unchanged BBB train prompts, eight generations per prompt,
the `gpt_oss_medium_reasoning` renderer, and the 20,480-token cap with natural
Harmony stops.  All 32 trajectories ended cleanly and produced parseable,
schema-complete JSON.  Two of four prompt groups had mixed rewards and were
retained for optimization; the other two constant-reward groups were sampled
but removed before training.  The retained metrics were 16 episodes, 46,624
observation tokens, 18,417 action tokens, 31.25% correctness, and mean reward
`-0.275`.  This satisfies the nonzero-advantage smoke gate, but the run used the
SDK's historical all-module default and is retained only as an unmatched
backend receipt.

Timing and token receipt:

| quantity | observed |
|---|---:|
| end-to-end warm step | 46.84 s |
| sampling | 32.63 s |
| optimizer train step | 8.25 s |
| final checkpoint save | 5.82 s |
| all sampled prompt tokens | 97,136 |
| unique uncached prompt tokens | 12,142 |
| sampled completion tokens | 33,111 |
| retained training tokens | 65,041 |

At four prompts per step, a one-pass sequential forecast is 734 BBB batches
(`9.55 h`), 419 Bioavailability batches (`5.45 h`), and 492 Skin batches
(`6.40 h`), or about `21.41 h` total before queue variation and final export.
Using the public token rates, the observed 50%-mixed-group step costs about
`$0.0854` in token charges.  Extrapolation is about `$62.64`, `$35.73`, and
`$41.96` by task (`$140.32` total).  If every group is trainable rather than
half, the conservative token-charge ceiling is about `$97.90`, `$55.84`, and
`$65.58` (`$219.31` total).  Checkpoint/storage charges are additional.  The
billing API returned no event yet because usage aggregation may lag by hours;
replace this forecast with the session-attributed billing row when available.

## Matched attention-only receipt and forecast

The matched rerun is stored at:

```text
outputs/paper/rl_lora_gpt_oss_120b/feasibility_20260810/tinker/bbb_step1_rank16_attention_only/
```

Its `backend_contract.json` records rank 16, attention enabled, MLP/unembedding
disabled, and 11,943,936 trainable parameters.  All 32 trajectories stopped
cleanly.  Two groups were constant and removed; two had mixed rewards and
entered optimization.  The retained training set contained 47,464 observation
tokens and 17,155 action tokens.  The attention-only optimizer step and final
checkpoint both completed successfully.

| quantity | matched observed |
|---|---:|
| four-prompt warm step | 27.58 s |
| sampling | 19.04 s |
| optimizer train step | 3.62 s |
| final checkpoint save | 4.80 s |
| all sampled prompt tokens | 97,136 |
| sampled completion tokens | 33,374 |
| retained training tokens | 64,619 |

Compared with the local full-prompt NeMo receipt (one prompt group in 117.04 s),
Tinker processes prompt groups about 16.97 times faster after normalizing the
four-versus-one batch shape.  Sampling token throughput is about 13 times
higher.  Local NeMo also failed its second-step refit OOM gate, so its projected
`213.76 h` one-pass runtime is not currently a stable executable forecast.

For one pass over every current train molecule, with eight rollouts per prompt:

| task | prompts | batches | matched time | observed-mix token estimate | all-mixed ceiling |
|---|---:|---:|---:|---:|---:|
| BBB | 2,935 | 734 | 5.62 h | $62.57 | $98.20 |
| Bioavailability | 1,674 | 419 | 3.21 h | $35.69 | $56.01 |
| Skin | 1,966 | 492 | 3.77 h | $41.91 | $65.78 |
| all three | 6,575 | 1,645 | 12.60 h | $140.17 | $219.99 |

The observed-mix estimate applies this smoke's 50% mixed-reward retention;
the ceiling assumes every sampled group is also trained.  Tinker bills by
tokens rather than trainable parameter count, so matching attention-only
targets materially improved wall time but did not materially change price.
The full pre-registered development plus frozen all-train refit is about 1.8
passes: `22.69 h` and approximately `$252.31-$395.99`, before small valid
evaluation, checkpoint, and storage charges.  Additional epochs scale almost
linearly.

The API credential used for this run was pasted into a Codex conversation.
Although its node-local copy is mode `0600`, it remains an exposed credential;
the user explicitly authorized continued use on 2026-08-10.  Never print it or
copy it into repository artifacts, manifests, commands, or reports.

Choose Tinker only if the measured end-to-end development run is materially
faster after including queue/export time, or if its expected all-task price is
worth the operational simplification.  A nominal accelerator claim without a
matched prompt/token receipt is not sufficient.

## E18 one-pass 128K receipt and cost gate

E18 fuses the single, full-flat analog, and final schemas and evidence into one
generation. Its current-valid renderer prompt distribution is substantially
longer than E17:

| task | n | prompt p50 / p95 / max | prompt >28,672 | prompt >32,000 |
|---|---:|---:|---:|---:|
| BBB | 366 | 16,168 / 28,120 / 35,259 | 16 | 4 |
| Bioavailability | 209 | 25,968 / 36,440 / 40,911 | 84 | 54 |
| Skin | 245 | 26,883 / 34,630 / 44,469 | 100 | 43 |

The 28,672 threshold reserves 4,096 output tokens under a 32K context. Thus
the formal E18 run cannot use the cheaper 32K model without changing the
evidence surface; it uses `openai/gpt-oss-120b:peft:131072`. Current public
128K rates per million tokens are `$0.78` uncached prefill, `$0.156` cached
prefill, `$1.94` sampling, and `$2.33` training.

The bounded 128K smoke is stored at:

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_v1/tinker_smoke/bbb_step1_rank16_attention_128k/
```

It used four BBB train prompts, eight trajectories per prompt, rank-16
attention-only LoRA, a 4,096-token cap, and the pre-freeze v1 hierarchical
one-pass reward. Formal v2 changes only the `+0.03` schema bonus by checking
nested single/analog keys as well as top-level keys. All four groups had mixed rewards and entered optimization. The step
completed successfully with 93.75% full-schema output, 50% final correctness,
28.125% single correctness, 50% analog correctness, and 28.125% correctly
resolved conflicts.

| quantity | observed |
|---|---:|
| end-to-end step | 107.83 s |
| sampling | 48.10 s |
| optimizer train step | 56.88 s |
| all sampled prompt tokens | 576,864 |
| unique prompt tokens | 72,108 |
| sampled completion tokens | 99,954 |
| retained training tokens | 676,818 |
| token-price estimate | $1.91 |

Using the measured step timing, one sequential all-train pass is approximately
`21.97 h` for BBB, `12.54 h` for Bioavailability, and `14.72 h` for Skin, or
`49.23 h` total. Using each task's current-valid mean prompt/completion length
as the train forecast, seven-of-eight cached prefills, and all groups retained,
the token estimate is `$1,309` BBB + `$997` Bioavailability + `$1,139` Skin =
`$3,445` total. Three full-valid checkpoint evaluations add about `$54` at
greedy one-sample decoding; checkpoint, storage, queue variation, and the final
test call are additional. This estimate is materially larger than E17 and is
a mandatory user cost gate before starting the full hosted run.

### Bio-only 6,144-token preflight and formal launch

The formal Bio-only contract supersedes the earlier BBB-derived 4,096-token
output assumption. In the frozen-base Bio valid outputs, 41/209 completions exceeded
4,096 tokens and the maximum was 6,143. The Bio contract therefore uses a
6,144-token generation cap for training and checkpoint evaluation. Exact
Tinker rendering of the sanitized all-train rows has prompt
min/p50/p90/p95/p99/max `2,426 / 27,731 / 35,114 / 36,498 / 38,442 / 41,959`;
no prompt plus 6,144 output tokens exceeds the 131,072 context.

The all-train and valid data gates are:

| split | rows | label 0 / 1 | data SHA-256 | tool ok / explicit error |
|---|---:|---:|---|---:|
| train | 1,674 | 461 / 1,213 | `6ba340b178675f542ad9ca0452556e78b97539c9dd1c67d31ed5f9fe7f2fd90f` | 24,445 / 241 |
| valid | 209 | 58 / 151 | `6570c753a54109fdc0fedcd7ed55d108b485526aed4ebfb40f2b76d6626c1d82` | 2,991 / 22 |

Both audits pass contiguous source-index and source-label alignment, global
`parent_disjoint`, prompt hash, anonymous identity, private gold, fixed tool
receipt, and no-live-tool gates. Failed tool calls retain an explicit `error`
status but expose only a generic missing-evidence message; backend exception
text and mmpdb structure fragments are not model-visible.

The paid Bio one-step preflight is stored at:

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_v1/tinker_preflight/bio_step1_rank16_attention_128k_6144/
```

It sampled all 32 requested trajectories. All 32 parsed as JSON and stopped
cleanly; 30/32 had the complete nested schema. Mean/max completion length was
`3,114.7 / 4,403`, below the 6,144 cap. Three groups had reward variance and
entered GRPO; one all-1.35 group was correctly removed as constant reward.
The retained optimizer batch contained 24 trajectories, 644,032 observation
tokens and 76,967 action tokens. Sampling plus the optimizer step and final
checkpoint export completed in 64.78 seconds. The final sampler checkpoint was
then read back through evaluator v2 on four valid rows; all four passed the
complete-schema gate on their first attempt.

The superseded formal Bio draft used rank-16 attention-only LoRA. After the
user's 2026-08-10 capacity decision, the formal candidate is rank-32 all-module,
with group size 8, four groups per batch, learning rate `1e-4`, 419 batches,
sampler exports at 140/280/final, and rolling resume state every 20 batches. Its
new log root is:

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_v1/tinker_train/bioavailability_ma_rank32_all_128k_6144/
```

The token-derived Bio estimate remains approximately `$0.8k-$1.0k` after
constant-group filtering; with three valid evaluations and one selected test
evaluation, `$1.1k` remains a token-charge ceiling. This is not yet a rank-32
all-module full-run wall-time receipt. The bounded matched smoke below removes
the mechanical capacity blocker. This text records the pre-launch gate; the
later hosted run was intentionally stopped and is now a frozen partial receipt.

## Visible-prefetched one-pass gate

The independent upper-bound contract is `one_pass_full_flat_visible_prefetched.v1`.
It shows query/neighbor structures and allowed source identifiers while keeping
retrieval and all three molecular tools harness-prefetched; there are no live
LLM tool calls and gold remains private. The full current valid data and result
receipt is in `VISIBLE_ONE_PASS_VALID_RESULTS.md`.

The bounded Bio smoke used 12 train rows (5 label-0 / 7 label-1), rank-16
attention-only LoRA, three batches, four prompts per batch, eight rollouts per
prompt, and a 6,144-token cap. It completed successfully and exported both
step-3 and final sampler checkpoints:

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_visible_prefetched_v1/
  tinker_smoke/bio_3step_rank16_attention_128k/
```

The local/W&B metrics contain three rows with reward, final/single/analog
correctness, `train/loss_sum`, entropy, and both sampled-vs-train KL variants.
`train/loss_sum` is the signed importance-sampling policy objective reported by
Tinker's forward/backward response, not cross-entropy and not expected to be
positive or monotonic. All-rollout mean reward was `0.7750 -> 1.0038 -> 0.7059`;
because each step uses different molecules, this smoke proves observability and
optimization, not learning improvement. The exported step-3 sampler then
generated three Bio valid rows; all three passed the complete-schema gate on
their first attempt.

W&B run (API-verified `finished`, three history rows):
<https://wandb.ai/reasonv/txagent-one-pass-rl/runs/jjwsc0d8>

This receipt remains valid for observability wiring only. It does not validate
the subsequently selected rank-32 all-module formal configuration.

### Matched rank-32 all-module smoke

The user-selected capacity configuration completed three hosted optimizer steps
over 12 audited Bio train rows (four prompts per step, eight rollouts per prompt):

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_visible_prefetched_v1/
  tinker_smoke/bio_3step_rank32_all_128k/
```

`backend_contract.json` records rank 32, MLP/attention/unembedding enabled, and
1,314,387,968 trainable parameters. All 96 sampled rollouts are retained in the
three `train_rollout_summaries.jsonl` files, including constant-reward groups.
Completion tokens were min/mean/p50/p90/p95/p99/max
`1,575/3,129.4/3,158/4,024/4,214/4,911/4,911`; zero equaled or exceeded the
6,144-token cap. Maximum prompt plus completion was 46,996 tokens, so no 128K
context overflow occurred. All 96 had a clean stop; 95 parsed as JSON and 92
had the complete nested schema. The four format failures were 2,051-4,050
tokens and cleanly stopped, proving they were format-adherence failures rather
than length truncations.

The three steps took `63.05/60.06/79.25` seconds. Signed importance-sampling
loss was `0.0025/538.2314/75.5882`, entropy `0.3859/0.3923/0.3647`, and KL-v2
`0.01145/0.01103/0.01031`; no capacity-related training or refit error occurred.
Step-3 and final state/sampler checkpoints were exported. The final sampler then
completed three frozen valid rows on the first attempt with full schema and
completion lengths `2,914/3,417/3,694`, again with zero cap hits. This is a
mechanical and observability pass, not evidence of learning improvement from
three non-matched molecule batches.

W&B run: <https://wandb.ai/reasonv/txagent-one-pass-rl/runs/n4np9pc8>

The machine-readable receipt is `smoke_audit.json`; curves are
`training_curves.png` and `training_curves.json`. Formal training and all test
rows remain untouched. The hosted 120B run must remain stopped unless the user
explicitly starts a new task for it.

The formal visible-prefetched Bio train set is also fully materialized and
audited: 1,674 rows (461/1,213 labels), 24,445 successful tool receipts, 241
generic missing-evidence receipts, zero gold/live-tool leak, and zero context
overflow. Its prompt total is 44,148,331 tokens; min/p50/p90/p95/p99/max are
`2,473 / 30,576 / 39,001 / 40,668 / 43,683 / 48,301`, with 6,144 output tokens
reserved under 128K.

Using the smoke's 2,983.9 mean completion tokens, 41.7% retained-rollout
fraction, seven-of-eight cached prefills, and current 128K rates, one complete
Bio train pass is about `$542`; the all-groups-retained ceiling is `$1,076`.
Three full-valid checkpoint evaluations plus one selected test evaluation add
about `$76` at the measured base-valid token rate. Thus the current campaign
forecast is approximately `$618-$1,152`, plus checkpoint storage and queue
variation. The three-step smoke averaged 60.6 seconds per four-prompt batch,
which extrapolates to about 7.0 hours for 419 train steps before queue and
checkpoint-evaluation time.
