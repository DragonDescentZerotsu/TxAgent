# E18 frozen-base one-pass valid result

> Provenance note (2026-08-10): the metrics below use the original fused-valid
> Bio prompt hash `6c49b8...`. Pre-training audit later found that explicit
> backend error receipts could expose internal mmpdb fragments. The current
> sanitized Bio valid hash is `6570c7...`; 206/209 normalized prompts are
> otherwise identical, and the other three differ only by explicit empty-group
> metadata. The frozen-base Bio result must be refreshed on the sanitized valid
> rows before the final base-versus-RL table; checkpoint selection already uses
> only the sanitized valid rows.

Status: complete on 2026-08-10. All 820 current scaffold-valid rows completed
with zero failed outputs. Test was not read or run.

## Final prediction comparison

The aligned reference is each task's current GPT-OSS-120B three-stage
`starling_full_flat` result. The candidate changes only the LLM stage topology:
the same anonymous query properties, full-flat analog cards, pairwise tools,
and task profile are emitted in one response.

| task | n | three-stage macro-F1 | one-pass macro-F1 | paired delta | bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 366 | 0.6690 | 0.6751 | +0.0061 | [-0.0415, +0.0516] | 0.1421 |
| Bioavailability | 209 | 0.7004 | 0.6611 | -0.0394 | [-0.1026, +0.0247] | 0.0385 |
| Skin | 245 | 0.5698 | 0.5685 | -0.0013 | [-0.0605, +0.0581] | 0.8854 |

BBB and Skin preserve the three-stage result within paired uncertainty.
Bioavailability loses 0.0394 macro-F1 and 0.0670 accuracy; its asymmetric
paired correctness count is 27 three-stage-only versus 13 one-pass-only.
Consequently E18 RL must first recover the frozen one-pass architecture loss
on Bioavailability before it can claim an improvement over the current method.

## Explicit branch behavior

| task | single accuracy / macro-F1 | analog accuracy / macro-F1 | conflicts | final accuracy on conflicts |
|---|---:|---:|---:|---:|
| BBB | 0.6721 / 0.6539 | 0.7158 / 0.6621 | 100 | 0.6200 |
| Bioavailability | 0.5072 / 0.4994 | 0.6794 / 0.6506 | 70 | 0.7714 |
| Skin | 0.6082 / 0.5918 | 0.5878 / 0.5555 | 55 | 0.5636 |

These fields are model outputs, not post-hoc labels. They are parsed directly
as `single_prediction`, `analog_prediction`, and the task final prediction.

## Tool and label-privacy audit

The one-pass model cannot issue live tool calls. Retrieval and the fixed
`molecule_properties`, `mmp_structure_compare`, and `properties_compare`
bundle are completed by the harness before generation. The three valid audits
found 10,650 explicit tool receipts: 10,597 `ok` and 53 deterministic `error`
receipts already present in the aligned three-stage inputs. Explicit error text
is preserved as evidence of tool failure; it is not treated as a missing call
or silently repaired. All prompts are identity blind, contain no source
assistant answer or gold label, and have no live tool surface.

## Inference efficiency

| task | one-pass calls | three-stage calls | one-pass tokens | three-stage tokens | token reduction |
|---|---:|---:|---:|---:|---:|
| BBB | 368 | 1,090 | 7,166,756 | 8,696,453 | 17.59% |
| Bioavailability | 209 | 624 | 5,580,673 | 6,431,717 | 13.23% |
| Skin | 248 | 725 | 6,427,263 | 7,140,909 | 9.99% |
| total | 825 | 2,439 | 19,174,692 | 22,269,079 | 13.90% |

Five one-pass rows needed one schema retry; every other row succeeded on the
first attempt. The topology reduces the normal model-call count from three to
one and removes the group-to-final dependency barrier. Under a saturated
token-throughput endpoint, the measured token reduction is the more
conservative speed proxy; under latency-limited serving, the one-versus-three
call topology provides additional benefit.

## Artifacts

```text
outputs/paper/rl_lora_gpt_oss_120b/one_pass_full_flat_v1/base_valid/
  bbb_martins/{metrics.json,predictions.jsonl,runs/}
  bioavailability_ma/{metrics.json,predictions.jsonl,runs/}
  skin_reaction/{metrics.json,predictions.jsonl,runs/}
```

The full hosted all-train run remains gated because the one-pass context raises
its forecast to approximately $3.45k plus evaluation, checkpoint, storage, and
test costs.
