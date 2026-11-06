# TxGemma-style supervised label prediction

`python -m predict.baselines.tinker_sft {prepare,run,verify}` is the shared implementation.
The concrete 12-task paper experiment is `experiments/baselines/tinker_sft.sh`.
This trains **DeepSeek**, not TxGemma, and is not a reproduction of the published
TxGemma checkpoint or its full 66-task training mixture.
Both six-task suites completed on 2026-11-01; the maintained score table and
completion record are in [results](../../docs/results.md#new-supervised-label-baseline).

## Scientific contract

- Six tasks, independently trained on each of the TDC and conditioned benchmarks.
  The user selected the **frozen paper ML/naive baseline splits** on 2026-10-31.
  `experiments/baselines/tinker_sft_paper_sources.json` pins all 36 source files.
  Skin/Carcinogens TDC use the corrected seed-42 release. These inputs must not
  be silently replaced by Joseph's default TDC manifest, which differs.
- Exact Tinker model: `deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144`.
  Native `deepseekv4_1_disable_thinking` renderer; no reasoning targets or
  free-form answer parsing. `A` means Y=0, `B` means Y=1; the tokenizer must
  represent each with exactly one token. Loss is full-vocabulary cross-entropy
  on that answer token only, with zero weights on the prompt. No EOS loss.
- In each epoch, 70% of training rows have no examples; 30% (rounded to the
  nearest row) have 1–10 uniformly sampled training examples. Randomization is
  deterministic for seed, task and epoch and shared across hyperparameters.
  Evaluation uses the top 10 training rows by Morgan radius-2/2048-bit Tanimoto.
  Ties follow source order. Exclude all examples sharing the query's normalized
  parent connectivity; retain source split rows unchanged. Conditioned examples
  include their own reported conditions. No retrieval evidence or priors enter
  this baseline. The nearest-neighbor pool is not condition-filtered.
- Fixed LoRA rank 32, attention and MLP adaptation, no unembedding adaptation,
  seed 42, batch size 32. Two learning rates (1e-4, 5e-4), five epochs each,
  constant LR and Tinker Adam defaults. Each task chooses LR/checkpoint by
  validation binary cross-entropy, then its threshold by validation Macro-F1
  using the existing MiniMol metric helpers. No train+valid refit, test selection,
  or cross-task training. This is a single seed, not an ensemble.
- At evaluation, score A and B with teacher-forced candidate log probabilities,
  then normalize the pair to obtain P(Y=1). Report AUROC, Macro-F1, accuracy
  and the selected threshold; also retain fixed-0.5 metrics. The SDK's
  `compute_logprobs` generates one discarded token per request; it does not
  generate a reasoning trace.

The [TxGemma report](https://storage.googleapis.com/research-media/txgemma/txgemma-report.pdf)
describes 70/30 zero/few-shot training and nearest-neighbor evaluation. Its
published evaluation also uses validation examples in the example pool; ours
uses **train only**, as requested. Backbone, LoRA, independent task training,
completion-only loss, label spelling and validation selection also differ.
Do not label this result “TxGemma” in a comparison table.

Frozen TDC splits may contain historical parent overlap after normalization;
preparation records parent-connectivity overlap rather than altering cohorts.
Conditioned parent overlap is rejected. Historical Ames/DILI validation/test
renaming and prior examination are not undone: these are held-out from this
training run, not previously unseen study cohorts.

## Run and resume

Use an isolated Python environment with RDKit, NumPy, scikit-learn, PyTorch,
python-dotenv and the project dependencies. The run was prepared using
`tinker==0.36.0`, `tinker-cookbook==0.5.9`; preparation records installed versions.
Do not change the shared vllm environment to install training clients.

For an existing project environment, create a small overlay using its Python:

```bash
python -m venv --system-site-packages /path/to/tinker-runtime
/path/to/tinker-runtime/bin/python -m pip install \
  'tinker==0.36.0' 'tinker-cookbook==0.5.9'
```

The native SDK supplies training and candidate scoring; the OpenRouter inference
transport is not involved. Run the focused offline checks with
`python -m pytest tests/predict/test_tinker_sft.py -q`.

```bash
export PYTHON=/data1/tianang/cache/txagent/tinker_sft/runtime/bin/python
export PAPER_SOURCE_ROOT=/data1/tianang/Projects/TxAgent
export TINKER_ENV_FILE=/data1/tianang/Projects/TxAgent/.env
bash experiments/baselines/tinker_sft.sh
```

`prepare` makes no Tinker calls. It checks input hashes, IDs and binary labels,
renders all prompts without truncation, saves token IDs and example UIDs, and
estimates the complete sweep cost. Credentials are read at runtime from the
specified existing file (or `TINKER_API_KEY`), never copied into the checkout.
An explicitly supplied credential file must contain the key; it cannot silently
fall back to another account. This renderer and historical price estimate support
only the pinned model; the CLI rejects model substitutions.

`run` verifies prepared hashes and exact model support. A process lock prevents
duplicate launches. The recipe runs four tasks concurrently; up to 32 query pairs
are scored concurrently. Within each task, forward/backward and optimizer pairs
are pre-submitted in order in blocks of at most 25 batches. Tinker's sequence IDs
preserve update order; all submitted operations finish before saving the cursor.
Optimizer checkpoints are saved every 25 batches and at
epoch boundaries. Resume restores optimizer state; interrupted unsaved batches
may be recomputed and billed again. Completed validation predictions are reused.
`selected.json` is written before test requests; its checkpoint and threshold
are then frozen. Test predictions are resumable per query. Cached predictions
must have unique expected IDs, matching labels/conditions and probabilities
consistent with their finite A/B log probabilities before missing rows are scored.

For an existing completed run, use the same command owner for an offline audit:

```bash
"$PYTHON" -m predict.baselines.tinker_sft verify \
  --output outputs/predictions/tinker_sft/txgemma_style_label_v1/deepseek_v41_flash_1017_paper_20261031
```

`verify` checks prepared hashes, all validation predictions/BCE/thresholds,
checkpoint selection, test predictions/metrics and the aggregate `results.json`.
It requires neither credentials nor the Tinker service and does not change
scientific outputs. `run` on a root containing `results.json` takes this same
offline path, preserving completion timestamps and provider metadata. To create
a new experiment, choose a new `RUN_NAME` or `OUTPUT_ROOT`; do not reuse a frozen run.

Artifacts under the run root:

- `plan.json`: protocol, source hashes, row counts/overlap, package versions,
  token counts, cost estimate and prepared-file hashes.
- `<benchmark>/<task>/train_epoch_*.jsonl`, `valid.jsonl`, `test.jsonl`: ordered
  tokens, labels, row IDs, conditions and training-example IDs.
- `lr_*/`: optimizer resume pointer, training events, validation predictions,
  checkpoint locators and validation metrics per epoch.
- `selected.json`, `test_predictions.jsonl`, `test_metrics.json`; after all
  tasks finish, the root `results.json` contains all 12 results.

Keep datasets, checkpoints and prediction logs in this single run root; keep the
isolated runtime outside the checkout. `source_snapshot.zip` and the implementation
receipt bind the original experiment code (`8f0fcad6`), while later runner fixes
are verified against those saved outputs. `execution_history/initial_serial_requests/`
preserves the intentionally interrupted first attempt, including its cancellation
trace; it is provenance, not a second active run. Billing snapshots dated before
completion are partial usage estimates, not final charges or remaining balance.

Intermediate checkpoint TTL is 30 days; selection removes the chosen sampling
checkpoint's expiration so the final adapter remains available. Long-term offline
export is still needed to survive provider/model retirement. Token estimates use the
[2026-10-31 Tinker rates](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/):
$4.98/M training, $0.60/M uncached prefill, $3.60/M sampling. They exclude
retries and storage ($0.10/GB-month), and do not assume cache discounts.
The $1,000 launch guard is an estimate check, not a provider-enforced spending
cap or proof of remaining credits. Grant emails do not establish live balance.
