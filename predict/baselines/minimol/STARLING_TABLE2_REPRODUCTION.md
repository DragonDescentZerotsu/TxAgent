# Starling Table 2 MiniMol 复现

状态：2026-08-12 在 node001 A100 GPU 上完成；同日根据数据作者补充说明完成
`n_extractions` 加权复现。

这是针对论文 *Self-Driving Datasets: From 20 Million Papers to Nuanced
Biomedical Knowledge at Scale*（arXiv `2605.07022v3`）released CSV snapshot 的
复现，不是作者未公开的 Table 2 训练和预处理代码的 exact rerun。

## 协议

- 使用冻结的 MiniMol v1 512 维 embedding 和 upstream task-specific MLP head。
- 比较 TDC-only 与 TDC+literature train，在 released TDC test 与 literature test 上评估。
- 每个条件 5 次 repetition，每次 5-member ensemble；25 epochs；seeded TDC-style
  87.5/12.5 scaffold train/validation split；按 validation loss 选 checkpoint。
- 分类训练直接把 released fractional label 传给数值稳定的
  `BCEWithLogitsLoss(reduction="none")`；Oral/BBB evaluation 仍分别用
  `label >= 0.20` 和 `label >= 0.50` 的 hard target。AUROC 默认按 molecule 等权；
  `--evaluation-extraction-weight` 可显式运行 extraction-count-weighted sensitivity。
  LD50 保持连续回归。
- 数据作者于 2026-08-12 澄清：fractional label 是正 extraction 的票数比例；collapse
  到 molecule row 后，应按 `n_extractions` 上调 loss weight，并建议以
  `sqrt(n_extractions)` 作为温和权重。TDC row 没有该字段，固定 weight=1。
- 训练 minibatch 和整个 inner-validation split 都按 loss weight sum 归一化。与 weighted
  run 比较的 no-weight 对照使用同一份修正后的 global validation-loss 实现、相同 seed 和 split，
  避免把权重效应与 checkpoint-selection 修正混在一起。

若一个 molecule 有 `k` 个 positive 和 `n-k` 个 negative extraction，则 fractional target
`k/n` 配合线性 weight `n`，在 BCE 目标上等价于展开全部 `n` 个 extraction；`sqrt(n)` 是
作者建议的保守折中，避免极少数高计数 molecule 主导训练。

## 作者信息驱动的 matched 结果

下表均为 5 次 repetition 的 AUROC 均值。`none` 和 `sqrt(n)` 除 extraction weight 外完全
matched；括号内是论文 Table 2 数值。

| Task | 权重 | TDC Test | TDC Test Aug. | Lit. Test | Lit. Test Aug. |
|---|---|---:|---:|---:|---:|
| Oral Bioavailability | none | 0.689 (0.692) | 0.755 (0.779) | 0.636 (0.768) | 0.716 (0.821) |
| Oral Bioavailability | sqrt(n) | 0.689 (0.692) | 0.745 (0.779) | 0.636 (0.768) | 0.720 (0.821) |
| BBB | none | 0.939 (0.916) | 0.896 (0.909) | 0.644 (0.768) | 0.787 (0.890) |
| BBB | sqrt(n) | 0.939 (0.916) | 0.895 (0.909) | 0.644 (0.768) | 0.781 (0.890) |

因为 TDC-only train row 都没有 `n_extractions`，其 weight 恒为 1；同一 task 两种 policy 的
TDC-only 结果相同。权重只改变 augmented head：

- Oral Lit.-Aug. `+0.0036`，但 TDC-Aug. `-0.0098`。
- BBB Lit.-Aug. `-0.0067`，TDC-Aug. `-0.0015`。
- 对 averaged ensemble predictions 做 paired molecule bootstrap，Oral Lit.-Aug. delta 为
  `+0.0053`，95% CI `[-0.0020, +0.0127]`；BBB 为 `-0.0064`，95% CI
  `[-0.0094, -0.0033]`。

Oral 的线性 `n` sensitivity run 将 Lit.-Aug. 提到 `0.7221`，但 TDC-Aug. 降到
`0.7275`。更强权重没有解决差距，且跨 evaluation surface 的代价更明显，因此不再搜索任意
幂次，以免变成 test-informed tuning。

结论：作者补充说明修复了真实缺失的训练合同，但训练阶段的 `n_extractions` weighting 不是
Table 2 复现差距的主要来源。完整 released literature test 上，Oral/BBB 仅训练加权的
Lit.-Aug. 仍比论文低约 `0.10/0.11` AUROC。

## 测试集 `sqrt(n_extractions)` sensitivity

应用户要求，另做了训练 loss 与 literature-test AUROC 都使用 `sqrt(n_extractions)` 的完整
5-repetition rerun。测试权重不参与 checkpoint selection，也不改变模型 prediction；它只在每次
repetition 的 `roc_auc_score` 中作为 `sample_weight`，之后再对 5 个 AUROC 求均值和标准差。
TDC test 缺少 extraction count，仍全部为 weight 1。

| Task | Test weight | Lit. Test | Lit. Test Aug. |
|---|---|---:|---:|
| Oral Bioavailability | none | 0.6361 | 0.7200 |
| Oral Bioavailability | sqrt(n) | 0.6539 +/- 0.0028 | 0.7373 +/- 0.0074 |
| BBB | none | 0.6442 | 0.7805 |
| BBB | sqrt(n) | 0.6217 +/- 0.0024 | 0.7719 +/- 0.0061 |

测试加权使 Oral 两列分别提高 `+0.0178/+0.0173`，但使 BBB 分别下降
`-0.0225/-0.0087`。即使采用训练、测试双重 `sqrt(n)`，Oral/BBB Lit.-Aug. 仍分别比
论文的 `0.821/0.890` 低 `0.084/0.118`。因此测试加权也不能解释 Table 2 差距；该实验保留为
sensitivity，不替换默认 molecule-level AUROC。

LD50 不是 BCE 分类任务，该说明不适用；此前完整结果仍为 TDC/TDC-Aug./Lit./Lit.-Aug. MAE
`0.623/0.622/0.588/0.530`，论文为 `0.602/0.575/0.650/0.641`。

## 数据和剩余差距

fractional classification label 是有意保留的 molecule-level extraction aggregate，不是坏掉的
binary label。released literature test 同时含 aggregate `label`、`n_extractions` 和 thresholded
`label_hard`。

release 已通过 exact-SMILES uniqueness 与 train/test-overlap 检查。BBB literature train 中一条
`C[CH3+]OC(=O)c1ccc(N)cc1` 无法通过当前 RDKit sanitized parsing，因此不能进入 MiniMol；
modeled augmented pool 从 28,349 变为 28,348。两个 BBB test surface 均不受影响，raw CSV
及其 hash 未改动，排除记录完整保存在 `metrics.json`。

三个诊断进一步缩小了剩余原因范围：

1. 将 Oral literature train fractional label hard-threshold 后，旧口径 Lit.-Aug. 从 `0.715`
   提高到 `0.733`，仍显著低于论文 `0.821`。
2. 在 full released Oral literature test 上，仅训练使用 matched `sqrt(n)` 只把 Lit.-Aug.
   从 `0.716` 提高到 `0.720`；训练和测试都用 `sqrt(n)` 后为 `0.737`。
3. 在 post-hoc 的 49 个 `n_extractions >= 9` Oral test molecules 上，matched no-weight
   averaged prediction 的 molecule-equal AUROC 为 `0.821`，`sqrt(n)` 训练模型为 `0.848`；
   这里的 `sqrt(n)` 指训练权重，不是对子集再次做测试加权。49 个分子合计 1,176 条
   extraction，占 full test 2,545 条的 46.2%，但只占 49/839=5.84% 的 molecule rows；
   在 full test 使用 `sqrt(n)` 时，它们仅占 18.2% 的测试权重。因此加权 full test 与删除
   其余 790 个分子不是同一操作。即便改用线性 `n`，`sqrt(n)` 训练模型的 full-test
   averaged-prediction AUROC 也只有 `0.739`，仍远低于过滤后的 `0.848`。该过滤子集只有
   31 positive / 18 negative，只作 post-hoc sensitivity diagnosis，不能替代 headline result。

现有证据更支持以下未公开差异之一：论文 Table 2 对 literature test 使用了 extraction-count /
confidence filter，或论文表格与 released CSV 的 split/version 不一致。要做 exact reproduction，仍需
作者确认 Table 2 AUROC 是否 sample-weighted、精确 weight transform、minimum extraction count，以及
使用的 test snapshot/version。

## 入口与 artifacts

作者信息直接支持的训练加权命令：

```bash
/data1/tianang/anaconda3/condabin/conda run -n intern \
  python -m predict.baselines.minimol.run_starling_table2 \
  --task oral_bioavailability --device cuda \
  --extraction-weight sqrt
```

测试加权不是作者澄清中明确给出的 Table 2 合同；它作为独立 sensitivity 显式追加
`--evaluation-extraction-weight sqrt`，不得与上述训练权重混写成作者确认的默认协议。

Task choices 为 `oral_bioavailability`、`ld50` 和 `bbb`；非 `none` extraction weight 只允许
用于 BCE classification task。长运行可通过 `--repetition-indices` 分片，并用
`predict.baselines.minimol.merge_starling_table2_shards` 严格校验、合并 repetitions 1–5。

```text
matched no-weight:
  outputs/baselines/minimol_starling_table2_v3_global_validation/<task>/paper_soft/

sqrt(n) author-informed:
  outputs/baselines/minimol_starling_table2_v3/<task>/paper_soft_sqrt_extraction_weight/

sqrt(n) train + test sensitivity:
  outputs/baselines/minimol_starling_table2_v3/<task>/paper_soft_sqrt_extraction_weight_sqrt_evaluation_weight/

Oral linear-n sensitivity:
  outputs/baselines/minimol_starling_table2_v3/oral_bioavailability/paper_soft_linear_extraction_weight/
```

旧 batch-mean no-weight 结果仍保留在 `<task>/paper_soft/` 作为第一轮 historical receipt，不能拿来
估算纯 extraction-weight delta。完整 MiniMol 入口索引见 `predict/baselines/minimol/README.md`；immutable
data/label contract 见 `data/legacy/artifacts/starling_table2_v3/README.md`。

## 验证 receipt

- 12 个 local raw CSV 的 SHA-256 全部匹配 `source_manifest.json`。
- 每个正式条件均完整覆盖 repetitions 1–5 和 25 个 member receipt。
- 每个 prediction artifact 均有且仅有一条 finite score 对应每个 modeled test molecule。
- `metrics.json` 分别记录 train/evaluation extraction-weight policy 及
  minimum/median/maximum/sum。
- Ruff 通过全部新增或重构的 MiniMol modules。
- `pytest tests/baselines/test_minimol_run.py tests/baselines/test_minimol_starling_table2.py`：
  16 passed。
- `git diff --check`：passed。
