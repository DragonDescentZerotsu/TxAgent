# Ames existing-record family/level audit

本轮按用户修正后的范围，逐条阅读 **200 条已有 record 的全部非空语义字段**，审核其 family/level
归属及 L2 的直接实验召回。没有查询原始论文，没有验证 PMID、抽取真实性或重新查询化合物名称。
8 条投票候选只与已有冻结身份表、当前投票台账作本地联接。

**这是分层审核结果及改动建议，尚未应用到 source、votes、gold、split、catalog 或 index。**
当前活跃数据仍保留 v4 的构建 lineage；本轮没有暗中撤销此前已应用的 v4 决定，也没有应用已停止的
外部来源核查草稿。不能据此声称当前全库分层已经合格。

## 审核覆盖与结果

| 当前 level | 逐条阅读数 |
|---|---:|
| L1 | 24 |
| L2 | 71 |
| L3 | 24 |
| L4 | 24 |
| L5 | 57 |
| 合计 | 200 |

四个来源分别为 ames_base 71、ames_v1 24、ames_v2 36、ames_v3 69 条。
先按固定 seed、endpoint 分层及边界风险准备 800 条候选，再阅读其中覆盖全部 source/level 的 200 条：
144 条来自 endpoint 分层部分，56 条来自边界/召回定向部分。另 600 条没有被计作已审核，也没有自动生成判断。
本轮不是等概率随机抽样，**不能把建议改层比例外推为全库错误率**。

- 137 条保留当前层级。
- 49 条有明确改层建议，全部继续作为可检索证据；没有提出直接删除这些记录。
- 8 条为直接投票召回候选，尚未获得 L1 membership。
- 6 条保留边界问题：2 条 L1 的实体/条件范围，2 条 L3/L4 的 assay 含义，2 条尚不能明确对应现有 family。

| 改层建议 | 条数 |
|---|---:|
| L2 → L3 | 15 |
| L2 → L4 | 7 |
| L2 → L5 | 3 |
| L4 → L3 | 1 |
| L5 → L3 | 5 |
| L5 → L4 | 18 |

## 主要发现

**间接 family 不应由 Starling run 决定。** 当前 `source_contract.classify` 中，base/v1、v2、v3
分别进入 L3、L4、L5 的分支。已有记录显示这会造成同类读数错层：

- `ames_v3:492236`：32P-postlabelling 测 styrene-DNA guanine adduct，应 L5 → L4。
- `ames_v3:160729`：comet tail DNA 与损伤重接动力学，应 L5 → L4。
- `ames_v3:465422`：c-PTIO 对微核频率的逐 clone 实测结果，应 L5 → L3。
- `ames_v3:81309`：胚胎染色体计数/非整倍体，应 L5 → L3。
- 对照案例 `ames_v3:449198` 的 native category 虽含 `dna`，实际测 TBARS/protein carbonyls，仍应 L5。
- `ames_v2:282963` 的 chromosome fragmentation 来自 PFGE 对易断裂 DNA 位点的检测，仍应 L4；
  不能见到 chromosome 一词就升到固定染色体改变的 L3。

**L2 防泄漏需要识别含义，不能只匹配字符串。** 本轮发现：

- `ames_v2:177291` 与 `ames_v3:505875` 的 Ames 是被引用作者，应按 DNA 损伤/加合物归 L4。
- `ames_base:305476` 明说未报告 Ames 结果，实际是 ROS/apoptosis，应 L2 → L5。
- `ames_v1:61278` 是 C. elegans 表型回复，`ames_v1:45652` 是 V79/HPRT 重组回复，应 L2 → L3。
- `ames_v2:117524`、`ames_v3:313659` 的 Salmonella umu/SOS reporter 测 DNA damage response，
  `ames_v3:15634` 测 repair-deficient/proficient 差异存活，都应 L2 → L4。
- 相反，`ames_v1:38440`、`ames_v2:9168`、`ames_v3:187404` 的全文记录确实夹带 Ames 结果，保留 L2。
  原始 run 的“该结果不在本 run 抽取范围”说明不能解除完整记录的 direct 限制。

**预测按预测对象分 family。** `ames_v3:485094` 预测的是 DNA adduct 数量，应 L4；
`ames_v3:52220` 比较细菌生长抑制的 CA/IA prediction，保持 L5。
`ames_v3:81309` 中 predicted 修饰排卵时间，并不表示染色体终点来自模型。
本轮没有因 prediction、needs_more_context、引用结果或 modifier/protectant 角色删除间接证据。

**L2 中确有可进一步回收的实测 direct 记录。** 8 条候选中，5 条已经通过现有冻结名称-parent 匹配，
且当前没有同 PMID-parent 的票：`ames_base:55173`、`:134959`、`:198214`、`:240657`、`:38208`。
它们给出了阳性对照的自身实测回复结果。另 3 条需要既有身份或完整 panel 条件的处理。
与之相对，`ames_base:254386`、`:205469` 只指定对照/判阳标准而没有实测结果，保持 L2。
候选的语义认可不等于投票已经接受；必须经完整身份、PMID-parent-condition 去重/冲突处理与 gold builder，
由实际输出的 vote 赋予 L1，不能凭本审核直接把 `group_id` 改成 L1。

## 审核文件与验证

- `sample.jsonl`：200 条原始完整 payload 及审核时的 canonical membership。
- `authored_judgments.tsv`、`audit_annotations.jsonl`：200 条逐条撰写的判定与理由；后者绑定 raw payload hash。
- `placement_changes.jsonl`：49 条改层建议，`applied=false`。
- `vote_recall_candidates.jsonl`：8 条投票召回候选，附冻结身份表/当前票的本地检查结果。
- `summary.json`：覆盖、计数、输入 hash、验证与限制。
- `reviewed_numbers.json`、`prior_reviewed_source_ids.json`、`sample_selection.py`：固定抽样及阅读子集的复现依据。

200 条 source ID 均按 **零起始 Parquet 行序号**回连并验证完整 payload；当前 canonical 行的分组、
endpoint、context、support、identity 等字段均一致，L1 与 actual-voter 台账一致。
这些检查仅保证审核的记录与当前数据对应，不是对文献真实性的核验。
source manifest 与 source votes 相比本轮开始均未改变。抽样重放输出到独立目录：

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python \
  data/starling_data/ames/source_review_v5/sample_selection.py \
  --output /local/tmp/ames-placement-replay
```

正式修复需要同步生产分层规则与独立验证器，再重建受影响的 catalog/indices；两者目前都把宽泛
Ames/Salmonella/revertant 字符串当作 direct，因此不能只手改分组后放行验证。
