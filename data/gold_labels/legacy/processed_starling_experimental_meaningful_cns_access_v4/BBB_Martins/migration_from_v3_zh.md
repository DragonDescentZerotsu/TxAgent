# BBB experimental metric-direction review：v3 → v4

## 审查合同

- 只审查 v3 中仅因缺少 `bbb_permeability_label` 而被拒绝的 7,226 条记录。
- 不对 heterogeneous numeric endpoints 使用统一阈值。
- 只有原记录自身给出无歧义 qualitative direction 才进入人工复核。
- 81 条规则候选全部做 source-index 级复核；16 条拒绝，65 条获准投票。
- 无法可靠确定方向的 7,145 条继续作为 non-voting retrieval evidence。

## Parent 重投票结果

- binary parents：3,666 → 3,675
- 新增/移除 parents：15/6
- shared-parent label flips：0
- shared-parent split changes：0
- 新增 voting records：65
- 受影响 parent vote groups：56

## Split 合同

所有 surviving v3 parents 保留原 split。新增 parent 若 scaffold 已存在则继承该 scaffold 的 split；全新 scaffold 只进入 train。最终 identity/scaffold overlap 均为 0。

详细 parent 变化见 `changed_gold_parents.jsonl`；所有新增 vote 对 parent 的影响见 `reviewed_vote_affected_parents.jsonl`。
