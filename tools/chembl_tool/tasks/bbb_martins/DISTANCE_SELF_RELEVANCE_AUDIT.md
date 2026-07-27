# BBB distance expansion：same-molecule causal continuity audit

日期：2026-07-21
状态：citation-backed internal audit；不是 external expert annotation

## 审计问题

结构上正确的 biological graph 不一定能预测 **同一个 query molecule** 的 BBB label。每条 H1/H2 路径还必须回答：

> assay molecule A 引起的 downstream change，是否能影响 A 自己的 BBB disposition？还是文献只证明它会影响另一个 probe/substrate B？

若路径需要额外命题（例如 “A 本身是 P-gp substrate” 或 “A 本身是 GLUT1 substrate”），而当前 retrieval row
没有提供该证据，则属于 `requires_query_role`。这不是 scope、quality 或 hop 数问题，而是 causal subject 在路径中
从 A 切换到了未观测的 B。

公共机器可读 contract 位于：

```text
tools/chembl_tool/common/evidence_distance.py
  FamilySelfRelevanceAudit
  validate_self_relevance_audit(..., require_publishable=True)

tools/chembl_tool/tasks/bbb_martins/distance_self_relevance.py
  SELF_RELEVANCE_AUDIT
```

## v3 family-by-family 结论

| family | 当前层级 | 结论 | same-molecule continuity |
|---|---:|---|---|
| `mmp9_activity` | Passive H1 | `pass_same_molecule`，但 scope 条件很强 | assay molecule 作为 MMP-9 perturbagen 改变它自己也要穿过的物理 barrier；没有自动换成另一个 substrate。需要 injury/inflammation、target exposure、时间尺度和 junction-sensitive route。 |
| `mmp3_activity` | Passive H1 | `pass_same_molecule`，但 scope 条件很强 | 与 MMP-9 相同；直接 barrier/junction perturbation 没有必需的 transporter-substrate 身份切换。 |
| `nrf2_activation` | Efflux H1 | `requires_query_role`，主实验不通过 | 文献中的 NRF2 activator 提高 ABC transporter；降低脑积累的是另一个已知 probe substrate。必须另证 query 自身是 ABCB1/ABCG2/ABCC2 substrate。 |
| `keap1_nrf2_interaction` | Efflux H2 | `requires_query_role`，主实验不通过 | KEAP1→NRF2 edge 正确，但继承 NRF2→efflux 的 substrate 断点；PPI assay 不提供 query substratehood。 |
| `hif1_activation` | Influx H1 | `requires_query_role`，主实验不通过 | HIF-1 提高的是 GLUT1 abundance/glucose uptake；不能推出任意 HIF perturbagen 自身是 GLUT1 substrate。 |
| `phd2_activity` | Influx H2 | `requires_query_role`，主实验不通过 | PHD2→HIF-1 edge 正确，但不能修复 HIF-1→GLUT1 transport 的 query-identity 断点。 |

支持文献：

- MMP-9 与 BBB junction disruption：[PMID 28523565](https://pubmed.ncbi.nlm.nih.gov/28523565/)、[PMID 21803344](https://pubmed.ncbi.nlm.nih.gov/21803344/)
- MMP-3 与 barrier/junction disruption：[PMID 33859779](https://pubmed.ncbi.nlm.nih.gov/33859779/)、[PMID 16624562](https://pubmed.ncbi.nlm.nih.gov/16624562/)
- NRF2 study 使用 sulforaphane 做 inducer，但用 verapamil/fluorescent probe 证明 transporter function：[PMID 24948812](https://pubmed.ncbi.nlm.nih.gov/24948812/)
- HIF-1 study 证明 endothelial GLUT1 和 glucose uptake，不是任意 HIF perturbagen 的 influx：[PMID 23047702](https://pubmed.ncbi.nlm.nih.gov/23047702/)

## AhR candidate 的处理

AhR 不在 v3，但本轮曾作为 Efflux H1 candidate。它同样归为 `requires_query_role`，不能进入主 H1/H2 graph。
TCDD/AhR activation 提高 transporter 后，原研究使用 verapamil 作为独立 P-gp substrate 验证脑积累下降；没有证明
TCDD 或任意 AhR activator 自身是 P-gp substrate：[PMID 21048045](https://pubmed.ncbi.nlm.nih.gov/21048045/)。此外，
在人 hCMEC/D3 cerebral endothelial cells 中，TCDD/AhR 增加 CYP1B1，但没有增加 ABCB1/ABCG2，进一步说明该
环境调节关系具有 model/species dependence：[PMID 25858487](https://pubmed.ncbi.nlm.nih.gov/25858487/)。

因此 AhR 只能作为独立 `context/exposure modifier` candidate，不能作为预测同一 molecule BBB label 的普通 H1。

## 对冻结 v3 产物的影响

- 已生成的 v3 manifest、index、retrieval replay 和 coverage 数字保留，作为结构/retrieval engineering artifact。
- 旧 D/C index、旧 paper matrix 和已有实验结果完全不改。
- v3 **不能直接进入正式 E12 LLM performance run**：Efflux H1/H2 与 Influx H1/H2 未通过
  `require_publishable=True` self-relevance gate。
- 不能用 prompt 提醒或让 LLM 自行猜测 substratehood 来修补断点；缺失的是 evidence contract 中没有的事实。
- 若未来能为同一 query/neighbor molecule 联合提供经过验证的 transporter-substrate evidence，才可以把
  `requires_query_role` family 重新审计为可发布。单独的 NRF2/HIF/PHD/KEAP assay 不够。

## Role-gated same-parent feasibility audit（2026-07-22）

已对上述补救条件做 ChEMBL-only 实测。可复现入口：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.audit_role_gated_overlap
```

严格 role evidence 要求 activity-row molecule 自身有阳性 transporter-substrate 结果：Efflux 只接受
ABCB1/ABCG2/ABCC2 的 compound efflux ratio >=2 或明确阳性 drug transport；Influx 只接受明确阳性的
GLUT1/SLC2A1 substrate uptake。single-direction Papp、低 efflux ratio、inactive、inhibitor/binding、ATPase、
probe accumulation 和 glucose/probe uptake inhibition 均不算 query-role evidence。identity 使用
`rdkit_fragment_parent.v1`；coverage 沿用 operational identity、Morgan similarity >=0.30、每 family top-3，
输入为 392 个正式 BBB test molecules。

| family | H parents | 同-parent阳性 role overlap | overlap fraction | 可用 H assays | 可用 H molecules | test >=1 coverage | test top-3 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nrf2_activation` | 12,522 | 27 | 0.216% | 12 | 27 | 23/392 = 5.87% | 2/392 = 0.51% |
| `keap1_nrf2_interaction` | 896 | 6 | 0.670% | 5 | 6 | 0/392 = 0% | 0/392 = 0% |
| `hif1_activation` | 3,031 | 0 | 0% | 0 | 0 | 0/392 = 0% | 0/392 = 0% |
| `phd2_activity` | 2,126 | 0 | 0% | 0 | 0 | 0/392 = 0% | 0/392 = 0% |

结论：同-parent join 在数据上可以实现，但不足以挽救当前 Efflux/Influx tree。NRF2 只剩 5.87% 的 query 有
任一 neighbor，KEAP1/HIF-1/PHD2 均为 0；不能形成稳定的 H1/H2 branch。它们继续保持
`requires_query_role`，不得进入正式 E12 LLM run。完整产物位于：

```text
outputs/chembl_tool/tasks/bbb_martins/distance_expansion/analysis/v3/role_gated_overlap/
  summary.json
  overlap_parents.tsv
  query_coverage.tsv
  report_zh.md
```

进一步检查“调控 assay 与 transporter-role assay 来自同一 ChEMBL document”的最强例外：

- NRF2：27 个 overlap parents 中 5 个来自同一 document 的 NRF2 nuclear-translocation 与 MDR1 efflux-ratio
  paired panel（CHEMBL4610043；PMID 31898999）；
- KEAP1：6 个 overlap parents 中 3 个来自同一 document 的 KEAP1-NRF2 PPI 与 MDR1 efflux paired panel
  （CHEMBL4145572；PMID 29750408）。

这些 paired panels 证明同一 compound 分别具有 upstream activity 和 transporter-substrate property，但没有证明
upstream perturbation **导致了该 compound 自身** 的 transporter disposition change；而且 query retrieval coverage
仍分别只有 5.87% 和 0%。因此同-document 配对也不改变 unavailable 结论。

最后一轮 bounded literature search 找到的最强 causal example 是 sulforaphane：它在 brain capillaries 中通过 NRF2
提高 P-gp/BCRP/MRP2，但 transporter function 与 brain accumulation 仍由 verapamil 等独立 probe/substrate 测量
（[PMID 24948812](https://pubmed.ncbi.nlm.nih.gov/24948812/)）。sulforaphane 自身在 Caco-2 中主要是 passive/
paracellular transport，而不是该实验所证明的 P-gp substrate
（[Ushida et al. 2016](https://www.jstage.jst.go.jp/article/fstr/22/1/22_127/_article/-char/en)）。这正是
`regulator A -> transporter state -> substrate B disposition` 的 subject switch，不能转写成 A 自身的 BBB evidence。

HIF/PHD bounded search 同样只找到 HIF-dependent GLUT1/glucose uptake，或 PHD inhibitor 作为肾 OAT1/OAT3
substrate 的证据；没有找到同一化合物通过 HIF/PHD 调控后又以 GLUT1 substrate 身份改变自身 BBB influx 的直接
实验。因此 Efflux 与 Influx 的 H1/H2 均冻结为 unavailable，不继续无边界扩张 regulator pathways。

## 下一版 graph gate

每个未来 task 的 H1/H2 family 在 freeze 前必须逐条回答：

1. 从 assay molecule 开始，沿 path 每一步追踪 causal subject。
2. 是否在某一步从 assay molecule 换成了 `another substrate/patient/cell/system`？
3. 若需要 query 具备 substrate、target engagement、metabolic precursor、sensitizer 或其它角色，retrieval row 是否
   对同一个 molecule 明确提供该角色？
4. 时间顺序是否允许 upstream perturbation 在 benchmark endpoint 之前发生？
5. 缺任一必需角色时标成 `requires_query_role`；只有环境影响时标成 `context_only`；不得进入主 H1/H2。

这套 gate 与 hop、scope 和 quality 正交：路径可以是正确的一跳且 assay 质量很好，但仍然因为预测对象发生切换而
不适合 self-molecule prediction。
