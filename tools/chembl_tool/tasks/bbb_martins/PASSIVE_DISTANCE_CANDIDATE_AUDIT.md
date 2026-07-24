# BBB Passive H1/H2 candidate audit

日期：2026-07-22
状态：ChEMBL-only bounded search complete；v3 graph 与旧 paper matrix 未改变

## 冻结判据

一个 family 只有同时满足以下条件才可进入下一版 Passive node：

1. assay 实际测量目标的 functional activity，而非 binding-only、表达量或泛化 phenotype；
2. measured node 到当前 C passive/barrier family 的最短合格路径与声明的 H1/H2 一致；
3. 不与冻结 D/C assay 重叠，并通过 same-molecule causal continuity；
4. ChEMBL 36 中至少 50 个 molecular parents、3 个独立 documents；
5. 在 392 个 BBB test molecules、Morgan Tanimoto >=0.30、operational identity、node-level top-3 下，
   至少 10% query 能检索到一个 neighbor；最大单一 document 不得贡献超过 80% parents。

这里的 coverage gate 是在看 LLM 实验是否有足够样本可比较，不是把数据库规模当作生物学证据。

## 结果

| candidate | 声明层级 | assays | parents | >=1 coverage | top-3 coverage | 决定 |
|---|---:|---:|---:|---:|---:|---|
| MMP-2 activity | H1 | 477 | 5,171 | 26.79% | 11.73% | 接受 |
| ROCK2 activity | H1 | 453 | 7,560 | 17.35% | 8.67% | 接受 |
| ROCK1 activity | H1 | 397 | 4,699 | 16.84% | 7.91% | 拒绝：缺少 BBB isoform-specific support |
| MYLK/MLCK activity | H1 | 95 | 171 | 1.02% | 0% | 拒绝：coverage 不足 |
| RhoA activity | H1，不是 H2 | 6 | 95 | 0.77% | 0.51% | 拒绝：coverage 不足 |
| MMP-14 activity | 候选 H2 | 186 | 1,331 | 12.24% | 2.81% | 拒绝：存在一跳捷径风险 |

新接受的 MMP-2 + ROCK2 聚合在同一个 Passive H1 node、共享 top-3 budget 时，候选库共 12,731 parents；
392-query 的 >=1-neighbor coverage 为 36.22%，top-3 coverage 为 17.35%。这不是两个额外 parallel branches。

## 机制决定

### 接受：MMP-2 activity -> tight-junction integrity（H1）

MMP-2 activation 与 occludin degradation、tight-junction disruption 和 BBB permeability change 有直接实验支持，
所以从 MMP-2 measured state 到冻结 C barrier state 是一条边：

- [PMID 22378877](https://pubmed.ncbi.nlm.nih.gov/22378877/)
- [PMID 21857898](https://pubmed.ncbi.nlm.nih.gov/21857898/)
- [PMID 24035828](https://pubmed.ncbi.nlm.nih.gov/24035828/)

### 接受：ROCK2 activity -> tight-junction integrity（H1）

ROCK2/Rho-kinase perturbation 可改变 endothelial junction/cytoskeletal contraction 和 BBB permeability；现有 BBB
研究对 ROCK2 的支持比 ROCK1 明确，因此只接受 isoform-specific ROCK2：

- [PMID 28510599](https://pubmed.ncbi.nlm.nih.gov/28510599/)
- [PMID 26903801](https://pubmed.ncbi.nlm.nih.gov/26903801/)

### 拒绝：ROCK1、MYLK 和 RhoA

ROCK1 在 ChEMBL 中有数据，但当前文献不足以把 ROCK-family effect 稳健归因到 BBB-specific ROCK1。MYLK 的
barrier mechanism 有支持，但可检索 coverage 只有 1.02%。RhoA 对 junction/barrier 有直接作用，因此不能作为
`RhoA -> ROCK2 -> barrier` 的 H2；重新归为 H1 后 coverage 又不足：

- MYLK/barrier：[PMID 17419808](https://pubmed.ncbi.nlm.nih.gov/17419808/)、[PMID 16638813](https://pubmed.ncbi.nlm.nih.gov/16638813/)
- RhoA direct barrier shortcut：[PMID 20369389](https://pubmed.ncbi.nlm.nih.gov/20369389/)、[PMID 30227623](https://pubmed.ncbi.nlm.nih.gov/30227623/)

### 拒绝：MMP-14 作为 H2

MMP-14 激活 proMMP-2 的 edge 有支持，但 MMP-14 也可直接切割 ECM/vascular-barrier-related substrates；因此无法
证明它到 C barrier state 的最短路径必须经过 MMP-2。按照 “存在一跳捷径就不得声明 H2” 的冻结定义，本轮不把
MMP-14 放入 H2，也不为了填满树而强行降为 H1：

- MMP-14 -> proMMP-2：[PMID 10998420](https://pubmed.ncbi.nlm.nih.gov/10998420/)、[PMID 12630911](https://pubmed.ncbi.nlm.nih.gov/12630911/)
- direct vascular-barrier substrate shortcut risk：[PMID 32838837](https://pubmed.ncbi.nlm.nih.gov/32838837/)

## 下一版建议

下一版 BBB tree 的候选内容应为：

```text
Passive C
  one aggregated H1 node:
    existing MMP-9 + existing MMP-3 + new MMP-2 + new ROCK2
  H2 unavailable

Efflux C
  H1 unavailable
  H2 unavailable

Influx C
  H1 unavailable
  H2 unavailable
```

这意味着 tree schema 需要允许一个 C family 明确声明 `H1 unavailable`，而不是为了满足“恰有一个 H1”塞入因果
不自洽或 coverage 极低的节点。正式修改 schema、构建 v4 manifest/index 前，必须先把 unavailable 作为显式、
可审计状态加入 contract。

可复现入口与产物：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.audit_passive_distance_candidates

outputs/chembl_tool/tasks/bbb_martins/distance_expansion/analysis/v3/passive_candidate_audit/
  summary.json
  candidate_assays.tsv
  query_coverage.tsv
  report_zh.md
```
