> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# Carcinogens 条件合并预览 v1

本轮只统一同义名称和精确识别的重复研究描述，保留物种、性别、品系、途径、剂量、疗程、
年龄、基因型及未解析限定。这是已执行的 staged preview，未发布为正式 benchmark。
没有新模型调用，也没有重新筛选或改写上一轮正负方向审阅结果。

## 当前已发布 cohort 的合并效果

| 项目 | 合并前 | 合并后 |
|---|---:|---:|
| 条件组 | 92 | 83 |
| 唯一分子–条件 rows | 1,464 | 1,404 |
| 正类 | 1,186 | 1,126 |
| 负类 | 278 | 278 |

57 个新分子–条件单元包含多个原始 rows，全部同向；折叠了 60 个重复正类 rows。
这不表示删除 60 条独立原始证据，所有原 benchmark row IDs 仍保留在合并 lineage 中。
本表按原 cohort 的同向标签折叠，不是基于全量已审 base 重建的新 gold。

主要合并包括 SD/Sprague-Dawley、雌性 SD 的不同词序，以及重复写在 model 中的
Rattus norvegicus / Mus musculus。F344/N 未与 F344 强行等同；普通 hamster 未与明确的
Mesocricetus auratus 强行等同。研究人数的精确文本被去掉，暴露时长等限定保留。

- `current_condition_mapping.csv`：原 92 个键到新键及原始正负数量。
- `current_conditions_after.csv`：合并去重后 83 个组的正负数量。
- `current_collapsed_rows.jsonl`：1,404 条预览 rows，含所有原 row IDs，无正式 split 分配。
- `current_label_collisions.jsonl`：57 个同向重合单元；当前 cohort 无正负冲突。

## 筛选前旧 gold 的独立诊断

全部 12,571 条旧 source votes 也应用相同映射。24 个同向研究单元去重后为 12,547 票，
没有研究单元内部的正负冲突；随后按现有共识规则重算，条件组 9,240 → 9,200，
已确定分子–条件 10,800 → 10,698（10,295 正、403 负）。未定单元从 1 增至 2。

新增未定是甲醛／Methanal 的 SD 大鼠吸入条件，两个既有来源标签分别为 0 和 1，
合并后各一票。名称、出处及实验条件尚需核对，不能直接认定是可信的生物学分歧。
详见 `new_label_conflicts.jsonl`；这条不在当前已发布 cohort 的冲突统计中。

旧 gold 的上述重算仅诊断合并效应，不能当成全量已审 records 的新 gold。
必要的三 parent／三 scaffold 支持门槛与实际三路 split 分配不同；summary 中的
coverage 只计算前者，未宣称得到新的可分配 benchmark。

## 全 base 审阅结果的覆盖

对上轮终结账本的全部 354,386 条 base records 逐 UID 生成了条件预览，输出为普通
Parquet：`reviewed_record_conditions.parquet`。其中 331,110 条是 binary 候选；
其正负数量仍为 296,079 / 35,031，所有 mixed、uncertain、scope/material holds 均保留。

binary records 的条件描述键为 204,386 → 204,264；全部 records 的描述键为
221,465 → 221,342，12,264 条记录的描述键改变。这里的键由原始模型、途径和完整
qualifying_conditions 生成，**不是已审核可用于 query 的 condition，也不是 gold 组**。
原文中可能含结果描述，物种也可能未明确或混合；没有用这些问题重新挡掉负类。
此预览没有完成全 base 的分子身份、研究出处去重、条件语义裁决或 gold 汇总。

结果表明，仅统一同义词对全量自由文本碎片化的改善很小。下一轮需要将研究叙述与真正
保留的条件轴分开，继续保存原始限定，而不是把每段不同原文当作独立条件。

## 复现与校验

在仓库根运行：

```sh
PYTHONPATH=. /data1/tianang/anaconda3/envs/vllm/bin/python \
  data/starling_data/new_tasks_gold_audit/targeted_review_v2/carcinogens_condition_review/preview.py
```

`policy.json` 保存精确规则；`summary.json` 保存计数与输入 SHA-256；`validation.json`
验证完整 UID 覆盖、方向不变、映射幂等、物种／途径原子保留、研究单元收支，以及输入
source votes、raw、已审标签和两种 split 六个输入文件哈希不变。
