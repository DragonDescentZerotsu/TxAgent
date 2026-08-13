"""Writers for the Bio unanimous-positive trace diagnosis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic


def write_outputs(
    output: Path, summary: dict[str, Any], cohort_rows: list[dict[str, Any]]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "summary.json", summary)
    write_jsonl_atomic(output / "case_rows.jsonl", cohort_rows)
    _write_tsv(output / "similarity_diagnostic.tsv", summary["similarity_diagnostic"])
    _write_trace_states(output / "trace_states.tsv", summary["trace_states_by_outcome"])
    (output / "report.md").write_text(_report(summary), encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_trace_states(path: Path, states: dict[str, Any]) -> None:
    rows = [
        {
            "outcome": outcome,
            "n": values["n"],
            "single_prior": json.dumps(values["single_prior"], sort_keys=True),
            "group_direction": json.dumps(values["group_direction"], sort_keys=True),
            "group_transferability": json.dumps(values["group_transferability"], sort_keys=True),
            "full_pool_correct": values["full_pool_correct"],
        }
        for outcome, values in states.items()
    ]
    _write_tsv(path, rows)


def _report(summary: dict[str, Any]) -> str:
    stratum = summary["all_positive_neighbor_stratum"]
    outcomes = stratum["outcomes"]
    reliability = summary["harm_gold_reliability"]
    lines = [
        "# Bioavailability unanimous-positive neighbor trace diagnosis", "", "## 结论", "",
        "Bio 的主要问题不是 gold 噪声，而是一个可重复的决策链：较低结构相似度使三个正类训练标签",
        "在 group 层被降为 neutral/unclear；single 再用通用 drug-likeness 风险形成 low 或 mixed prior；",
        "final 把‘正证据不可迁移/信息不足’系统性解释成 F<20%。", "",
        f"在 {stratum['n']} 个三邻居全为 Y=1 的样本中，gold 有 {stratum['gold']['1']} 个 Y=1；",
        f"agent 仍预测 low {stratum['agent']['0']} 次，其中误伤 {outcomes.get('harm_flip_to_low', 0)}、",
        f"救回 {outcomes.get('rescue_flip_to_low', 0)}，low precision 只有 "
        f"{stratum['agent_low_precision_for_gold_low']:.1%}。", "", "## 相似度不是有效的 low 判别器", "",
        "| top-1 similarity | n | gold Y=1 | agent 预测 low | low precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["similarity_diagnostic"]:
        lines.append(
            f"| {row['similarity_bin']} | {row['n']} | {row['gold_positive_rate']:.1%} | "
            f"{row['agent_predicted_low_rate']:.1%} | "
            f"{row['agent_low_precision_for_gold_low']:.1%} |"
        )
    lines += [
        "", "top-1 越远，agent 越常推翻三票正类；但真实正类率始终约 84–87%。",
        "相似度确实触发了改票，却没有识别出哪些样本真的低于 20%。", "", "## Trace 层定位", "",
        "| outcome | n | single prior | group direction | group transferability | full-pool correct |",
        "|---|---:|---|---|---|---:|",
    ]
    for name, row in summary["trace_states_by_outcome"].items():
        lines.append(
            f"| {name} | {row['n']} | {_compact(row['single_prior'])} | "
            f"{_compact(row['group_direction'])} | {_compact(row['group_transferability'])} | "
            f"{row['full_pool_correct']}/{row['n']} |"
        )
    lines += [
        "", f"39 个误伤中，{reliability['unanimous_gold']} 个 gold 是实验记录全票一致；source record "
        f"中位数为 {reliability['median_source_record_count']}，范围 "
        f"{reliability['min_source_record_count']}–{reliability['max_source_record_count']}。",
        "这排除了‘主要是模糊 gold 被 agent 正确纠正’。Full-Starling direct 只修复其中一部分，",
        "说明更近、更完整的证据有帮助，但低偏置不是 train-label 卡片单独引入的。", "",
        "## 病因排序与最小改进方向", "",
        "1. **阈值校准错误**：任务是 `F >= 20%`，不是判断优秀 drug-like exposure。",
        "2. **direction 与 transferability 混淆**：低 transferability 应保留弱正向，而非变成反证。",
        "3. **final 默认低偏置**：`mixed single + insufficient/neutral group` 通常落到 low。",
        "4. **卡片信息压缩是次要放大器**：只有 Y，没有定量 F%、species/formulation/context。", "",
        "下一轮只冻结一个 Bio 合同：`低 transferability 只能降低权重，不能改变 evidence direction；",
        "预测 low 必须有明确的 F<20 支持，mixed/insufficient 不得默认 low。` 先跑 matched valid，",
        "再检查 full-pool direct；不增加 router 或多候选 ablation，formal test 继续不运行。", "",
        "## Provenance", "", "该报告由 deterministic trace audit 生成，不含新模型调用。逐样本字段保存在",
        "`case_rows.jsonl`，汇总保存在 `summary.json`。",
    ]
    return "\n".join(lines) + "\n"


def _compact(values: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in values.items())
