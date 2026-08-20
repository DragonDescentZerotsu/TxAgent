"""Build a portable technical report for the qualifying-conditions audit.

The report consumes the reproducible outputs from
``analyze_qualifying_conditions.py`` and emits the bounded artifact contract
used by the Data Analytics portable report renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_ANALYSIS_DIR = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/qualifying_conditions_analysis_v1"
)

GROUP_LABELS = {
    "other_unresolved": "其他／未解析",
    "food_or_prandial_state": "食物与空腹／餐后状态",
    "first_pass_metabolism_or_clearance": "首过代谢或清除机制",
    "molecular_form_salt_or_prodrug": "盐型、前药与分子形态",
    "formulation_delivery_or_solid_state": "制剂、递送与固态",
    "co_treatment_or_ddi": "合并用药与药物相互作用",
    "disease_organ_function_or_surgery": "疾病、器官功能与手术",
    "dose_regimen_or_sampling_time": "剂量、给药方案与采样时间",
    "study_arm_comparator_or_model_context": "研究组、对照或模型语境",
    "absorption_solubility_or_permeability": "吸收、溶解度或渗透机制",
    "genotype_or_metabolizer_phenotype": "基因型与代谢表型",
    "demographic_or_physiologic_state": "人口学与生理状态",
    "administration_route_or_gi_environment": "给药途径与胃肠环境",
    "processing_storage_or_environment": "加工、储存与环境条件",
}

INTERPRETATION_LABELS = {
    "true_contextual_modifier": "真正的情境修饰条件",
    "mechanism_or_explanation_not_condition": "机制／解释，不是严格条件",
    "study_metadata_or_nonexperimental_context": "研究元数据／非实验条件",
    "unresolved": "未解析",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_artifact(analysis_dir: Path) -> dict:
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    groups = _read_csv(analysis_dir / "primary_group_counts.csv")
    values = _read_csv(analysis_dir / "condition_value_counts.csv")

    canonical = summary["canonical_claims"]
    quality = summary["quality_findings"]
    nonempty = canonical["n_nonempty"]

    group_rows = []
    content_type_counts: dict[str, int] = defaultdict(int)
    content_type_exact: dict[str, int] = defaultdict(int)
    for row in groups:
        interpretation = row["field_interpretation"]
        record_count = int(row["canonical_claim_records"])
        exact_count = int(row["canonical_claim_exact_values"])
        content_type_counts[interpretation] += record_count
        content_type_exact[interpretation] += exact_count
        group_rows.append(
            {
                "group": GROUP_LABELS[row["group"]],
                "group_id": row["group"],
                "records": record_count,
                "share_nonempty": float(row["canonical_claim_share_nonempty"]),
                "exact_values": exact_count,
                "normalized_values": int(row["canonical_claim_normalized_values"]),
                "interpretation": INTERPRETATION_LABELS[interpretation],
            }
        )

    content_order = [
        "true_contextual_modifier",
        "mechanism_or_explanation_not_condition",
        "study_metadata_or_nonexperimental_context",
        "unresolved",
    ]
    content_rows = [
        {
            "content_type": INTERPRETATION_LABELS[key],
            "content_type_id": key,
            "records": content_type_counts[key],
            "share_nonempty": content_type_counts[key] / nonempty,
            "exact_values": content_type_exact[key],
            "denominator": nonempty,
        }
        for key in content_order
    ]

    top_values = [
        {
            "qualifying_conditions": row["qualifying_conditions"],
            "records": int(row["canonical_claim_record_count"]),
            "share_nonempty": int(row["canonical_claim_record_count"]) / nonempty,
            "group": GROUP_LABELS[row["primary_group"]],
            "interpretation": INTERPRETATION_LABELS[row["field_interpretation"]],
        }
        for row in values[:25]
    ]

    headline = [
        {
            "total_claims": canonical["n_rows"],
            "nonempty_claims": nonempty,
            "nonempty_rate": canonical["nonempty_rate"],
            "distinct_exact": canonical["n_distinct_exact_nonempty"],
            "distinct_normalized": canonical["n_distinct_normalized_nonempty"],
            "singleton_record_share": quality["singleton_record_share_of_nonempty"],
            "multilabel_record_share": quality["multi_label_claim_record_share_of_nonempty"],
        }
    ]

    source = {
        "id": "canonical_claims_source",
        "label": "Bioavailability canonical direct claims v2",
        "path": "data/starling_data/bioavailability_ma/canonical_direct_v2/direct_claims.parquet",
        "query": {
            "language": "DuckDB SQL + Python taxonomy",
            "sql": (
                "SELECT canonical_claim_id, qualifying_conditions "
                "FROM read_parquet('data/starling_data/bioavailability_ma/"
                "canonical_direct_v2/direct_claims.parquet')"
            ),
            "description": (
                "逐行读取 canonical direct claims；空值定义为 null、空字符串、空 list/tuple/set/dict；"
                "非空字符串按原文计数，并由 analyze_qualifying_conditions.py 中透明的大小写不敏感多标签正则 taxonomy 分类。"
            ),
            "executed_at": summary["generated_at"],
            "tables_used": [
                "data/starling_data/bioavailability_ma/canonical_direct_v2/direct_claims.parquet",
                "data/starling_data/bioavailability_ma/canonical_direct_v2/direct_source_rows.parquet",
            ],
            "filters": [
                "主分析粒度：跨来源去重后的 canonical claim，一行算一条 record",
                "分类和 distinct-value 统计只使用非空 qualifying_conditions",
                "相同 PMID 的不同 canonical claims 仍是不同 records",
            ],
            "metric_definitions": [
                "非空率 = 非空 qualifying_conditions canonical claims / 全部 canonical claims",
                "分组占比 = 该 primary group records / 全部非空 qualifying_conditions records",
                "primary group = 按预先声明的类别优先级选择第一个匹配；所有匹配仍保存在明细 CSV",
            ],
        },
    }

    generated_at = summary["generated_at"]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Bioavailability qualifying_conditions 数据审计",
        "description": "当前 canonical direct-v2 检索源中 qualifying_conditions 的频数、taxonomy 与药代知识一致性检查。",
        "generatedAt": generated_at,
        "sources": [source],
        "cards": [
            {
                "id": "total_claims",
                "dataset": "headline",
                "sourceId": "canonical_claims_source",
                "description": "跨来源去重后的 canonical claims。",
                "metrics": [{"label": "Canonical records", "field": "total_claims", "format": "number"}],
            },
            {
                "id": "nonempty_rate",
                "dataset": "headline",
                "sourceId": "canonical_claims_source",
                "description": "qualifying_conditions 非空的 canonical claims 占比。",
                "metrics": [
                    {"label": "有 qualifying_conditions", "field": "nonempty_rate", "format": "percent"},
                    {"label": "Records", "field": "nonempty_claims", "format": "number"},
                ],
            },
            {
                "id": "distinct_exact",
                "dataset": "headline",
                "sourceId": "canonical_claims_source",
                "description": "按原文完全匹配的非空值数量。",
                "metrics": [
                    {"label": "不同原始字符串", "field": "distinct_exact", "format": "number"},
                    {"label": "简单规范化后", "field": "distinct_normalized", "format": "number"},
                ],
            },
            {
                "id": "singleton_share",
                "dataset": "headline",
                "sourceId": "canonical_claims_source",
                "description": "其原始字符串只出现一次的非空 records 占比，反映自由文本长尾程度。",
                "metrics": [
                    {"label": "Singleton record share", "field": "singleton_record_share", "format": "percent"}
                ],
            },
        ],
        "charts": [
            {
                "id": "content_type_chart",
                "title": "非空 qualifying_conditions 的语义组成",
                "subtitle": "互斥 primary-group 汇总；分母为 34,107 条非空 canonical claims",
                "intent": "composition",
                "question": "这些文本有多少是真正的情境条件，而不是机制说明或研究元数据？",
                "rationale": "四类汇总比 18,690 个长尾字符串更适合比较字段的语义纯度。",
                "comparisonContext": {
                    "denominator": "34,107 non-empty canonical claims",
                    "grain": "canonical claim",
                    "normalization": "mutually exclusive primary category",
                    "unit": "records",
                },
                "type": "horizontalBar",
                "dataset": "content_types",
                "sourceId": "canonical_claims_source",
                "encodings": {
                    "x": {"field": "content_type", "type": "nominal", "label": "语义类型"},
                    "y": {"field": "records", "type": "quantitative", "label": "Records", "format": "number"},
                    "tooltip": [
                        {"field": "records", "type": "quantitative", "label": "Records", "format": "number"},
                        {"field": "share_nonempty", "type": "quantitative", "label": "占非空比例", "format": "percent"},
                        {"field": "exact_values", "type": "quantitative", "label": "不同原始字符串", "format": "number"},
                    ],
                },
                "xAxisTitle": "Canonical records",
                "yAxisTitle": "",
                "valueFormat": "number",
                "layout": "full",
                "maxRows": 4,
                "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True},
                "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
            }
        ],
        "tables": [
            {
                "id": "primary_group_table",
                "title": "14 个 primary groups",
                "subtitle": "每条非空 claim 只归入一个 primary group；多标签匹配另保存在明细输出中。",
                "dataset": "primary_groups",
                "defaultSort": {"field": "records", "direction": "desc"},
                "density": "dense",
                "sourceId": "canonical_claims_source",
                "layout": "full",
                "columns": [
                    {"field": "group", "label": "Group", "type": "text"},
                    {"field": "records", "label": "Records", "format": "number"},
                    {"field": "share_nonempty", "label": "非空占比", "format": "percent"},
                    {"field": "exact_values", "label": "原始字符串数", "format": "number"},
                    {"field": "interpretation", "label": "语义判断", "type": "text"},
                ],
            },
            {
                "id": "top_values_table",
                "title": "出现次数最多的 25 个原始值",
                "subtitle": "完整 18,690 个 exact values 见 condition_value_counts.csv。",
                "dataset": "top_values",
                "defaultSort": {"field": "records", "direction": "desc"},
                "density": "dense",
                "sourceId": "canonical_claims_source",
                "layout": "full",
                "columns": [
                    {"field": "qualifying_conditions", "label": "原始值", "type": "text"},
                    {"field": "records", "label": "Records", "format": "number"},
                    {"field": "share_nonempty", "label": "非空占比", "format": "percent"},
                    {"field": "group", "label": "Primary group", "type": "text"},
                    {"field": "interpretation", "label": "语义判断", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Bioavailability qualifying_conditions 数据审计", "layout": "full"},
            {
                "id": "summary",
                "type": "markdown",
                "layout": "full",
                "sourceId": "canonical_claims_source",
                "body": (
                    "## 技术摘要\n\n"
                    "当前 agent direct-retrieval 的 canonical source 共有 **113,080** 条去重 claims，其中 "
                    "**34,107 条（30.16%）**带非空 `qualifying_conditions`。按原文完全匹配有 "
                    "**18,690 种**；其中 **14,550 条 records（42.66%）**的原始值只出现一次，说明它本质上是长尾自由文本，而不是受控枚举。\n\n"
                    "高精度 taxonomy 将文本分为 14 个 primary groups。只有 **56.01%** 可较明确视为真正的给药、患者、制剂或环境条件；"
                    "**16.94%** 更像机制／解释，**2.95%** 是研究元数据，另有 **24.10%** 未可靠解析。因此该字段总体符合影响口服生物利用度的药代知识，"
                    "但字段语义并不纯，不能把“非空”直接解释为一种统一的条件效应。"
                ),
            },
            {"id": "metrics", "type": "metric-strip", "cardIds": ["total_claims", "nonempty_rate", "distinct_exact", "singleton_share"], "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 范围与口径\n\n"
                    "主口径是一条跨来源去重后的 **canonical claim**，不是 molecule、PMID 或原始抽取行。"
                    "所以这里回答的“每种情况有多少 records”指该字符串出现在多少条 canonical claims 中。"
                    "敏感性核对的 pre-dedup direct source 有 115,560 行，其中 34,318 行非空、18,829 个 exact values；"
                    "去重前后结论基本一致。"
                ),
            },
            {"id": "content_type_heading", "type": "markdown", "body": "## 字段语义组成", "layout": "full"},
            {"id": "content_type_chart_block", "type": "chart", "chartId": "content_type_chart", "layout": "full"},
            {
                "id": "taxonomy_method",
                "type": "markdown",
                "layout": "full",
                "sourceId": "canonical_claims_source",
                "body": (
                    "## 分类方法\n\n"
                    "使用可复现、大小写不敏感的多标签正则 taxonomy。一个文本可以命中多个 group；为了给出互斥汇总，"
                    "primary group 取预先声明优先级中的第一个命中项。**3,164 条（9.28%）**非空 claims 命中多个 group。"
                    "该 taxonomy 是字段审计工具，不是因果模型，也不是人工逐条裁定的 gold ontology。"
                ),
            },
            {"id": "primary_group_table_block", "type": "table", "tableId": "primary_group_table", "layout": "full"},
            {
                "id": "knowledge",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 与 bioavailability 知识的一致性\n\n"
                    "总体上符合：食物状态会改变胃排空、胆汁分泌、溶出以及酶／转运体活动；制剂和固态影响释放与溶出；"
                    "盐型、前药和晶型影响溶解度、稳定性与渗透；合并用药会通过 CYP、转运体或胃内 pH 改变暴露；"
                    "疾病、器官功能、胃肠手术、基因型和代谢表型也会改变吸收或首过过程。剂量与给药方案可暴露非线性／饱和动力学。\n\n"
                    "但并非所有内容都是真正的 qualifying condition：`extensive first-pass metabolism`、低溶解度等是机制或分子属性；"
                    "`control`、baseline、model 等是研究设计元数据；部分疾病或肾功能描述主要影响清除和系统暴露，未必改变绝对口服 F。"
                    "而且条件的影响方向通常依赖具体分子与制剂，不能从 group 名称推断统一增减方向。"
                ),
            },
            {"id": "top_values_table_block", "type": "table", "tableId": "top_values_table", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 局限与建议\n\n"
                    "`other/unresolved` 仍占 24.10%，且 exact-value singleton 很多；若用于正式 ontology，需要对高频未解析值和随机长尾样本做人审。"
                    "当前 gold 规则要求 `qualifying_conditions` 为空，是一个保守的 molecule-only label 边界，但会同时排除真正的条件依赖 claim、机制描述和研究元数据，"
                    "因此它安全但偏宽。agent retrieval 中保留该字段是合理的，应把它当作 analog transfer 的上下文和降权依据，而不是 label vote 或统一的 invalid flag。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 可继续验证的问题\n\n"
                    "1. 对 `other/unresolved` 的高频值做人工复核后，能否形成更稳定的 controlled vocabulary？\n"
                    "2. 将真正的条件修饰项与机制／研究元数据分开后，gold 覆盖率与 parent label 是否发生实质变化？\n"
                    "3. agent 是否在 final reasoning 中正确使用这些条件，还是忽略／过度采用了它们？"
                ),
            },
        ],
    }

    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "content_types": content_rows,
                "primary_groups": group_rows,
                "top_values": top_values,
            },
        },
        "sources": [source],
        "packageInfo": {
            "root": ".",
            "manifestPath": "artifact.json",
            "snapshotPath": "artifact.json#snapshot",
            "analysisScript": "tools/chembl_tool/tasks/bioavailability_ma/analyze_qualifying_conditions.py",
            "notes": "主图仅展示四类语义组成；18,690 个 exact values 以可排序 CSV 提供，避免长尾标签图形化造成不可读。",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.analysis_dir / "artifact.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_artifact(args.analysis_dir), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
