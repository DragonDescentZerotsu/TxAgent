"""Apply the frozen Codex-authored biological assay relevance rubric.

This module is deliberately offline: it reads the normalized assay catalog and
does not call an LLM or any external service.  The rubric was authored by the
current Codex GPT-5.6-sol review and is intended to rank cumulative flat-assay
retrieval prefixes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Iterable

from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCORING_VERSION = "codex_gpt_5_6_sol_assay_relevance.v6"
SCORER = "codex-gpt-5.6-sol"


def _text(row: dict[str, Any]) -> tuple[str, str]:
    context = str(row["assay_context"]).lower().replace("_", " ")
    endpoints = " ".join(str(item["value"]) for item in row.get("endpoints", []))
    descriptions = " ".join(str(item) for item in row.get("assay_descriptions", []))
    return context, f"{context} {endpoints} {descriptions}".lower().replace("_", " ")


def _endpoint_text(row: dict[str, Any]) -> str:
    return " ".join(str(item["value"]) for item in row.get("endpoints", [])).lower().replace("_", " ")


def _endpoint_fraction(row: dict[str, Any], *patterns: str) -> float:
    endpoints = row.get("endpoints", [])
    total = sum(max(0, int(item.get("record_count", 0))) for item in endpoints)
    if total <= 0:
        return 0.0
    matched = sum(
        max(0, int(item.get("record_count", 0)))
        for item in endpoints
        if _has(str(item.get("value", "")).lower().replace("_", " "), *patterns)
    )
    return matched / total


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _semantic_bonus(text: str, patterns: Iterable[str], *, maximum: int = 4) -> int:
    return min(maximum, sum(bool(re.search(pattern, text)) for pattern in patterns))


def score_assay(row: dict[str, Any]) -> dict[str, Any]:
    task = row["task"]
    if task == "bbb_martins":
        score, rationale = _score_bbb(row)
    elif task == "bioavailability_ma":
        score, rationale = _score_bioavailability(row)
    elif task == "skin_reaction":
        score, rationale = _score_skin(row)
    elif task == "clintox":
        score, rationale = _score_clintox(row)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return {
        **row,
        "scoring_version": SCORING_VERSION,
        "scorer": SCORER,
        "scoring_method": "codex-authored deterministic biological relevance rubric",
        "relevance_score": int(max(0, min(100, score))),
        "rationale": rationale,
    }


def _score_bbb(row: dict[str, Any]) -> tuple[int, str]:
    context, text = _text(row)
    endpoints = _endpoint_text(row)
    computational = _has(
        context,
        r"in silico",
        r"predict",
        r"admetlab",
        r"swissadme",
        r"qikprop",
        r"database",
        r"model(?:ing|ling)?$",
        r"calculation",
    )
    direct_brain = (
        r"brain (?:concentration|exposure|uptake|distribution|tissue)",
        r"brain.to.(?:plasma|blood|serum)",
        r"kp uu brain",
        r"microdialysis",
        r"autoradiograph",
        r"\bpet\b",
        r"biodistribution",
        r"cns (?:access|exposure|penetration|distribution)",
    )
    if computational:
        return 18, "与BBB/CNS相关，但属于计算或文本预测环境，不是实验性CNS暴露测量。"
    if _has(context, r"csf", r"cerebrospinal"):
        return 82, "CSF暴露与CNS access相关，但属于脑实质暴露的代理指标。"
    if _has(context, r"unbound brain", r"kp uu brain", r"brain ecf microdialysis"):
        return 100, "直接测量游离脑暴露或脑细胞外液暴露，与meaningful CNS access高度一致。"
    if _has(context, *direct_brain):
        bonus = _semantic_bonus(context, direct_brain)
        return min(99, 94 + bonus), "直接或近直接测量脑组织摄取、脑暴露或脑/血分布。"
    if context.startswith("endpoint fallback") and _has(endpoints, r"unbound brain", r"kp uu brain"):
        return 99, "缺少实验context，但endpoint直接测量游离脑暴露。"
    if context.startswith("endpoint fallback") and _has(endpoints, *direct_brain):
        return 92, "缺少实验context，但endpoint直接描述脑暴露或脑分布。"
    if _has(context, r"in vivo", r"pharmacokinetic", r"preclinical", r"clinical study") and _has(
        text, r"bbb", r"brain", r"cns", r"permeability"
    ):
        return 90, "体内药代或BBB实验包含与CNS进入相关的观测结果。"
    if _has(
        text,
        r"brain endothelial",
        r"blood.brain barrier endothelial",
        r"bbb model",
        r"bbb permeability",
        r"pampa.bbb",
    ):
        base = 78 if _has(text, r"endothelial", r"bbb model") else 72
        return base, "实验性BBB屏障或被动通透模型，是CNS access的较强体外代理。"
    if _has(text, r"mdck", r"pampa", r"apparent permeability", r"papp"):
        return 66, "通透性模型提供间接的BBB进入先验，但不直接测量脑暴露。"
    if _has(text, r"efflux", r"influx", r"transporter", r"p.gp", r"abcb1", r"bcrp", r"abcg2"):
        return 58, "转运体或外排机制会影响BBB进入，但单独不足以确定整体CNS access。"
    if _has(endpoints, r"bbb permeability outcome", r"blood brain barrier permeability"):
        return 88, "记录直接给出BBB通透结局，但实验上下文缺失。"
    if _has(text, r"logp", r"logd", r"solubility", r"protein binding", r"pka"):
        return 35, "理化或结合性质与BBB进入有关，但距离体内CNS access较远。"
    return 8, "未发现与BBB或CNS暴露的明确实验联系。"


def _score_bioavailability(row: dict[str, Any]) -> tuple[int, str]:
    context, text = _text(row)
    endpoints = _endpoint_text(row)
    computational = _has(
        context,
        r"in silico",
        r"predict",
        r"pbpk",
        r"model(?:ing|ling)?$",
        r"calculation",
        r"database",
        r"literature",
    )
    if computational:
        return 20, "与口服ADME相关，但属于计算、汇总或文本环境，不是直接实验测量。"
    if _has(context, r"^endpoint fallback::oral bioavailability", r"^oral bioavailability$") or _has(
        endpoints, r"oral bioavailability outcome"
    ):
        return 99, "直接测量口服生物利用度，与F=20%任务终点一致。"
    if _has(
        context,
        r"\boral\b.*\bintravenous\b",
        r"\boral\b.*\biv\b",
        r"\bintravenous\b.*\boral\b",
        r"\biv\b.*\boral\b",
    ):
        return 96, "口服与静脉给药对照可直接估计绝对口服生物利用度。"
    if _has(context, r"oral pharmacokinetic", r"oral pk", r"oral dosing", r"in vivo oral study") and _has(
        endpoints, r"auc", r"cmax", r"exposure", r"bioavailability"
    ):
        return 90, "口服药代暴露是生物利用度的强实验代理，但未必包含静脉参照。"
    if _has(context, r"fraction absorbed", r"intestinal absorption", r"human intestinal absorption"):
        return 85, "直接测量肠道吸收或吸收分数，是口服F的强近端决定因素。"
    if _has(context, r"intestinal perfusion", r"ussing chamber", r"everted gut", r"intestinal permeability"):
        return 82, "肠道组织通透或灌流实验是口服吸收的强实验代理。"
    if _has(context, r"caco", r"mdck", r"pampa", r"permeability", r"artificial membrane"):
        return 78, "细胞或人工膜通透实验反映吸收潜力，但不覆盖完整首过过程。"
    if _has(context, r"dissolution", r"solubility"):
        return 74, "溶出或溶解度是口服吸收的重要前置因素，但不是完整生物利用度。"
    if _has(context, r"hepatocyte", r"liver microsome", r"microsome", r"intrinsic clearance"):
        return 76, "肝细胞或微粒体清除测量首过代谢的重要组成部分。"
    if _has(context, r"first.pass", r"hepatic clearance", r"gut wall extraction"):
        return 80, "首过提取或清除直接影响到达系统循环的口服剂量比例。"
    if _has(context, r"cyp", r"ugt", r"metabolism", r"metabolic stability"):
        return 66, "代谢实验与首过损失相关，但对口服F的解释仍是间接的。"
    if _has(context, r"efflux", r"transporter", r"abcb1", r"abcg2"):
        return 62, "肠道转运或外排可影响吸收，但单独不能决定口服F。"
    if _has(context, r"auc", r"cmax", r"pharmacokinetic", r"exposure"):
        return 68, "系统暴露或一般药代结果与口服F相关，但给药和参照上下文不足。"
    if _has(context, r"logp", r"logd", r"pka", r"protein binding"):
        return 32, "理化或结合性质仅提供远端口服ADME先验。"
    if _has(endpoints, r"fraction absorbed", r"intestinal absorption", r"absorption outcome"):
        return 64, "endpoint涉及吸收，但assay context过于泛化，无法确认是直接肠道吸收实验。"
    if _has(endpoints, r"permeability", r"dissolution", r"solubility"):
        return 52, "endpoint提示吸收相关性质，但assay context不足以确定实验与口服F的距离。"
    if _has(endpoints, r"clearance", r"metabolism", r"metabolic stability"):
        return 48, "endpoint提示清除或代谢，但缺少可定位首过过程的assay context。"
    return 8, "未发现与绝对口服生物利用度的明确实验联系。"


def _score_skin(row: dict[str, Any]) -> tuple[int, str]:
    context, text = _text(row)
    direct_fraction = _endpoint_fraction(
        row,
        r"sensitization",
        r"allergic contact dermatitis",
        r"contact allergy",
        r"positive patch test",
    )
    irritation_fraction = _endpoint_fraction(
        row, r"irritation", r"corrosion", r"local tissue injury"
    )
    computational = _has(
        context,
        r"in silico",
        r"predict",
        r"calculated",
        r"model(?:ing|ling)?$",
        r"admetlab",
        r"caesar",
        r"derek",
        r"qsar",
        r"toxtree",
    )
    photo_specific = _has(context, r"photo", r"\buv\b")
    if computational:
        return 20, "与皮肤致敏相关但属于计算预测，不是实验性致敏结局。"
    if photo_specific:
        return 18, "主要测量光毒或光敏反应，不属于普通皮肤致敏/接触过敏终点。"
    if irritation_fraction > direct_fraction and irritation_fraction >= 0.5:
        return 15, "该context虽可能使用皮肤实验形式，但记录主要测量刺激、腐蚀或局部损伤。"
    if context.startswith("endpoint fallback") and direct_fraction > 0:
        return 94, "缺少实验context，但endpoint直接描述皮肤致敏或接触过敏结局。"
    if _has(context, r"human patch test", r"contact allergy", r"allergic contact dermatitis") and direct_fraction > 0:
        return 100, "直接测量人体接触过敏或临床致敏结局。"
    if _has(context, r"\bllna\b", r"local lymph node", r"gpmt", r"guinea pig maxim", r"buehler"):
        return 96, "属于验证过的动物皮肤致敏终点实验。"
    if _has(context, r"patch test", r"contact sensitization", r"skin sensitization test", r"sensitization assay") and direct_fraction >= 0.2:
        return 92, "直接或近直接测量皮肤致敏/接触过敏。"
    if _has(text, r"dpra", r"kdpra", r"mdpra", r"peptide reactiv", r"haptenation", r"cysteine depletion", r"lysine depletion"):
        return 84, "测量蛋白/肽反应性，是皮肤致敏AOP的近端实验事件。"
    if _has(text, r"keratinosens", r"lusens", r"nrf2", r"keap1", r"are.nrf2"):
        return 80, "测量角质形成细胞应激或NRF2激活，是致敏AOP实验代理。"
    if _has(text, r"h.clat", r"u.sens", r"dendritic cell activation", r"cd86", r"cd54"):
        return 82, "测量树突细胞激活，是致敏AOP的实验代理。"
    if _has(text, r"t cell", r"lymphocyte proliferation", r"lymph node proliferation", r"dendritic cell migration"):
        return 78, "测量致敏相关免疫细胞反应，但未必形成完整终末结局。"
    if _has(
        context,
        r"sensitization",
        r"hypersensitivity",
        r"contact sensitivity",
        r"allerg",
        r"epicutaneous",
        r"intradermal",
        r"elicitation",
        r"challenge",
        r"hript",
        r"human maximization",
        r"human induction",
    ) and direct_fraction > 0:
        return 88, "实验上下文明确涉及致敏或迟发型超敏反应。"
    if direct_fraction >= 0.5:
        return 72, "合并endpoint以致敏结局为主，但assay context过于泛化，相关性低于明确的致敏实验。"
    if _has(text, r"irritation", r"corrosion", r"local tissue injury"):
        return 15, "主要测量刺激、腐蚀或局部损伤，不是皮肤致敏终点。"
    if _has(text, r"diffusion", r"dermal absorption", r"flux", r"skin retention", r"permeability"):
        return 22, "主要测量皮肤暴露或渗透，距离免疫致敏结局较远。"
    if _has(text, r"inflammatory", r"cytokine", r"keratinocyte damage"):
        return 45, "炎症或细胞损伤与皮肤反应相关，但对接触致敏缺乏特异性。"
    return 7, "未发现与皮肤致敏或接触过敏的明确实验联系。"


def _score_clintox(row: dict[str, Any]) -> tuple[int, str]:
    context, text = _text(row)
    source_id = str(row.get("source_id", ""))
    if source_id == "clinical_trial_failure":
        if _has(context, r"toxicity absent") or _has(
            text,
            r"not toxicity",
            r"no (?:significant |signs? of )?toxicity",
            r"well tolerated",
        ):
            return 94, "直接临床记录明确描述未观察到毒性或终止并非由毒性导致；与任务高度相关，但不是毒性失败阳性结局。"
        return 100, "直接记录毒性导致的临床试验、招募或药物开发失败；与冻结任务终点最接近，但direct gate仍待人工QA。"
    if _has(context, r"predict", r"in silico", r"qsar", r"model(?:ing|ling)?"):
        return 18, "属于毒性相关计算预测，不能作为实验性临床失败证据。"
    if source_id == "organ_specific_toxicity":
        if _has(context, r"human clinical"):
            return 84, "人体临床器官毒性是临床失败的强近端证据，但未直接说明试验因毒性终止。"
        if _has(context, r"in vivo", r"animal"):
            return 72, "体内器官毒性可转移到临床风险，但仍是非临床代理。"
        return 56, "器官相关体外毒性提示潜在风险，但距离毒性导致的临床失败较远。"
    if source_id == "nonclinical_in_vivo_toxicity":
        if _has(text, r"mortality", r"survival", r"ld50", r"noael", r"loael"):
            return 74, "体内死亡或剂量限制毒性是较强非临床安全代理，但不等于临床试验失败。"
        return 68, "非临床体内毒性与临床安全风险相关，但不能直接决定冻结的来源类别。"
    if source_id == "genotoxicity_carcinogenicity":
        if _has(text, r"animal carcinogenicity", r"in vivo"):
            return 66, "体内致癌或遗传毒性是重要开发风险，但与临床毒性失败仍为间接关系。"
        return 52, "遗传毒性实验反映开发风险机制，但不是临床试验失败结局。"
    if source_id == "general_cytotoxicity":
        return 42, "一般细胞毒性是广义危害信号，跨到临床失败的特异性较低。"
    if source_id == "cellular_stress":
        return 34, "细胞应激或损伤通路属于远端机制证据，不能单独支持临床失败类别。"
    if source_id == "off_target_ddi_exposure":
        if _has(text, r"clinical drug interaction", r"clinical exposure", r"toxicokinetic"):
            return 55, "临床相互作用或暴露变化可能导致安全问题，但未直接观察毒性失败。"
        if _has(text, r"herg", r"cardiac", r"qt"):
            return 50, "心脏安全相关脱靶信号与开发风险有关，但只是机制代理。"
        return 30, "代谢酶、转运体、结合或一般脱靶证据与临床失败关系较远。"
    return 7, "未发现与毒性导致的临床试验或开发失败的明确联系。"


def score_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [score_assay(row) for row in catalog]
    rows.sort(key=lambda row: (-row["relevance_score"], row["assay_context"], row["assay_id"]))
    for rank, row in enumerate(rows, start=1):
        row["relevance_rank"] = rank
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    catalog_path = Path(args.catalog)
    output_dir = Path(args.output_dir)
    catalog = read_jsonl(catalog_path)
    ranked = score_catalog(catalog)
    ranked_path = output_dir / "ranked_assays.jsonl"
    write_jsonl_atomic(ranked_path, ranked)
    write_json_atomic(
        output_dir / "score_manifest.json",
        {
            "scoring_version": SCORING_VERSION,
            "scorer": SCORER,
            "external_model_calls": 0,
            "catalog": str(catalog_path.resolve()),
            "catalog_sha256": sha256_file(catalog_path),
            "ranked_assays": str(ranked_path.resolve()),
            "ranked_assays_sha256": sha256_file(ranked_path),
            "task": catalog[0]["task"] if catalog else None,
            "n_scored": len(ranked),
            "score_distribution": {
                str(score): sum(row["relevance_score"] == score for row in ranked)
                for score in sorted({row["relevance_score"] for row in ranked}, reverse=True)
            },
        },
    )
    print(f"[{catalog[0]['task']}] Codex-scored {len(ranked):,} assay units")


if __name__ == "__main__":
    main()
