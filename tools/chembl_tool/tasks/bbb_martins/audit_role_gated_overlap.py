"""Audit same-parent overlap between BBB distance families and transporter roles.

This is a feasibility audit, not a source builder.  It leaves the frozen v3
manifest/index untouched and asks whether an H1/H2 molecule has independent
ChEMBL evidence that the same molecular parent is itself transported by the
downstream transporter required by the proposed causal path.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    normalize_molecule_identity,
)
from tools.chembl_tool.common.retrieval_policy import decide_candidate


DEFAULT_H_EVIDENCE = Path(
    "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/extension/"
    "bbb_distance_extension_evidence.jsonl"
)
DEFAULT_H_INDEX = Path(
    "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/extension/"
    "bbb_distance_extension_index.pkl"
)
DEFAULT_C_EVIDENCE = Path(
    "outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_molecule_evidence.jsonl"
)
DEFAULT_INPUT = Path("data/processed/BBB_Martins/test.jsonl")
DEFAULT_OUT_DIR = Path(
    "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/analysis/v3/role_gated_overlap"
)

MIN_SIMILARITY = 0.30
TOP_K = 3

EFFLUX_FAMILIES = ("nrf2_activation", "keap1_nrf2_interaction")
INFLUX_FAMILIES = ("hif1_activation", "phd2_activity")


@dataclass(frozen=True)
class RoleDecision:
    role: str | None
    reason: str


def classify_role_evidence(row: Mapping[str, Any]) -> RoleDecision:
    """Return a strict same-molecule transporter-role assignment.

    Inhibitor, binding, ATPase, probe-accumulation, and transporter-expression
    assays are deliberately excluded: they do not show that the activity-row
    molecule is itself transported.
    """

    group_id = str(row.get("group_id") or "").strip()
    standard_type = str(row.get("standard_type") or "").strip().lower()
    standard_relation = str(row.get("standard_relation") or "").strip()
    standard_value = _float_or_none(row.get("standard_value"))
    activity_comment = str(row.get("activity_comment") or "").strip().lower()
    description = str(row.get("assay_description") or "").strip().lower()
    target_text = " ".join(
        str(row.get(field) or "")
        for field in ("target_genes", "target_pref_name", "target_synonyms")
    ).lower()
    full_text = f"{description} {target_text}"

    if group_id == "Tier 3.efflux_functional_ratio_or_bidirectional":
        transporter = _efflux_transporter(full_text)
        if transporter is None:
            return RoleDecision(None, "efflux_transport_not_abcb1_abcg2_abcc2_specific")
        if standard_type not in {"ratio", "ratio_papp"}:
            return RoleDecision(None, "efflux_single_direction_papp_does_not_prove_substrate_role")
        if "probe" in description or _probe_only(description):
            return RoleDecision(None, "efflux_probe_molecule_not_activity_row_molecule")
        if not _is_efflux_ratio_description(description):
            return RoleDecision(None, "ratio_endpoint_is_not_a_compound_efflux_ratio")
        if not _supports_positive_role(
            value=standard_value,
            relation=standard_relation,
            activity_comment=activity_comment,
            minimum_value=2.0,
        ):
            return RoleDecision(None, "efflux_ratio_does_not_support_positive_substrate_role")
        return RoleDecision(f"efflux_substrate:{transporter}", "positive_compound_efflux_ratio")

    if group_id == "Tier 3.efflux_transport_or_accumulation":
        transporter = _efflux_transporter(full_text)
        direct_compound_transport = standard_type == "drug transport" and (
            "drug transport" in description
            or "transporter-mediated drug uptake" in description
            or "transporter-mediated drug transport" in description
        )
        if (
            transporter is not None
            and direct_compound_transport
            and not _probe_only(description)
            and _supports_positive_role(
                value=standard_value,
                relation=standard_relation,
                activity_comment=activity_comment,
                minimum_value=0.0,
            )
        ):
            return RoleDecision(f"efflux_substrate:{transporter}", "positive_direct_compound_transport")
        return RoleDecision(None, "efflux_accumulation_or_inhibition_does_not_prove_substrate_role")

    if group_id == "Tier 4.influx_functional_uptake_or_transport":
        if not _is_glut1(full_text):
            return RoleDecision(None, "influx_transport_not_glut1_specific")
        direct_substrate = (
            "substrate activity" in description
            and ("drug uptake" in description or "cellular uptake" in description)
        )
        if (
            direct_substrate
            and "inhibition of" not in description
            and _supports_positive_role(
                value=standard_value,
                relation=standard_relation,
                activity_comment=activity_comment,
                minimum_value=0.0,
            )
        ):
            return RoleDecision("influx_substrate:SLC2A1", "positive_direct_glut1_substrate_uptake")
        return RoleDecision(None, "glut1_probe_uptake_or_inhibition_does_not_prove_substrate_role")

    return RoleDecision(None, "outside_strict_role_groups")


def _efflux_transporter(text: str) -> str | None:
    aliases = {
        "ABCB1": ("abcb1", "p-glycoprotein", "p glycoprotein", "p-gp", "pgp", "mdr1"),
        "ABCG2": ("abcg2", "bcrp", "breast cancer resistance protein"),
        "ABCC2": ("abcc2", "mrp2", "multidrug resistance-associated protein 2"),
    }
    for transporter, names in aliases.items():
        if any(name in text for name in names):
            return transporter
    return None


def _is_glut1(text: str) -> bool:
    return any(name in text for name in ("slc2a1", "glut1", "glut-1", "glucose transporter type 1"))


def _probe_only(description: str) -> bool:
    probe_markers = (
        "rhodamine",
        "calcein",
        "mitoxantrone",
        "daunomycin",
        "doxorubicin",
        "vinblastine",
        "vincristine",
        "jc-1",
    )
    return any(marker in description for marker in probe_markers) and any(
        marker in description for marker in ("inhibition", "inhibitory", "accumulation", "uptake")
    )


def _is_efflux_ratio_description(description: str) -> bool:
    return any(
        marker in description
        for marker in (
            "efflux ratio",
            "ratio of permeability",
            "ratio of apparent permeability",
            "ratio of transport",
            "drug transport ratio",
            "asymmetry efflux ratio",
            "transporter substrate index",
        )
    )


def _supports_positive_role(
    *,
    value: float | None,
    relation: str,
    activity_comment: str,
    minimum_value: float,
) -> bool:
    if any(marker in activity_comment for marker in ("not active", "inactive", "not determined", "bld")):
        return False
    if activity_comment == "active":
        return True
    if value is None or relation in {"<", "<="}:
        return False
    return value > minimum_value if minimum_value == 0.0 else value >= minimum_value


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parent_key(row: Mapping[str, Any]) -> str:
    identity = normalize_molecule_identity(str(row.get("canonical_smiles") or ""))
    return identity.parent_inchi_key if identity.status == "ok" else ""


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit(
    *,
    h_evidence_path: Path,
    h_index_path: Path,
    c_evidence_path: Path,
    input_path: Path,
    min_similarity: float,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    h_rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    h_parents_by_family: dict[str, set[str]] = defaultdict(set)
    h_molecules_by_family_parent: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    h_assays_by_family_parent: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    invalid_h_smiles = Counter()
    for row in load_jsonl(h_evidence_path):
        family = str(row.get("distance_family_id") or "")
        if family not in {*EFFLUX_FAMILIES, *INFLUX_FAMILIES}:
            continue
        h_rows_by_family[family].append(row)
        key = parent_key(row)
        if not key:
            invalid_h_smiles[family] += 1
            continue
        h_parents_by_family[family].add(key)
        h_molecules_by_family_parent[family][key].add(str(row.get("molecule_chembl_id") or ""))
        h_assays_by_family_parent[family][key].add(str(row.get("assay_chembl_id") or ""))

    role_rows_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    role_assays: dict[str, set[str]] = defaultdict(set)
    role_molecules: dict[str, set[str]] = defaultdict(set)
    role_parents: dict[str, set[str]] = defaultdict(set)
    exclusion_reasons = Counter()
    invalid_role_smiles = 0
    for row in load_jsonl(c_evidence_path):
        decision = classify_role_evidence(row)
        if decision.role is None:
            if str(row.get("group_id") or "").startswith(("Tier 3.", "Tier 4.")):
                exclusion_reasons[decision.reason] += 1
            continue
        key = parent_key(row)
        if not key:
            invalid_role_smiles += 1
            continue
        enriched = dict(row)
        enriched["role_gate"] = decision.role
        enriched["role_gate_reason"] = decision.reason
        role_rows_by_parent[key].append(enriched)
        role_assays[decision.role].add(str(row.get("assay_chembl_id") or ""))
        role_molecules[decision.role].add(str(row.get("molecule_chembl_id") or ""))
        role_parents[decision.role].add(key)

    efflux_role_parents = set().union(
        *(parents for role, parents in role_parents.items() if role.startswith("efflux_substrate:"))
    )
    influx_role_parents = set(role_parents.get("influx_substrate:SLC2A1", set()))

    with h_index_path.open("rb") as handle:
        h_index = pickle.load(handle)
    indexed_molecules = {
        str(molecule.get("molecule_chembl_id") or ""): (molecule, fingerprint)
        for molecule, fingerprint in zip(
            h_index.get("molecules") or [], h_index.get("fingerprints") or [], strict=True
        )
    }

    test_rows = list(load_jsonl(input_path))
    family_summaries: dict[str, dict[str, Any]] = {}
    coverage_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for family in (*EFFLUX_FAMILIES, *INFLUX_FAMILIES):
        required_role_parents = efflux_role_parents if family in EFFLUX_FAMILIES else influx_role_parents
        overlap = h_parents_by_family[family] & required_role_parents
        candidate_ids = sorted(
            {
                molecule_id
                for key in overlap
                for molecule_id in h_molecules_by_family_parent[family][key]
                if molecule_id in indexed_molecules
            }
        )
        candidates = [(molecule_id, *indexed_molecules[molecule_id]) for molecule_id in candidate_ids]
        family_coverage = _coverage(
            test_rows,
            candidates,
            family=family,
            min_similarity=min_similarity,
            top_k=top_k,
        )
        coverage_rows.extend(family_coverage)
        n_any = sum(row["n_neighbors"] >= 1 for row in family_coverage)
        n_full = sum(row["n_neighbors"] >= top_k for row in family_coverage)
        max_similarities = [row["max_similarity"] for row in family_coverage if row["max_similarity"] is not None]
        family_summaries[family] = {
            "n_h_rows": len(h_rows_by_family[family]),
            "n_h_assays": len({str(row.get("assay_chembl_id") or "") for row in h_rows_by_family[family]}),
            "n_h_molecules": len({str(row.get("molecule_chembl_id") or "") for row in h_rows_by_family[family]}),
            "n_h_parents": len(h_parents_by_family[family]),
            "n_role_gated_overlap_parents": len(overlap),
            "parent_overlap_fraction": _safe_fraction(len(overlap), len(h_parents_by_family[family])),
            "n_role_gated_h_molecules": len(candidate_ids),
            "n_role_gated_h_assays": len(
                {assay for key in overlap for assay in h_assays_by_family_parent[family][key]}
            ),
            "n_test_queries": len(test_rows),
            "n_queries_with_at_least_1_neighbor": n_any,
            "coverage_at_least_1": _safe_fraction(n_any, len(test_rows)),
            "n_queries_with_top_k_neighbors": n_full,
            "coverage_top_k": _safe_fraction(n_full, len(test_rows)),
            "median_max_similarity": median(max_similarities) if max_similarities else None,
            "invalid_h_smiles_rows": invalid_h_smiles[family],
        }
        for key in sorted(overlap):
            family_role_rows = role_rows_by_parent[key]
            overlap_rows.append(
                {
                    "family": family,
                    "parent_inchi_key": key,
                    "h_molecule_chembl_ids": "|".join(sorted(h_molecules_by_family_parent[family][key])),
                    "h_assay_chembl_ids": "|".join(sorted(h_assays_by_family_parent[family][key])),
                    "role_gates": "|".join(sorted({str(row["role_gate"]) for row in family_role_rows})),
                    "role_molecule_chembl_ids": "|".join(
                        sorted({str(row.get("molecule_chembl_id") or "") for row in family_role_rows})
                    ),
                    "role_assay_chembl_ids": "|".join(
                        sorted({str(row.get("assay_chembl_id") or "") for row in family_role_rows})
                    ),
                }
            )

    role_summary = {
        role: {
            "n_rows": sum(
                1 for rows in role_rows_by_parent.values() for row in rows if row["role_gate"] == role
            ),
            "n_assays": len(role_assays[role]),
            "n_molecules": len(role_molecules[role]),
            "n_parents": len(role_parents[role]),
        }
        for role in sorted(role_parents)
    }
    summary = {
        "audit": "bbb_same_parent_role_gated_overlap.v1",
        "status": "feasibility_only_not_publishable",
        "identity_normalizer": IDENTITY_NORMALIZER_VERSION,
        "source": "ChEMBL 36 only",
        "inputs": {
            "h_evidence": str(h_evidence_path),
            "h_index": str(h_index_path),
            "c_evidence": str(c_evidence_path),
            "test_jsonl": str(input_path),
        },
        "retrieval_policy": {
            "min_similarity": min_similarity,
            "top_k": top_k,
            "neighbor_identity_policy": "operational",
        },
        "strict_role_definition": {
            "efflux": (
                "The activity-row molecule itself has a positive compound efflux ratio (>=2) or positive explicit "
                "drug-transport result for ABCB1, ABCG2, or ABCC2. Single-direction Papp, non-substrate results, "
                "inhibitor, binding, ATPase, and probe-accumulation assays are excluded."
            ),
            "influx": (
                "The activity-row molecule itself has a positive explicit GLUT1/SLC2A1 substrate result by "
                "drug/cellular uptake. Non-substrate results, inhibition of glucose/probe uptake, and binding-only "
                "assays are excluded."
            ),
        },
        "role_evidence": role_summary,
        "families": family_summaries,
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "invalid_role_smiles_rows": invalid_role_smiles,
    }
    return summary, coverage_rows, overlap_rows


def _coverage(
    test_rows: list[dict[str, Any]],
    candidates: list[tuple[str, Mapping[str, Any], Any]],
    *,
    family: str,
    min_similarity: float,
    top_k: int,
) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output = []
    for sample_index, row in enumerate(test_rows):
        smiles = str(row.get("drug") or row.get("canonical_smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            output.append(
                {
                    "sample_index": sample_index,
                    "family": family,
                    "n_neighbors": 0,
                    "max_similarity": None,
                    "neighbor_molecule_ids": "",
                    "status": "invalid_query_smiles",
                }
            )
            continue
        query_fp = generator.GetFingerprint(mol)
        query_identity = normalize_molecule_identity(smiles)
        retained = []
        for molecule_id, candidate, fingerprint in candidates:
            if decide_candidate(query_identity, candidate, "operational").excluded:
                continue
            similarity = float(DataStructs.TanimotoSimilarity(query_fp, fingerprint))
            if similarity + 1e-12 >= min_similarity:
                retained.append((similarity, molecule_id))
        retained.sort(key=lambda item: (-item[0], item[1]))
        selected = retained[:top_k]
        output.append(
            {
                "sample_index": sample_index,
                "family": family,
                "n_neighbors": len(selected),
                "max_similarity": selected[0][0] if selected else None,
                "neighbor_molecule_ids": "|".join(molecule_id for _, molecule_id in selected),
                "status": "ok",
            }
        )
    return output


def _safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def write_outputs(
    out_dir: Path,
    summary: Mapping[str, Any],
    coverage_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_tsv(out_dir / "query_coverage.tsv", coverage_rows)
    _write_tsv(out_dir / "overlap_parents.tsv", overlap_rows)
    (out_dir / "report_zh.md").write_text(_render_report(summary))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# BBB role-gated same-parent overlap audit",
        "",
        "状态：feasibility only；不改变 v3 graph，也不授权正式 E12 LLM run。",
        "",
        "## 严格角色定义",
        "",
        f"- Efflux：{summary['strict_role_definition']['efflux']}",
        f"- Influx：{summary['strict_role_definition']['influx']}",
        "",
        "## ChEMBL role evidence",
        "",
        "| role | assays | molecules | molecular parents |",
        "|---|---:|---:|---:|",
    ]
    for role, values in summary["role_evidence"].items():
        lines.append(
            f"| `{role}` | {values['n_assays']} | {values['n_molecules']} | {values['n_parents']} |"
        )
    lines.extend(
        [
            "",
            "## H family overlap and test coverage",
            "",
            "| family | H parents | overlap parents | overlap fraction | gated H assays | gated H molecules | "
            "test coverage >=1 | test coverage top-3 | median max similarity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family, values in summary["families"].items():
        median_similarity = values["median_max_similarity"]
        median_text = f"{median_similarity:.3f}" if median_similarity is not None else "NA"
        lines.append(
            f"| `{family}` | {values['n_h_parents']} | {values['n_role_gated_overlap_parents']} | "
            f"{values['parent_overlap_fraction']:.3%} | {values['n_role_gated_h_assays']} | "
            f"{values['n_role_gated_h_molecules']} | {values['coverage_at_least_1']:.3%} | "
            f"{values['coverage_top_k']:.3%} | {median_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            "同 parent 共现只解决了 hidden substrate-role 的最低 eligibility 条件，不能证明两类 activity "
            "发生在同一细胞、同一时间或同一暴露条件，也不能证明调控方向会改善或损害 BBB。只有 overlap "
            "和 retrieval coverage 足够时，才值得进一步设计 role-gated evidence bundle；否则 family 应保持 "
            "`requires_query_role` / unavailable。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-evidence", type=Path, default=DEFAULT_H_EVIDENCE)
    parser.add_argument("--h-index", type=Path, default=DEFAULT_H_INDEX)
    parser.add_argument("--c-evidence", type=Path, default=DEFAULT_C_EVIDENCE)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, coverage_rows, overlap_rows = audit(
        h_evidence_path=args.h_evidence,
        h_index_path=args.h_index,
        c_evidence_path=args.c_evidence,
        input_path=args.input_jsonl,
        min_similarity=args.min_similarity,
        top_k=args.top_k,
    )
    write_outputs(args.out_dir, summary, coverage_rows, overlap_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
