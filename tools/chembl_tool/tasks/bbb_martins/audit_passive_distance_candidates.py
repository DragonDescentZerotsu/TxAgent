"""Audit candidate Passive H1 families without mutating the frozen BBB v3 graph."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import decide_candidate


DEFAULT_CHEMBL_SQLITE = Path("tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db")
DEFAULT_BASE_ASSAYS = Path(
    "outputs/chembl_tool/tasks/bbb_martins/assay_screening/v3/bbb_assay_candidates.csv"
)
DEFAULT_INPUT = Path("data/gold_labels/legacy/processed/BBB_Martins/test.jsonl")
DEFAULT_OUT_DIR = Path(
    "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/analysis/v3/passive_candidate_audit"
)

MIN_SIMILARITY = 0.30
TOP_K = 3


@dataclass(frozen=True)
class CandidateSpec:
    family_id: str
    display_name: str
    target_chembl_ids: tuple[str, ...]
    literature_edge: str
    citations: tuple[str, ...]
    declared_level: str = "H1"
    admissible_path: tuple[str, ...] = ()
    mechanism_status: str = "supported"


CANDIDATES = (
    CandidateSpec(
        family_id="mmp2_activity",
        display_name="MMP-2 proteolytic activity",
        target_chembl_ids=("CHEMBL333",),
        literature_edge="MMP-2 activity -> occludin/tight-junction degradation",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/22378877/",
            "https://pubmed.ncbi.nlm.nih.gov/21857898/",
            "https://pubmed.ncbi.nlm.nih.gov/24035828/",
        ),
        admissible_path=("mmp2_activity", "tight_junction_integrity"),
    ),
    CandidateSpec(
        family_id="rock1_activity",
        display_name="ROCK1 kinase activity",
        target_chembl_ids=("CHEMBL3231",),
        literature_edge="ROCK activity -> myosin/cytoskeletal contraction -> junction/barrier disruption",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/28510599/",
            "https://pubmed.ncbi.nlm.nih.gov/26903801/",
        ),
        admissible_path=("rock1_activity", "tight_junction_integrity"),
        mechanism_status="insufficient_bbb_isoform_specific_support",
    ),
    CandidateSpec(
        family_id="rock2_activity",
        display_name="ROCK2 kinase activity",
        target_chembl_ids=("CHEMBL2973",),
        literature_edge="ROCK2 activity -> myosin/cytoskeletal contraction -> junction/barrier disruption",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/28510599/",
            "https://pubmed.ncbi.nlm.nih.gov/26903801/",
        ),
        admissible_path=("rock2_activity", "tight_junction_integrity"),
    ),
    CandidateSpec(
        family_id="mylk_activity",
        display_name="MYLK/MLCK kinase activity",
        target_chembl_ids=("CHEMBL2428",),
        literature_edge="MLCK activity -> MLC phosphorylation -> tight-junction/barrier permeability",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/17419808/",
            "https://pubmed.ncbi.nlm.nih.gov/16638813/",
            "https://pubmed.ncbi.nlm.nih.gov/11003609/",
        ),
        admissible_path=("mylk_activity", "tight_junction_integrity"),
    ),
    CandidateSpec(
        family_id="rhoa_activity",
        display_name="RhoA GTPase activity",
        target_chembl_ids=("CHEMBL6052",),
        literature_edge="RhoA activity -> junction/cytoskeletal disassembly -> barrier permeability",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/20369389/",
            "https://pubmed.ncbi.nlm.nih.gov/30227623/",
        ),
        admissible_path=("rhoa_activity", "tight_junction_integrity"),
    ),
    CandidateSpec(
        family_id="mmp14_activity",
        display_name="MMP-14/MT1-MMP proteolytic activity",
        target_chembl_ids=("CHEMBL3869",),
        literature_edge="MMP-14 activity -> proMMP-2 activation -> tight-junction degradation",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/10998420/",
            "https://pubmed.ncbi.nlm.nih.gov/12630911/",
            "https://pubmed.ncbi.nlm.nih.gov/20571065/",
            "https://pubmed.ncbi.nlm.nih.gov/32838837/",
        ),
        declared_level="H2",
        admissible_path=("mmp14_activity", "mmp2_activity", "tight_junction_integrity"),
        mechanism_status="rejected_h2_shortcut_risk",
    ),
)

TARGET_TO_FAMILY = {
    target_id: spec.family_id for spec in CANDIDATES for target_id in spec.target_chembl_ids
}

FUNCTIONAL_STANDARD_TYPES = {
    "activity",
    "control",
    "ec50",
    "ic50",
    "inh",
    "inhibition",
    "ki",
    "pki",
    "residual activity",
    "residual_activity",
    "% activity remaining",
    "% inhibition",
    "% of control",
    "% of inhibition",
    "% residual activity with skepinone-l",
    "% residual kinase activity",
}

BINDING_ONLY_MARKERS = (
    "binding affinity",
    "binding activity",
    "competitive binding",
    "dissociation constant",
    "kdelect",
    "kinomescan",
    "lanthascreen",
    "surface plasmon resonance",
    "thermal shift",
    "tracer displacement",
    "tracer binding",
)

INDIRECT_PHENOTYPE_MARKERS = (
    "cell viability",
    "cell proliferation",
    "cytotoxicity",
    "tumor growth",
    "cell migration",
    "cell invasion",
)


def classify_activity_row(row: Mapping[str, Any]) -> tuple[str | None, str]:
    family = TARGET_TO_FAMILY.get(str(row.get("target_chembl_id") or ""))
    if family is None:
        return None, "target_not_candidate"
    if int(row.get("confidence_score") or 0) < 8:
        return None, "target_confidence_below_8"
    if str(row.get("relationship_type") or "") not in {"D", "H"}:
        return None, "target_relationship_not_direct_or_homologous"
    description = str(row.get("description") or "").lower()
    standard_type = str(row.get("standard_type") or "").strip().lower()
    if standard_type not in FUNCTIONAL_STANDARD_TYPES:
        return None, "endpoint_not_functional_activity"
    if any(marker in description for marker in BINDING_ONLY_MARKERS):
        return None, "binding_only_assay"
    if any(marker in description for marker in INDIRECT_PHENOTYPE_MARKERS):
        return None, "indirect_phenotype"
    if not str(row.get("canonical_smiles") or "").strip():
        return None, "missing_smiles"
    validity = str(row.get("data_validity_comment") or "").strip().lower()
    if validity:
        return None, "data_validity_flag"
    if int(row.get("potential_duplicate") or 0):
        return None, "potential_duplicate"
    return family, "direct_functional_target_activity"


def audit(
    *,
    chembl_sqlite: Path,
    base_assays_path: Path,
    input_path: Path,
    min_similarity: float,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    base_assay_ids = _load_base_assay_ids(base_assays_path)
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    target_ids = sorted(TARGET_TO_FAMILY)
    conn = sqlite3.connect(str(chembl_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        for row in _iter_candidate_activity_rows(conn, target_ids):
            family, reason = classify_activity_row(row)
            candidate_family = TARGET_TO_FAMILY.get(str(row.get("target_chembl_id") or ""), "")
            if family is None:
                exclusions[candidate_family][reason] += 1
                continue
            row["family_id"] = family
            row["parent_inchi_key"] = _parent_key(row)
            if not row["parent_inchi_key"]:
                exclusions[family]["invalid_parent_identity"] += 1
                continue
            row["base_assay_overlap"] = str(row.get("assay_chembl_id") or "") in base_assay_ids
            if row["base_assay_overlap"]:
                exclusions[family]["base_assay_overlap"] += 1
                continue
            rows_by_family[family].append(row)
    finally:
        conn.close()

    test_rows = list(_load_jsonl(input_path))
    coverage_rows: list[dict[str, Any]] = []
    family_summaries = {}
    candidates_by_family = {}
    for spec in CANDIDATES:
        rows = rows_by_family[spec.family_id]
        molecules = _candidate_molecules(rows)
        candidates_by_family[spec.family_id] = molecules
        family_coverage = _coverage(
            test_rows,
            molecules,
            family_id=spec.family_id,
            min_similarity=min_similarity,
            top_k=top_k,
        )
        coverage_rows.extend(family_coverage)
        family_summaries[spec.family_id] = _family_summary(
            spec,
            rows,
            molecules,
            family_coverage,
            exclusions=dict(exclusions[spec.family_id]),
            top_k=top_k,
        )

    combined_summaries = {}
    for level in ("H1", "H2"):
        passing_family_ids = {
            spec.family_id
            for spec in CANDIDATES
            if spec.declared_level == level
            and spec.mechanism_status == "supported"
            and family_summaries[spec.family_id]["engineering_gate_pass"]
        }
        combined_candidates = _merge_candidates(
            candidates_by_family[family_id] for family_id in passing_family_ids
        )
        combined_coverage = _coverage(
            test_rows,
            combined_candidates,
            family_id=f"passive_{level.lower()}_passing_combined",
            min_similarity=min_similarity,
            top_k=top_k,
        )
        coverage_rows.extend(combined_coverage)
        combined_summary = _coverage_summary(combined_coverage, top_k=top_k)
        combined_summary.update(
            {
                "included_family_ids": sorted(passing_family_ids),
                "n_candidate_molecules": len(combined_candidates),
                "n_candidate_parents": len(
                    {str(candidate.get("parent_inchi_key") or "") for candidate in combined_candidates}
                ),
            }
        )
        combined_summaries[level] = combined_summary

    assay_rows = _assay_manifest_rows(rows_by_family)
    summary = {
        "audit": "bbb_passive_distance_candidate_audit.v1",
        "status": "candidate_feasibility_only_v3_unchanged",
        "source": "ChEMBL 36 only",
        "inputs": {
            "chembl_sqlite": str(chembl_sqlite),
            "base_assays": str(base_assays_path),
            "test_jsonl": str(input_path),
        },
        "retrieval_policy": {
            "min_similarity": min_similarity,
            "top_k": top_k,
            "neighbor_identity_policy": "operational",
        },
        "predeclared_engineering_gate": {
            "minimum_parents": 50,
            "minimum_independent_documents": 3,
            "minimum_test_coverage_at_least_1": 0.10,
            "maximum_largest_document_parent_fraction": 0.80,
        },
        "families": family_summaries,
        "combined_passing_nodes": combined_summaries,
    }
    return summary, coverage_rows, assay_rows


def _iter_candidate_activity_rows(
    conn: sqlite3.Connection,
    target_ids: list[str],
) -> Iterable[dict[str, Any]]:
    target_placeholders = ",".join("?" for _ in target_ids)
    assay_rows = list(
        conn.execute(
            f"""
        SELECT
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_pref_name,
            a.chembl_id AS assay_chembl_id,
            a.assay_id,
            a.description,
            a.assay_type,
            a.confidence_score,
            a.relationship_type,
            a.doc_id,
            d.chembl_id AS document_chembl_id,
            d.pubmed_id,
            d.doi,
            d.doc_type
        FROM target_dictionary td
        JOIN assays a ON a.tid = td.tid
        LEFT JOIN docs d ON d.doc_id = a.doc_id
        WHERE td.chembl_id IN ({target_placeholders})
        """,
            target_ids,
        )
    )
    metadata_by_assay = {int(row["assay_id"]): dict(row) for row in assay_rows}
    assay_ids = sorted(metadata_by_assay)
    for offset in range(0, len(assay_ids), 400):
        chunk = assay_ids[offset : offset + 400]
        placeholders = ",".join("?" for _ in chunk)
        activity_rows = conn.execute(
            f"""
        SELECT
            act.assay_id,
            md.chembl_id AS molecule_chembl_id,
            cs.canonical_smiles,
            act.standard_type,
            act.standard_relation,
            act.standard_value,
            act.standard_units,
            act.pchembl_value,
            act.activity_comment,
            act.data_validity_comment,
            act.potential_duplicate
        FROM activities act INDEXED BY fk_act_assay_id
        JOIN molecule_dictionary md ON md.molregno = act.molregno
        JOIN compound_structures cs ON cs.molregno = act.molregno
        WHERE act.assay_id IN ({placeholders})
          AND COALESCE(act.standard_flag, 0) = 1
        """,
            chunk,
        )
        for raw_row in activity_rows:
            activity = dict(raw_row)
            metadata = dict(metadata_by_assay[int(activity.pop("assay_id"))])
            metadata.update(activity)
            yield metadata


def _load_base_assay_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {
            str(row.get("assay_chembl_id") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("assay_chembl_id") or "").strip()
        }


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _parent_key(row: Mapping[str, Any]) -> str:
    identity = normalize_molecule_identity(str(row.get("canonical_smiles") or ""))
    return identity.parent_inchi_key if identity.status == "ok" else ""


def _candidate_molecules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_molecule: dict[str, dict[str, Any]] = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for row in rows:
        molecule_id = str(row.get("molecule_chembl_id") or "")
        if not molecule_id or molecule_id in by_molecule:
            continue
        mol = Chem.MolFromSmiles(str(row.get("canonical_smiles") or ""))
        if mol is None:
            continue
        by_molecule[molecule_id] = {
            "molecule_chembl_id": molecule_id,
            "canonical_smiles": str(row.get("canonical_smiles") or ""),
            "parent_inchi_key": str(row.get("parent_inchi_key") or ""),
            "fingerprint": generator.GetFingerprint(mol),
        }
    return [by_molecule[key] for key in sorted(by_molecule)]


def _merge_candidates(candidate_groups: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged = {}
    for candidates in candidate_groups:
        for candidate in candidates:
            merged.setdefault(str(candidate["molecule_chembl_id"]), candidate)
    return [merged[key] for key in sorted(merged)]


def _coverage(
    test_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    family_id: str,
    min_similarity: float,
    top_k: int,
) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    candidate_fingerprints = [candidate["fingerprint"] for candidate in candidates]
    output = []
    for sample_index, sample in enumerate(test_rows):
        smiles = str(sample.get("drug") or sample.get("canonical_smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            output.append(
                {
                    "sample_index": sample_index,
                    "family_id": family_id,
                    "n_neighbors": 0,
                    "max_similarity": None,
                    "neighbor_molecule_ids": "",
                    "status": "invalid_query_smiles",
                }
            )
            continue
        query_fp = generator.GetFingerprint(mol)
        query_identity = normalize_molecule_identity(smiles)
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, candidate_fingerprints)
        retained = []
        for candidate, similarity in zip(candidates, similarities, strict=True):
            if similarity + 1e-12 >= min_similarity:
                if decide_candidate(query_identity, candidate, "operational").excluded:
                    continue
                retained.append((float(similarity), str(candidate["molecule_chembl_id"])))
        retained.sort(key=lambda item: (-item[0], item[1]))
        selected = retained[:top_k]
        output.append(
            {
                "sample_index": sample_index,
                "family_id": family_id,
                "n_neighbors": len(selected),
                "max_similarity": selected[0][0] if selected else None,
                "neighbor_molecule_ids": "|".join(item[1] for item in selected),
                "status": "ok",
            }
        )
    return output


def _family_summary(
    spec: CandidateSpec,
    rows: list[dict[str, Any]],
    molecules: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    *,
    exclusions: dict[str, int],
    top_k: int,
) -> dict[str, Any]:
    assay_ids = {str(row.get("assay_chembl_id") or "") for row in rows}
    documents = {str(row.get("document_chembl_id") or "") for row in rows}
    parents = {str(row.get("parent_inchi_key") or "") for row in rows}
    doc_parents: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc_parents[str(row.get("document_chembl_id") or "")].add(str(row.get("parent_inchi_key") or ""))
    largest_doc_fraction = max((len(values) for values in doc_parents.values()), default=0) / max(len(parents), 1)
    coverage = _coverage_summary(coverage_rows, top_k=top_k)
    engineering_pass = (
        len(parents) >= 50
        and len(documents) >= 3
        and coverage["coverage_at_least_1"] >= 0.10
        and largest_doc_fraction <= 0.80
    )
    return {
        "display_name": spec.display_name,
        "target_chembl_ids": list(spec.target_chembl_ids),
        "declared_level": spec.declared_level,
        "parent_c_family_id": "Mechanism.tier_2",
        "measured_node": spec.family_id,
        "admissible_path": list(spec.admissible_path),
        "literature_edge": spec.literature_edge,
        "citations": list(spec.citations),
        "mechanism_status": spec.mechanism_status,
        "same_molecule_status": "pass_same_molecule_with_scope_conditions",
        "n_activity_rows": len(rows),
        "n_assays": len(assay_ids),
        "n_molecules": len(molecules),
        "n_parents": len(parents),
        "n_independent_documents": len(documents),
        "largest_document_parent_fraction": largest_doc_fraction,
        **coverage,
        "engineering_gate_pass": engineering_pass,
        "exclusion_reason_counts": dict(sorted(exclusions.items())),
    }


def _coverage_summary(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    n_queries = len(rows)
    n_any = sum(int(row["n_neighbors"]) >= 1 for row in rows)
    n_full = sum(int(row["n_neighbors"]) >= top_k for row in rows)
    max_similarities = [float(row["max_similarity"]) for row in rows if row["max_similarity"] is not None]
    return {
        "n_test_queries": n_queries,
        "n_queries_with_at_least_1_neighbor": n_any,
        "coverage_at_least_1": n_any / n_queries if n_queries else 0.0,
        "n_queries_with_top_k_neighbors": n_full,
        "coverage_top_k": n_full / n_queries if n_queries else 0.0,
        "median_max_similarity": median(max_similarities) if max_similarities else None,
    }


def _assay_manifest_rows(rows_by_family: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for family, rows in rows_by_family.items():
        by_assay: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_assay[str(row.get("assay_chembl_id") or "")].append(row)
        for assay_id, assay_rows in by_assay.items():
            first = assay_rows[0]
            output.append(
                {
                    "family_id": family,
                    "assay_chembl_id": assay_id,
                    "target_chembl_id": first.get("target_chembl_id"),
                    "description": first.get("description"),
                    "document_chembl_id": first.get("document_chembl_id"),
                    "pubmed_id": first.get("pubmed_id"),
                    "doi": first.get("doi"),
                    "n_activity_rows": len(assay_rows),
                    "n_molecules": len({str(row.get("molecule_chembl_id") or "") for row in assay_rows}),
                    "n_parents": len({str(row.get("parent_inchi_key") or "") for row in assay_rows}),
                }
            )
    return sorted(output, key=lambda row: (str(row["family_id"]), -int(row["n_parents"]), str(row["assay_chembl_id"])))


def write_outputs(
    out_dir: Path,
    summary: Mapping[str, Any],
    coverage_rows: list[dict[str, Any]],
    assay_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_tsv(out_dir / "query_coverage.tsv", coverage_rows)
    _write_tsv(out_dir / "candidate_assays.tsv", assay_rows)
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
        "# BBB Passive distance candidate audit",
        "",
        "状态：candidate feasibility only；不改变 v3 graph，不授权正式 E12 LLM run。",
        "",
        "| family | level | mechanism | assays | parents | docs | >=1 coverage | top-3 coverage | engineering | publishable |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for family, values in summary["families"].items():
        publishable = values["mechanism_status"] == "supported" and values["engineering_gate_pass"]
        lines.append(
            f"| `{family}` | {values['declared_level']} | `{values['mechanism_status']}` | "
            f"{values['n_assays']} | {values['n_parents']} | {values['n_independent_documents']} | "
            f"{values['coverage_at_least_1']:.2%} | {values['coverage_top_k']:.2%} | "
            f"{'pass' if values['engineering_gate_pass'] else 'fail'} | "
            f"{'yes' if publishable else 'no'} |"
        )
    combined_h1 = summary["combined_passing_nodes"]["H1"]
    combined_h2 = summary["combined_passing_nodes"]["H2"]
    lines.extend(
        [
            "",
            "## Combined Passive H1 node",
            "",
            f"- included families: {', '.join(combined_h1['included_family_ids']) or 'none'}",
            f"- candidate parents: {combined_h1['n_candidate_parents']}",
            f"- >=1-neighbor coverage: {combined_h1['coverage_at_least_1']:.2%}",
            f"- top-3 coverage: {combined_h1['coverage_top_k']:.2%}",
            "",
            "## Combined passing Passive H2 node",
            "",
            f"- included families: {', '.join(combined_h2['included_family_ids']) or 'none'}",
            f"- candidate parents: {combined_h2['n_candidate_parents']}",
            f"- >=1-neighbor coverage: {combined_h2['coverage_at_least_1']:.2%}",
            f"- top-3 coverage: {combined_h2['coverage_top_k']:.2%}",
            "",
            "MMP-14 虽有足够 ChEMBL coverage，但因其自身可直接切割 ECM/vascular-barrier substrates，"
            "不能证明到 C 的最短路径必须经过 MMP-2，因此不接受为 H2。",
            "",
            "该 audit 只冻结 candidate feasibility。它不改变 v3 graph，也不授权正式 E12 LLM run。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-sqlite", type=Path, default=DEFAULT_CHEMBL_SQLITE)
    parser.add_argument("--base-assays", type=Path, default=DEFAULT_BASE_ASSAYS)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, coverage_rows, assay_rows = audit(
        chembl_sqlite=args.chembl_sqlite,
        base_assays_path=args.base_assays,
        input_path=args.input_jsonl,
        min_similarity=args.min_similarity,
        top_k=args.top_k,
    )
    write_outputs(args.out_dir, summary, coverage_rows, assay_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
