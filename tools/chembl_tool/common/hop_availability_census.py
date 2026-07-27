"""Read-only ChEMBL census for candidate H1/H2 measurement families.

This module deliberately stops before building a distance manifest or running
reasoning.  It answers whether a literature-declared candidate remains outside
the frozen D/C evidence envelope and has enough ChEMBL analog coverage to
justify further task-specific implementation.
"""

from __future__ import annotations

import csv
import json
import pickle
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from tools.chembl_tool.common.experiment_retrieval import SourceExperimentConfig
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import decide_candidate


MIN_SIMILARITY = 0.30
TOP_K = 3


@dataclass(frozen=True)
class CandidateSpec:
    family_id: str
    display_name: str
    declared_level: str
    parent_c_family_id: str
    measured_node: str
    admissible_path: tuple[str, ...]
    selector_kind: str
    selector_values: tuple[str, ...]
    assay_mode: str
    citations: tuple[str, ...]
    mechanism_status: str = "supported"
    same_molecule_status: str = "pass_same_molecule_with_scope_conditions"
    scope_status: str = "task_relevant_with_conditions"
    note: str = ""


@dataclass(frozen=True)
class CensusConfig:
    task_name: str
    chembl_sqlite: Path
    base_index: Path
    input_jsonl: Path
    output_dir: Path
    source_config: SourceExperimentConfig
    candidates: tuple[CandidateSpec, ...]
    normalization_workers: int = 16


FUNCTIONAL_STANDARD_TYPES = {
    "% activity remaining",
    "% inhibition",
    "% of control",
    "% of inhibition",
    "% residual activity",
    "activity",
    "ec50",
    "ic50",
    "inh",
    "inhibition",
    "ki",
    "pki",
    "residual activity",
    "residual_activity",
}

PPI_STANDARD_TYPES = FUNCTIONAL_STANDARD_TYPES | {
    "binding",
    "binding affinity",
    "kd",
    "kon",
    "koff",
}

BINDING_ONLY_MARKERS = (
    "binding affinity",
    "competitive binding",
    "dissociation constant",
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

PPI_MARKERS = (
    "keap1/nrf2",
    "keap1-nrf2",
    "keap1 nrf2",
    "protein-protein interaction",
    "protein protein interaction",
    "interaction between",
    "disruption of",
    "fluorescence polarization",
    "alpha screen",
    "alphascreen",
    "tr-fret",
)


def run_census(
    config: CensusConfig,
    *,
    min_similarity: float = MIN_SIMILARITY,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    base_assay_ids, base_groups = load_base_assay_ids(config.base_index, config.source_config)
    rows_by_family, exclusions = load_candidate_rows(
        config.chembl_sqlite,
        config.candidates,
        base_assay_ids=base_assay_ids,
        normalization_workers=config.normalization_workers,
    )
    test_rows = list(load_jsonl(config.input_jsonl))
    family_summaries: dict[str, Any] = {}
    coverage_rows: list[dict[str, Any]] = []
    assay_rows: list[dict[str, Any]] = []
    candidates_by_family: dict[str, list[dict[str, Any]]] = {}

    for spec in config.candidates:
        rows = rows_by_family[spec.family_id]
        molecules = candidate_molecules(rows)
        candidates_by_family[spec.family_id] = molecules
        coverage_by_policy = {}
        for policy in ("operational", "parent_disjoint"):
            rows_for_policy = coverage(
                test_rows,
                molecules,
                family_id=spec.family_id,
                min_similarity=min_similarity,
                top_k=top_k,
                neighbor_identity_policy=policy,
            )
            coverage_rows.extend(rows_for_policy)
            coverage_by_policy[policy] = coverage_summary(rows_for_policy, top_k=top_k)

        family_summaries[spec.family_id] = family_summary(
            spec,
            rows,
            molecules,
            coverage_by_policy,
            exclusions=dict(exclusions[spec.family_id]),
        )
        assay_rows.extend(assay_manifest_rows(spec.family_id, rows))

    combined_by_level = {}
    for level in ("H1", "H2"):
        included = {
            spec.family_id
            for spec in config.candidates
            if spec.declared_level == level
            and family_summaries[spec.family_id]["publishable_candidate"]
        }
        merged = merge_candidates(candidates_by_family[family_id] for family_id in included)
        policy_summaries = {}
        for policy in ("operational", "parent_disjoint"):
            combined_rows = coverage(
                test_rows,
                merged,
                family_id=f"{config.task_name}_{level.lower()}_accepted_union",
                min_similarity=min_similarity,
                top_k=top_k,
                neighbor_identity_policy=policy,
            )
            coverage_rows.extend(combined_rows)
            policy_summaries[policy] = coverage_summary(combined_rows, top_k=top_k)
        combined_by_level[level] = {
            "included_family_ids": sorted(included),
            "n_candidate_molecules": len(merged),
            "n_candidate_parents": len(
                {str(candidate.get("parent_inchi_key") or "") for candidate in merged}
            ),
            "coverage": policy_summaries,
        }

    summary = {
        "audit": "task_hop_availability_census.v1",
        "task": config.task_name,
        "status": "census_only_existing_experiments_unchanged",
        "source": "ChEMBL 36 only",
        "inputs": {
            "chembl_sqlite": str(config.chembl_sqlite),
            "base_index": str(config.base_index),
            "test_jsonl": str(config.input_jsonl),
        },
        "base_envelope": {
            "n_source_groups": len(base_groups),
            "source_groups": sorted(base_groups),
            "n_assays": len(base_assay_ids),
        },
        "retrieval_policy": {
            "min_similarity": min_similarity,
            "top_k": top_k,
            "reported_identity_policies": ["operational", "parent_disjoint"],
        },
        "predeclared_engineering_gate": {
            "minimum_parents": 50,
            "minimum_independent_documents": 3,
            "minimum_parent_disjoint_test_coverage_at_least_1": 0.10,
            "maximum_largest_document_parent_fraction": 0.80,
        },
        "families": family_summaries,
        "combined_accepted_nodes": combined_by_level,
    }
    write_outputs(config.output_dir, summary, coverage_rows, assay_rows)
    return summary


def load_base_assay_ids(
    index_path: Path,
    source_config: SourceExperimentConfig,
) -> tuple[set[str], set[str]]:
    with index_path.open("rb") as handle:
        index = pickle.load(handle)
    available_groups = set(index["group_to_molecule_indices"])
    selected_groups = {
        source_group
        for spec in source_config.mechanism_groups
        for source_group in spec.resolve(available_groups)
    }
    assay_ids = set()
    for evidence_by_group in index["evidence_by_molecule_group"].values():
        for group_id, rows in evidence_by_group.items():
            if group_id not in selected_groups:
                continue
            assay_ids.update(
                str(row.get("assay_chembl_id") or "")
                for row in rows
                if str(row.get("assay_chembl_id") or "")
            )
    return assay_ids, selected_groups


def load_candidate_rows(
    chembl_sqlite: Path,
    specs: Sequence[CandidateSpec],
    *,
    base_assay_ids: set[str],
    normalization_workers: int = 1,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    accepted_before_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    conn = sqlite3.connect(str(chembl_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        for spec in specs:
            for row in iter_spec_rows(conn, spec):
                reason = classify_row(row, spec)
                if reason != "accepted":
                    exclusions[spec.family_id][reason] += 1
                    continue
                assay_id = str(row.get("assay_chembl_id") or "")
                if assay_id in base_assay_ids:
                    exclusions[spec.family_id]["base_dc_assay_overlap"] += 1
                    continue
                row["family_id"] = spec.family_id
                accepted_before_parent[spec.family_id].append(row)
    finally:
        conn.close()

    unique_smiles = sorted(
        {
            str(row.get("canonical_smiles") or "")
            for rows in accepted_before_parent.values()
            for row in rows
        }
    )
    if normalization_workers > 1 and unique_smiles:
        with ProcessPoolExecutor(max_workers=normalization_workers) as executor:
            parent_keys = executor.map(molecular_parent_key, unique_smiles, chunksize=64)
            parent_by_smiles = dict(zip(unique_smiles, parent_keys, strict=True))
    else:
        parent_by_smiles = {smiles: molecular_parent_key(smiles) for smiles in unique_smiles}

    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family_id, rows in accepted_before_parent.items():
        for row in rows:
            parent_key = parent_by_smiles.get(str(row.get("canonical_smiles") or ""), "")
            if not parent_key:
                exclusions[family_id]["invalid_parent_identity"] += 1
                continue
            row["parent_inchi_key"] = parent_key
            rows_by_family[family_id].append(row)
    return rows_by_family, exclusions


def iter_spec_rows(conn: sqlite3.Connection, spec: CandidateSpec) -> Iterable[dict[str, Any]]:
    if spec.selector_kind == "target":
        yield from iter_target_rows(conn, spec.selector_values)
        return
    if spec.selector_kind == "standard_type":
        yield from iter_standard_type_rows(conn, spec.selector_values)
        return
    raise ValueError(f"Unsupported selector_kind: {spec.selector_kind}")


def iter_target_rows(
    conn: sqlite3.Connection,
    target_ids: Sequence[str],
) -> Iterable[dict[str, Any]]:
    placeholders = ",".join("?" for _ in target_ids)
    assays = list(
        conn.execute(
            f"""
            SELECT td.chembl_id AS target_chembl_id, td.pref_name AS target_pref_name,
                   a.chembl_id AS assay_chembl_id, a.assay_id, a.description,
                   a.assay_type, a.confidence_score, a.relationship_type,
                   d.chembl_id AS document_chembl_id, d.pubmed_id, d.doi
            FROM target_dictionary td
            JOIN assays a ON a.tid = td.tid
            LEFT JOIN docs d ON d.doc_id = a.doc_id
            WHERE td.chembl_id IN ({placeholders})
            """,
            tuple(target_ids),
        )
    )
    metadata = {int(row["assay_id"]): dict(row) for row in assays}
    yield from iter_activity_rows_for_assays(conn, metadata)


def iter_standard_type_rows(
    conn: sqlite3.Connection,
    standard_types: Sequence[str],
) -> Iterable[dict[str, Any]]:
    placeholders = ",".join("?" for _ in standard_types)
    query = f"""
        SELECT td.chembl_id AS target_chembl_id, td.pref_name AS target_pref_name,
               a.chembl_id AS assay_chembl_id, a.assay_id, a.description,
               a.assay_type, a.confidence_score, a.relationship_type,
               d.chembl_id AS document_chembl_id, d.pubmed_id, d.doi,
               md.chembl_id AS molecule_chembl_id, cs.canonical_smiles,
               act.standard_type, act.standard_relation, act.standard_value,
               act.standard_units, act.pchembl_value, act.activity_comment,
               act.data_validity_comment, act.potential_duplicate
        FROM activities act INDEXED BY idx_act_std_type
        JOIN assays a ON a.assay_id = act.assay_id
        LEFT JOIN target_dictionary td ON td.tid = a.tid
        LEFT JOIN docs d ON d.doc_id = a.doc_id
        JOIN molecule_dictionary md ON md.molregno = act.molregno
        JOIN compound_structures cs ON cs.molregno = act.molregno
        WHERE act.standard_type IN ({placeholders})
    """
    for row in conn.execute(query, tuple(standard_types)):
        yield dict(row)


def iter_activity_rows_for_assays(
    conn: sqlite3.Connection,
    metadata_by_assay: Mapping[int, Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    assay_ids = sorted(metadata_by_assay)
    for offset in range(0, len(assay_ids), 400):
        chunk = assay_ids[offset : offset + 400]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT act.assay_id, md.chembl_id AS molecule_chembl_id,
                   cs.canonical_smiles, act.standard_type, act.standard_relation,
                   act.standard_value, act.standard_units, act.pchembl_value,
                   act.activity_comment, act.data_validity_comment,
                   act.potential_duplicate
            FROM activities act INDEXED BY fk_act_assay_id
            JOIN molecule_dictionary md ON md.molregno = act.molregno
            JOIN compound_structures cs ON cs.molregno = act.molregno
            WHERE act.assay_id IN ({placeholders})
              AND COALESCE(act.standard_flag, 0) = 1
            """,
            chunk,
        )
        for raw in rows:
            activity = dict(raw)
            assay_id = int(activity.pop("assay_id"))
            merged = dict(metadata_by_assay[assay_id])
            merged.update(activity)
            yield merged


def classify_row(row: Mapping[str, Any], spec: CandidateSpec) -> str:
    if not str(row.get("canonical_smiles") or "").strip():
        return "missing_smiles"
    if str(row.get("data_validity_comment") or "").strip():
        return "data_validity_flag"
    if int(row.get("potential_duplicate") or 0):
        return "potential_duplicate"
    if spec.assay_mode == "property_measurement":
        return "accepted"

    if int(row.get("confidence_score") or 0) < 8:
        return "target_confidence_below_8"
    if str(row.get("relationship_type") or "") not in {"D", "H"}:
        return "target_relationship_not_direct_or_homologous"
    description = str(row.get("description") or "").lower()
    standard_type = str(row.get("standard_type") or "").strip().lower()
    if spec.assay_mode == "direct_ppi":
        if standard_type not in PPI_STANDARD_TYPES:
            return "endpoint_not_direct_ppi"
        target_text = f"{row.get('target_pref_name') or ''} {description}".lower()
        if not any(marker in target_text for marker in PPI_MARKERS):
            return "description_not_direct_ppi"
        if any(marker in description for marker in INDIRECT_PHENOTYPE_MARKERS):
            return "indirect_phenotype"
        return "accepted"
    if spec.assay_mode == "functional_target":
        if standard_type not in FUNCTIONAL_STANDARD_TYPES:
            return "endpoint_not_functional_activity"
        if any(marker in description for marker in BINDING_ONLY_MARKERS):
            return "binding_only_assay"
        if any(marker in description for marker in INDIRECT_PHENOTYPE_MARKERS):
            return "indirect_phenotype"
        return "accepted"
    raise ValueError(f"Unsupported assay_mode: {spec.assay_mode}")


def molecular_parent_key(smiles: str) -> str:
    identity = normalize_molecule_identity(smiles)
    return identity.parent_inchi_key if identity.status == "ok" else ""


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def candidate_molecules(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output = {}
    for row in rows:
        molecule_id = str(row.get("molecule_chembl_id") or "")
        if not molecule_id or molecule_id in output:
            continue
        smiles = str(row.get("canonical_smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        output[molecule_id] = {
            "molecule_chembl_id": molecule_id,
            "canonical_smiles": smiles,
            "parent_inchi_key": str(row.get("parent_inchi_key") or ""),
            "fingerprint": generator.GetFingerprint(mol),
        }
    return [output[key] for key in sorted(output)]


def merge_candidates(groups: Iterable[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged = {}
    for group in groups:
        for candidate in group:
            merged.setdefault(str(candidate["molecule_chembl_id"]), candidate)
    return [merged[key] for key in sorted(merged)]


def coverage(
    test_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    family_id: str,
    min_similarity: float,
    top_k: int,
    neighbor_identity_policy: str,
) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    candidate_fps = [candidate["fingerprint"] for candidate in candidates]
    output = []
    for sample_index, sample in enumerate(test_rows):
        smiles = str(sample.get("drug") or sample.get("canonical_smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            output.append(
                {
                    "sample_index": sample_index,
                    "family_id": family_id,
                    "neighbor_identity_policy": neighbor_identity_policy,
                    "n_neighbors": 0,
                    "max_similarity": None,
                    "neighbor_molecule_ids": "",
                    "status": "invalid_query_smiles",
                }
            )
            continue
        query_fp = generator.GetFingerprint(mol)
        query_identity = normalize_molecule_identity(smiles)
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, candidate_fps)
        retained = []
        for candidate, similarity in zip(candidates, similarities, strict=True):
            if similarity + 1e-12 < min_similarity:
                continue
            if decide_candidate(query_identity, candidate, neighbor_identity_policy).excluded:
                continue
            retained.append((float(similarity), str(candidate["molecule_chembl_id"])))
        retained.sort(key=lambda item: (-item[0], item[1]))
        selected = retained[:top_k]
        output.append(
            {
                "sample_index": sample_index,
                "family_id": family_id,
                "neighbor_identity_policy": neighbor_identity_policy,
                "n_neighbors": len(selected),
                "max_similarity": selected[0][0] if selected else None,
                "neighbor_molecule_ids": "|".join(item[1] for item in selected),
                "status": "ok",
            }
        )
    return output


def coverage_summary(rows: Sequence[Mapping[str, Any]], *, top_k: int) -> dict[str, Any]:
    n_queries = len(rows)
    n_any = sum(int(row["n_neighbors"]) >= 1 for row in rows)
    n_full = sum(int(row["n_neighbors"]) >= top_k for row in rows)
    similarities = [
        float(row["max_similarity"]) for row in rows if row.get("max_similarity") is not None
    ]
    return {
        "n_test_queries": n_queries,
        "n_queries_with_at_least_1_neighbor": n_any,
        "coverage_at_least_1": n_any / n_queries if n_queries else 0.0,
        "n_queries_with_top_k_neighbors": n_full,
        "coverage_top_k": n_full / n_queries if n_queries else 0.0,
        "median_max_similarity": median(similarities) if similarities else None,
    }


def family_summary(
    spec: CandidateSpec,
    rows: Sequence[Mapping[str, Any]],
    molecules: Sequence[Mapping[str, Any]],
    coverage_by_policy: Mapping[str, Mapping[str, Any]],
    *,
    exclusions: Mapping[str, int],
) -> dict[str, Any]:
    assay_ids = {str(row.get("assay_chembl_id") or "") for row in rows}
    documents = {str(row.get("document_chembl_id") or "") for row in rows}
    parents = {str(row.get("parent_inchi_key") or "") for row in rows}
    doc_parents: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc_parents[str(row.get("document_chembl_id") or "")].add(
            str(row.get("parent_inchi_key") or "")
        )
    largest_doc_fraction = max((len(values) for values in doc_parents.values()), default=0) / max(
        len(parents), 1
    )
    parent_disjoint_coverage = coverage_by_policy["parent_disjoint"]["coverage_at_least_1"]
    engineering_gate_pass = (
        len(parents) >= 50
        and len(documents) >= 3
        and parent_disjoint_coverage >= 0.10
        and largest_doc_fraction <= 0.80
    )
    publishable = (
        spec.mechanism_status == "supported"
        and spec.same_molecule_status.startswith("pass_same_molecule")
        and engineering_gate_pass
    )
    return {
        "display_name": spec.display_name,
        "declared_level": spec.declared_level,
        "parent_c_family_id": spec.parent_c_family_id,
        "measured_node": spec.measured_node,
        "admissible_path": list(spec.admissible_path),
        "selector_kind": spec.selector_kind,
        "selector_values": list(spec.selector_values),
        "assay_mode": spec.assay_mode,
        "citations": list(spec.citations),
        "mechanism_status": spec.mechanism_status,
        "same_molecule_status": spec.same_molecule_status,
        "scope_status": spec.scope_status,
        "note": spec.note,
        "n_activity_rows": len(rows),
        "n_assays": len(assay_ids),
        "n_molecules": len(molecules),
        "n_parents": len(parents),
        "n_independent_documents": len(documents),
        "largest_document_parent_fraction": largest_doc_fraction,
        "coverage": dict(coverage_by_policy),
        "engineering_gate_pass": engineering_gate_pass,
        "publishable_candidate": publishable,
        "exclusion_reason_counts": dict(sorted(exclusions.items())),
    }


def assay_manifest_rows(family_id: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_assay: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_assay[str(row.get("assay_chembl_id") or "")].append(row)
    output = []
    for assay_id, assay_rows in by_assay.items():
        first = assay_rows[0]
        output.append(
            {
                "family_id": family_id,
                "assay_chembl_id": assay_id,
                "target_chembl_id": first.get("target_chembl_id"),
                "description": first.get("description"),
                "document_chembl_id": first.get("document_chembl_id"),
                "pubmed_id": first.get("pubmed_id"),
                "doi": first.get("doi"),
                "n_activity_rows": len(assay_rows),
                "n_molecules": len(
                    {str(row.get("molecule_chembl_id") or "") for row in assay_rows}
                ),
                "n_parents": len(
                    {str(row.get("parent_inchi_key") or "") for row in assay_rows}
                ),
            }
        )
    return sorted(output, key=lambda row: (-int(row["n_parents"]), str(row["assay_chembl_id"])))


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    coverage_rows: Sequence[Mapping[str, Any]],
    assay_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_tsv(output_dir / "query_coverage.tsv", coverage_rows)
    write_tsv(output_dir / "candidate_assays.tsv", assay_rows)
    (output_dir / "report_zh.md").write_text(render_report(summary))


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['task']} strict-hop availability census",
        "",
        "状态：ChEMBL-only census；不改变现有 D/C、paper matrix、index 或 reasoning 结果。",
        "",
        "| family | level | assays | parents | parent-disjoint >=1 | top-3 | mechanism | engineering | accepted |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for family_id, values in summary["families"].items():
        coverage_values = values["coverage"]["parent_disjoint"]
        lines.append(
            f"| `{family_id}` | {values['declared_level']} | {values['n_assays']} | "
            f"{values['n_parents']} | {coverage_values['coverage_at_least_1']:.2%} | "
            f"{coverage_values['coverage_top_k']:.2%} | `{values['mechanism_status']}` | "
            f"{'pass' if values['engineering_gate_pass'] else 'fail'} | "
            f"{'yes' if values['publishable_candidate'] else 'no'} |"
        )
    for level in ("H1", "H2"):
        values = summary["combined_accepted_nodes"][level]
        parent_disjoint = values["coverage"]["parent_disjoint"]
        lines.extend(
            [
                "",
                f"## Accepted {level} union",
                "",
                f"- families: {', '.join(values['included_family_ids']) or 'none'}",
                f"- parents: {values['n_candidate_parents']}",
                f"- parent-disjoint >=1 coverage: {parent_disjoint['coverage_at_least_1']:.2%}",
                f"- parent-disjoint top-3 coverage: {parent_disjoint['coverage_top_k']:.2%}",
            ]
        )
    lines.extend(
        [
            "",
            "该报告只判断是否值得进入下一步 manifest/index 实现；`accepted=no` 的 family 不得因数据量大而自动进入 H1/H2。",
            "",
        ]
    )
    return "\n".join(lines)
