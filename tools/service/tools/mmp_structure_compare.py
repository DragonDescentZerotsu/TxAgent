from __future__ import annotations

from dataclasses import replace
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from tools.service.config import ServiceSettings
from tools.service.errors import InvalidInputError
from tools.service.tools.base import BaseTool


RDLogger.DisableLog("rdApp.*")


def _mol_from_smiles(smiles: str, field_name: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidInputError(f"{field_name} must be a non-empty string", code="INVALID_SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidInputError(f"Could not parse {field_name}.", code="INVALID_SMILES")
    return mol


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _round_number(value: object) -> float:
    return round(float(value), 2)


class MmpStructureCompareTool(BaseTool):
    name = "mmp_structure_compare"
    version = "v1"
    description = "Compare two molecules with RDKit similarity/MCS and mmpdb matched-pair fragmentation."
    input_schema = {
        "type": "object",
        "properties": {
            "query_smiles": {"type": "string"},
            "reference_smiles": {"type": "string"},
            "max_mmp_alternatives": {"type": "integer", "default": 5},
            "mcs_timeout_s": {"type": "integer", "default": 5},
        },
        "required": ["query_smiles", "reference_smiles"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "object"},
            "reference": {"type": "object"},
            "similarity": {"type": "object"},
            "matched_pair": {"type": "object"},
            "mcs": {"type": "object"},
            "text": {"type": "string"},
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self._fingerprint_generator = GetMorganGenerator(radius=2, fpSize=2048)
        self._fragment_filter: Any | None = None
        self._fragment_algorithm: Any | None = None
        self._mmpdb_error: str | None = None

    def initialize(self, settings: ServiceSettings) -> None:
        self.initialized = True
        self.initialization_error = None
        try:
            from mmpdblib import config, fragment_algorithm

            options = replace(
                config.DEFAULT_FRAGMENT_OPTIONS,
                max_heavies=100,
                max_rotatable_bonds=40,
                min_heavies_per_const_frag=0,
                min_heavies_total_const_frag=0,
            )
            self._fragment_filter = options.get_fragment_filter()
            self._fragment_algorithm = fragment_algorithm
        except Exception as exc:
            self._fragment_filter = None
            self._fragment_algorithm = None
            self._mmpdb_error = f"{type(exc).__name__}: {exc}"
            self.initialization_error = self._mmpdb_error

    def invoke(self, payload: dict[str, Any], *, return_debug: bool = False) -> dict[str, Any]:
        query_input = str(payload.get("query_smiles") or "").strip()
        reference_input = str(payload.get("reference_smiles") or payload.get("neighbor_smiles") or "").strip()
        max_alternatives = int(payload.get("max_mmp_alternatives", 5))
        mcs_timeout_s = int(payload.get("mcs_timeout_s", 5))

        query_mol = _mol_from_smiles(query_input, "query_smiles")
        reference_mol = _mol_from_smiles(reference_input, "reference_smiles")
        query_smiles = _canonical_smiles(query_mol)
        reference_smiles = _canonical_smiles(reference_mol)

        similarity = self._similarity(query_mol, reference_mol)
        matched_pair = self._matched_pair(query_mol, reference_mol, max_alternatives=max_alternatives)
        mcs = self._mcs(query_mol, reference_mol, timeout_s=mcs_timeout_s)

        warnings: list[str] = []
        if self._mmpdb_error:
            warnings.append(self._mmpdb_error)

        output = {
            "query": {"input_smiles": query_input, "canonical_smiles": query_smiles},
            "reference": {"input_smiles": reference_input, "canonical_smiles": reference_smiles},
            "similarity": similarity,
            "matched_pair": matched_pair,
            "mcs": mcs,
            "text": self._render_text(similarity, matched_pair, mcs),
            "_warnings": warnings,
        }
        if return_debug:
            output["debug"] = {
                "mmpdb_available": self._fragment_filter is not None and self._fragment_algorithm is not None,
            }
        return output

    def _similarity(self, query_mol: Chem.Mol, reference_mol: Chem.Mol) -> dict[str, Any]:
        query_fp = self._fingerprint_generator.GetFingerprint(query_mol)
        reference_fp = self._fingerprint_generator.GetFingerprint(reference_mol)
        tanimoto = _round_number(DataStructs.TanimotoSimilarity(query_fp, reference_fp))
        if tanimoto >= 0.95:
            bucket = "very_close_analog"
        elif tanimoto >= 0.80:
            bucket = "close_analog"
        elif tanimoto >= 0.60:
            bucket = "moderate_analog"
        elif tanimoto >= 0.40:
            bucket = "weak_analog"
        elif tanimoto >= 0.20:
            bucket = "distant_analog"
        else:
            bucket = "very_distant_analog"
        return {
            "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048},
            "tanimoto": tanimoto,
            "similarity_bucket": bucket,
        }

    def _fragmentations(self, mol: Chem.Mol, *, limit: int = 10000) -> list[dict[str, Any]]:
        if self._fragment_filter is None or self._fragment_algorithm is None:
            return []
        errmsg, normalized_mol = self._fragment_filter.normalize(mol)
        if errmsg:
            return []
        filter_error = self._fragment_filter.apply_filters(normalized_mol)
        if filter_error:
            return []

        records = []
        for frag in self._fragment_algorithm.fragment_mol(
            normalized_mol,
            self._fragment_filter,
            num_heavies=normalized_mol.GetNumHeavyAtoms(),
        ):
            records.append(
                {
                    "num_cuts": int(frag.num_cuts),
                    "variable_num_heavies": int(frag.variable_num_heavies),
                    "variable_smiles": frag.variable_smiles,
                    "constant_num_heavies": int(frag.constant_num_heavies),
                    "constant_smiles": frag.constant_smiles,
                    "attachment_order": frag.attachment_order,
                }
            )
            if len(records) >= limit:
                break
        return records

    def _matched_pair(
        self,
        query_mol: Chem.Mol,
        reference_mol: Chem.Mol,
        *,
        max_alternatives: int,
    ) -> dict[str, Any]:
        query_frags = self._fragmentations(query_mol)
        reference_frags = self._fragmentations(reference_mol)
        if not query_frags or not reference_frags:
            return {
                "matched_pair_found": False,
                "reason": "mmpdb fragmentation unavailable or no compatible cuts found",
                "alternatives": [],
            }

        reference_by_constant: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for frag in reference_frags:
            key = (str(frag["constant_smiles"]), int(frag["num_cuts"]))
            reference_by_constant.setdefault(key, []).append(frag)

        alternatives = []
        for query_frag in query_frags:
            key = (str(query_frag["constant_smiles"]), int(query_frag["num_cuts"]))
            for reference_frag in reference_by_constant.get(key, []):
                alternatives.append(self._matched_pair_payload(query_frag, reference_frag))

        alternatives.sort(
            key=lambda item: (
                -int(item["constant_num_heavy_atoms"]),
                int(item["query_variable_num_heavy_atoms"]) + int(item["reference_variable_num_heavy_atoms"]),
                int(item["num_cuts"]),
                item["query_variable_smiles"],
                item["reference_variable_smiles"],
            )
        )
        alternatives = alternatives[: max(1, max_alternatives)]
        if not alternatives:
            return {
                "matched_pair_found": False,
                "reason": "no shared mmpdb constant fragment was found",
                "alternatives": [],
            }

        return {
            "matched_pair_found": True,
            "best_transformation": alternatives[0],
            "alternatives": alternatives,
        }

    def _matched_pair_payload(self, query_frag: dict[str, Any], reference_frag: dict[str, Any]) -> dict[str, Any]:
        return {
            "shared_constant_smiles": query_frag["constant_smiles"],
            "query_variable_smiles": query_frag["variable_smiles"],
            "reference_variable_smiles": reference_frag["variable_smiles"],
            "transformation": f"{reference_frag['variable_smiles']} -> {query_frag['variable_smiles']}",
            "interpretation": "reference variable fragment replaced by query variable fragment on the shared constant scaffold",
            "num_cuts": query_frag["num_cuts"],
            "constant_num_heavy_atoms": query_frag["constant_num_heavies"],
            "query_variable_num_heavy_atoms": query_frag["variable_num_heavies"],
            "reference_variable_num_heavy_atoms": reference_frag["variable_num_heavies"],
        }

    def _mcs(self, query_mol: Chem.Mol, reference_mol: Chem.Mol, *, timeout_s: int) -> dict[str, Any]:
        result = rdFMCS.FindMCS(
            [query_mol, reference_mol],
            timeout=max(1, timeout_s),
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        query_heavies = query_mol.GetNumHeavyAtoms()
        reference_heavies = reference_mol.GetNumHeavyAtoms()
        common_atoms = int(result.numAtoms)
        return {
            "smarts": result.smartsString or None,
            "num_common_atoms": common_atoms,
            "num_common_bonds": int(result.numBonds),
            "query_heavy_atom_coverage": _round_number(common_atoms / query_heavies) if query_heavies else None,
            "reference_heavy_atom_coverage": _round_number(common_atoms / reference_heavies) if reference_heavies else None,
            "timed_out": bool(result.canceled),
        }

    def _render_text(
        self,
        similarity: dict[str, Any],
        matched_pair: dict[str, Any],
        mcs: dict[str, Any],
    ) -> str:
        lines = [
            f"Morgan fingerprint Tanimoto similarity: {similarity['tanimoto']} ({similarity['similarity_bucket']}).",
            (
                "Maximum common substructure coverage: "
                f"query={mcs['query_heavy_atom_coverage']}, reference={mcs['reference_heavy_atom_coverage']}."
            ),
        ]
        if matched_pair["matched_pair_found"]:
            best = matched_pair["best_transformation"]
            lines.append(
                "mmpdb matched-pair transformation: "
                f"{best['transformation']} on shared constant {best['shared_constant_smiles']}."
            )
        else:
            lines.append(f"mmpdb matched-pair transformation: not applicable ({matched_pair['reason']}).")
        return "\n".join(lines)
