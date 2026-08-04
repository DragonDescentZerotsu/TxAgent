from __future__ import annotations

import math
import os
import hashlib
import json
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from tools.service.config import ServiceSettings
from tools.service.cache import ToolResultCache
from tools.service.errors import InvalidInputError
from tools.service.tools.base import BaseTool


RDLogger.DisableLog("rdApp.*")

CORE_FEATURE_COLUMNS = [
    "pka__fraction_neutral",
    "pka__logd_estimate",
    "pka__most_acidic_pka",
    "pka__most_basic_pka",
    "pka__num_acidic_sites",
    "pka__num_basic_sites",
    "pka__num_ionizable_sites",
    "rdkit__ExactMolWt",
    "rdkit__FractionCSP3",
    "rdkit__HeavyAtomCount",
    "rdkit__HeavyAtomMolWt",
    "rdkit__LabuteASA",
    "rdkit__MaxAbsPartialCharge",
    "rdkit__MaxPartialCharge",
    "rdkit__MinAbsPartialCharge",
    "rdkit__MinPartialCharge",
    "rdkit__MolLogP",
    "rdkit__MolWt",
    "rdkit__NHOHCount",
    "rdkit__NOCount",
    "rdkit__NumAliphaticCarbocycles",
    "rdkit__NumAliphaticHeterocycles",
    "rdkit__NumAliphaticRings",
    "rdkit__NumAromaticCarbocycles",
    "rdkit__NumAromaticHeterocycles",
    "rdkit__NumAromaticRings",
    "rdkit__NumHAcceptors",
    "rdkit__NumHDonors",
    "rdkit__NumHeteroatoms",
    "rdkit__NumRotatableBonds",
    "rdkit__NumSaturatedCarbocycles",
    "rdkit__NumSaturatedHeterocycles",
    "rdkit__NumSaturatedRings",
    "rdkit__RingCount",
    "rdkit__TPSA",
    "rdkit__qed",
]

PKA_DESCRIPTIONS = {
    "fraction_neutral": ("neutral fraction", "estimated fraction of the molecule that is neutral at the configured pH"),
    "logd_estimate": ("estimated logD", "estimated logD at the configured pH"),
    "most_acidic_pka": ("strongest acidic pKa", "pKa of the strongest acidic site"),
    "most_basic_pka": ("strongest basic pKa", "pKa of the strongest basic site"),
    "num_acidic_sites": ("number of acidic sites", "number of acidic ionizable sites in the molecule"),
    "num_basic_sites": ("number of basic sites", "number of basic ionizable sites in the molecule"),
    "num_ionizable_sites": ("number of ionizable sites", "total number of acidic and basic ionizable sites"),
}

RDKIT_DESCRIPTIONS = {
    "ExactMolWt": ("exact molecular weight", "exact isotopic molecular weight"),
    "FractionCSP3": ("fraction of sp3 carbons", "fraction of carbon atoms that are sp3 hybridized"),
    "HeavyAtomCount": ("heavy-atom count", "number of non-hydrogen atoms"),
    "HeavyAtomMolWt": ("heavy-atom molecular weight", "molecular weight contributed by heavy atoms"),
    "LabuteASA": ("Labute surface area", "Labute approximate surface area"),
    "MaxAbsPartialCharge": ("maximum absolute partial charge", "largest absolute atomic partial charge"),
    "MaxPartialCharge": ("maximum partial charge", "most positive atomic partial charge"),
    "MinAbsPartialCharge": ("minimum absolute partial charge", "smallest absolute atomic partial charge"),
    "MinPartialCharge": ("minimum partial charge", "most negative atomic partial charge"),
    "MolLogP": ("estimated logP", "RDKit-estimated octanol/water partition coefficient"),
    "MolWt": ("molecular weight", "molecular weight"),
    "NHOHCount": ("NH/OH group count", "number of NH or OH groups"),
    "NOCount": ("nitrogen/oxygen atom count", "number of nitrogen and oxygen atoms"),
    "NumAliphaticCarbocycles": ("aliphatic carbocycle count", "number of aliphatic carbocyclic rings"),
    "NumAliphaticHeterocycles": ("aliphatic heterocycle count", "number of aliphatic heterocyclic rings"),
    "NumAliphaticRings": ("aliphatic ring count", "number of aliphatic rings"),
    "NumAromaticCarbocycles": ("aromatic carbocycle count", "number of aromatic carbocyclic rings"),
    "NumAromaticHeterocycles": ("aromatic heterocycle count", "number of aromatic heterocyclic rings"),
    "NumAromaticRings": ("aromatic ring count", "number of aromatic rings"),
    "NumHAcceptors": ("hydrogen-bond acceptor count", "number of hydrogen-bond acceptors"),
    "NumHDonors": ("hydrogen-bond donor count", "number of hydrogen-bond donors"),
    "NumHeteroatoms": ("heteroatom count", "number of heteroatoms, such as N, O, or S"),
    "NumRotatableBonds": ("rotatable-bond count", "number of rotatable bonds"),
    "NumSaturatedCarbocycles": ("saturated carbocycle count", "number of saturated carbocyclic rings"),
    "NumSaturatedHeterocycles": ("saturated heterocycle count", "number of saturated heterocyclic rings"),
    "NumSaturatedRings": ("saturated ring count", "number of saturated rings"),
    "RingCount": ("ring count", "total number of rings"),
    "TPSA": ("topological polar surface area", "topological polar surface area of the molecule"),
    "qed": ("QED drug-likeness", "quantitative estimate of drug-likeness"),
}


def _round_number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, 2)


def _format_number(value: object) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    rendered = f"{numeric:.2f}".rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _format_value(value: object, missing_reason: str | None = None) -> str:
    if missing_reason == "no_acidic_site":
        return "not applicable (no acidic site)"
    if missing_reason == "no_basic_site":
        return "not applicable (no basic site)"
    if missing_reason == "pka_unavailable":
        return "not applicable (pKa predictor unavailable)"
    if value is None:
        return "not applicable"
    try:
        return _format_number(value)
    except (TypeError, ValueError):
        return str(value)


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidInputError("query_smiles must be a non-empty string", code="INVALID_SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidInputError("Could not parse query SMILES.", code="INVALID_SMILES")
    return mol


def _canonicalize_mol(mol: Chem.Mol) -> Chem.Mol:
    try:
        return rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    except Exception:
        return mol


def _inchi_key(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


class MoleculePropertiesTool(BaseTool):
    name = "molecule_properties"
    version = "v1"
    description = "Compute a compact, natural-language-ready RDKit descriptor and MolGpKa/logD profile."
    input_schema = {
        "type": "object",
        "properties": {
            "query_smiles": {"type": "string"},
            "logd_ph": {"type": "number", "default": 7.4},
        },
        "required": ["query_smiles"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "object"},
            "features": {"type": "array"},
            "raw_features": {"type": "object"},
            "functional_groups": {"type": "array"},
            "present_functional_groups": {"type": "array"},
            "text": {"type": "string"},
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self._pka_predictor: Any | None = None
        self._pka_error: str | None = None
        self._fg_detector: Any | None = None
        self._fg_error: str | None = None
        self._logd_ph = 7.4
        self._descriptor_funcs = dict(Descriptors._descList)
        self._result_cache: ToolResultCache | None = None

    def initialize(self, settings: ServiceSettings) -> None:
        self.initialized = True
        self.initialization_error = None
        self._logd_ph = settings.logd_ph
        self._result_cache = ToolResultCache(None, memory_entries=settings.cache_memory_entries)
        self._initialize_functional_group_detector()
        if not settings.enable_molgpka:
            self._pka_error = "MolGpKa initialization disabled by TXAGENT_ENABLE_MOLGPKA"
            return
        try:
            from tools.service.molgpka_predictor import ResidentMolGpKa

            self._pka_predictor = ResidentMolGpKa(
                uncharged=True,
                max_concurrency=settings.batch_workers,
            )
            if settings.prewarm_molgpka:
                self._predict_pka("CC(=O)O")
        except Exception as exc:
            self._pka_predictor = None
            self._pka_error = f"{type(exc).__name__}: {exc}"
            self.initialization_error = self._pka_error

    def close(self) -> None:
        if self._result_cache is not None:
            self._result_cache.close()
            self._result_cache = None
        self._pka_predictor = None

    def _initialize_functional_group_detector(self) -> None:
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/local/tmp/matplotlib")
            from accfg import AccFG

            self._fg_detector = AccFG(print_load_info=False)
        except Exception as exc:
            self._fg_detector = None
            self._fg_error = f"{type(exc).__name__}: {exc}"
            self.initialization_error = self._fg_error

    def invoke(self, payload: dict[str, Any], *, return_debug: bool = False) -> dict[str, Any]:
        input_smiles = str(payload.get("query_smiles") or payload.get("smiles") or "").strip()
        logd_ph = float(payload.get("logd_ph", self._logd_ph))
        mol = _canonicalize_mol(_mol_from_smiles(input_smiles))
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        cache_key = hashlib.sha256(
            json.dumps(
                ["molecule-properties-internal-v1", canonical_smiles, logd_ph, return_debug],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self._result_cache is not None:
            output, _ = self._result_cache.get_or_compute(
                cache_key,
                lambda: self._invoke_canonical(canonical_smiles, logd_ph, return_debug),
            )
        else:
            output = self._invoke_canonical(canonical_smiles, logd_ph, return_debug)
        output["query"]["input_smiles"] = input_smiles
        return output

    def _invoke_canonical(
        self,
        canonical_smiles: str,
        logd_ph: float,
        return_debug: bool,
    ) -> dict[str, Any]:
        mol = _mol_from_smiles(canonical_smiles)

        raw_features = self._compute_rdkit_features(mol)
        warnings: list[str] = []
        pka_features, pka_debug = self._compute_pka_features(canonical_smiles, logd_ph)
        raw_features.update(pka_features)
        if self._pka_error:
            warnings.append(self._pka_error)
        functional_groups = self._functional_groups(canonical_smiles)
        if self._fg_error:
            warnings.append(self._fg_error)

        features = [
            self._feature_payload(feature_name, raw_features.get(feature_name), raw_features)
            for feature_name in CORE_FEATURE_COLUMNS
        ]
        output = {
            "query": {
                "input_smiles": canonical_smiles,
                "canonical_smiles": canonical_smiles,
                "standard_inchi_key": _inchi_key(mol),
            },
            "features": features,
            "raw_features": raw_features,
            "functional_groups": functional_groups,
            "present_functional_groups": functional_groups,
            "text": self._render_text(features, functional_groups),
            "_warnings": warnings,
        }
        if return_debug:
            output["debug"] = {"pka": pka_debug, "functional_groups_available": self._fg_detector is not None}
        return output

    def _compute_rdkit_features(self, mol: Chem.Mol) -> dict[str, float | None]:
        features: dict[str, float | None] = {}
        for feature_name in CORE_FEATURE_COLUMNS:
            if not feature_name.startswith("rdkit__"):
                continue
            descriptor_name = feature_name.split("__", 1)[1]
            descriptor_fn = self._descriptor_funcs.get(descriptor_name)
            if descriptor_fn is None:
                features[feature_name] = None
                continue
            try:
                features[feature_name] = _round_number(descriptor_fn(mol))
            except Exception:
                features[feature_name] = None
        return features

    def _predict_pka(self, smiles: str) -> dict[str, Any]:
        if self._pka_predictor is None:
            raise RuntimeError(self._pka_error or "MolGpKa predictor is unavailable")
        mol = _mol_from_smiles(smiles)
        prediction = self._pka_predictor.predict(mol)
        base_sites = {int(key): _round_number(value) for key, value in prediction.base_sites_1.items()}
        acid_sites = {int(key): _round_number(value) for key, value in prediction.acid_sites_1.items()}
        return {
            "base_sites": base_sites,
            "acid_sites": acid_sites,
            "most_basic_pka": max(base_sites.values()) if base_sites else None,
            "most_acidic_pka": min(acid_sites.values()) if acid_sites else None,
            "num_basic_sites": len(base_sites),
            "num_acidic_sites": len(acid_sites),
            "mapped_smiles": Chem.MolToSmiles(prediction.mol),
        }

    def _compute_pka_features(self, smiles: str, logd_ph: float) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._pka_predictor is None:
            return (
                {
                    "pka__fraction_neutral": None,
                    "pka__logd_estimate": None,
                    "pka__most_acidic_pka": None,
                    "pka__most_basic_pka": None,
                    "pka__num_acidic_sites": None,
                    "pka__num_basic_sites": None,
                    "pka__num_ionizable_sites": None,
                },
                {"available": False, "error": self._pka_error},
            )

        pka = self._predict_pka(smiles)
        mol = _mol_from_smiles(smiles)
        logp = float(Crippen.MolLogP(mol))
        most_basic = pka["most_basic_pka"]
        most_acidic = pka["most_acidic_pka"]

        fraction_neutral_base = 1.0
        if most_basic is not None:
            fraction_neutral_base = 1.0 / (1.0 + 10.0 ** (float(most_basic) - logd_ph))
        fraction_neutral_acid = 1.0
        if most_acidic is not None:
            fraction_neutral_acid = 1.0 / (1.0 + 10.0 ** (logd_ph - float(most_acidic)))
        fraction_neutral = min(1.0, max(1e-12, fraction_neutral_base * fraction_neutral_acid))
        logd = logp + math.log10(fraction_neutral)

        features = {
            "pka__fraction_neutral": _round_number(fraction_neutral),
            "pka__logd_estimate": _round_number(logd),
            "pka__most_acidic_pka": most_acidic,
            "pka__most_basic_pka": most_basic,
            "pka__num_acidic_sites": pka["num_acidic_sites"],
            "pka__num_basic_sites": pka["num_basic_sites"],
            "pka__num_ionizable_sites": pka["num_acidic_sites"] + pka["num_basic_sites"],
        }
        return features, {"available": True, "logd_ph": logd_ph, **pka}

    def _feature_payload(
        self,
        feature_name: str,
        value: object,
        raw_features: dict[str, Any],
    ) -> dict[str, Any]:
        source_family, raw_name = feature_name.split("__", 1)
        descriptions = PKA_DESCRIPTIONS if source_family == "pka" else RDKIT_DESCRIPTIONS
        display_name, description = descriptions.get(raw_name, (raw_name.replace("_", " "), raw_name))
        missing_reason = None
        if source_family == "pka" and self._pka_predictor is None and value is None:
            missing_reason = "pka_unavailable"
        elif feature_name == "pka__most_acidic_pka" and value is None:
            missing_reason = "no_acidic_site"
        elif feature_name == "pka__most_basic_pka" and value is None:
            missing_reason = "no_basic_site"

        return {
            "feature_name": feature_name,
            "display_name": display_name,
            "description": description,
            "source_family": source_family,
            "raw_name": raw_name,
            "feature_value": value,
            "feature_value_text": _format_value(value, missing_reason),
            "feature_value_missing_reason": missing_reason,
        }

    def _functional_groups(self, smiles: str) -> list[dict[str, Any]]:
        if self._fg_detector is None:
            return []
        try:
            matched_fgs = self._fg_detector.run(
                smiles,
                show_atoms=True,
                show_graph=False,
                canonical=True,
            )
        except Exception as exc:
            self._fg_error = f"{type(exc).__name__}: {exc}"
            return []

        functional_groups = []
        for fg_name, atom_matches in sorted(matched_fgs.items(), key=lambda item: str(item[0]).lower()):
            normalized_matches = [list(match) for match in atom_matches]
            functional_groups.append(
                {
                    "name": str(fg_name),
                    "display_name": str(fg_name),
                    "count": len(normalized_matches),
                    "atom_matches": normalized_matches,
                }
            )
        return functional_groups

    def _render_text(self, features: list[dict[str, Any]], functional_groups: list[dict[str, Any]]) -> str:
        lines = []
        for feature in features:
            lines.append(f"{feature['display_name']}: {feature['feature_value_text']}")
        lines.append("")
        if functional_groups:
            lines.append("functional groups:")
            for group in functional_groups:
                lines.append(f"{group['display_name']}: {group['count']}")
        else:
            lines.append("functional groups: none")
        return "\n".join(lines)
