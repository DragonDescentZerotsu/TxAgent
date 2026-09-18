"""Experimental meaningful-CNS-access gold adapter for Starling BBB records.

This lineage treats passive permeability, transporter assays, and computational
predictions as mechanism evidence rather than gold outcomes.  A source row is
eligible only when it reports an experimentally observed brain/CSF outcome.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from data.processing.gold_labels.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    has_reported_text,
    rejected,
)

from .starling_benchmark import (
    NEGATIVE_LABELS,
    POSITIVE_LABELS,
    SOURCE_DATASET,
    SOURCE_REVISION,
    _normalized_label,
)


CONTRACT_VERSION = "bbb_experimental_meaningful_cns_access_gold.v2"
_MANUAL_EXCLUSIONS_PATH = Path(__file__).with_name(
    "experimental_meaningful_cns_access_exclusions.json"
)
_MANUAL_EXCLUSION_SPEC = json.loads(
    _MANUAL_EXCLUSIONS_PATH.read_text(encoding="utf-8")
)
if _MANUAL_EXCLUSION_SPEC["source_revision"] != SOURCE_REVISION:
    raise RuntimeError("manual BBB exclusions are bound to a different source revision")
MANUAL_SOURCE_EXCLUSIONS = {
    int(index): reason
    for index, reason in _MANUAL_EXCLUSION_SPEC["exclusions"].items()
}


@dataclass(frozen=True)
class ScopeDecision:
    """Outcome family and experimental-basis provenance for one source row."""

    endpoint_family: str | None
    basis: str | None
    rejection_reason: str | None = None


_PREDICTION_PATTERN = re.compile(
    r"\bin[ -]?silico\b|comput(?:ational|ed)|predict(?:ed|ion|ive)|"
    r"\bqsar\b|qikprop|swiss\s*adme|admet(?:sar)?|pkcsm|boiled[ -]?egg|"
    r"machine learning|neural network|clark[’'s ]+equation|\bpbpk\b|"
    r"model(?:ing|led)\s+(?:prediction|simulation)|simulat(?:ed|ion)",
    re.IGNORECASE,
)
_PREDICTION_MODEL_PATTERN = re.compile(
    r"\bcalculation\b|\bcalculated\b|\bestimat(?:e|ed|ion)\b|"
    r"\bformula\b|\bequation\b",
    re.IGNORECASE,
)
_CALCULATED_BBB_METRIC_PATTERN = re.compile(
    r"(?:calculat(?:e|ed|ion)|estimat(?:e|ed|ion)).{0,20}"
    r"(?:log\s*bb|bbb|blood.brain barrier)|"
    r"(?:log\s*bb|bbb|blood.brain barrier).{0,20}"
    r"(?:calculat(?:e|ed|ion)|estimat(?:e|ed|ion))",
    re.IGNORECASE,
)
_IN_VITRO_PATTERN = re.compile(
    r"\bin[ -]?vitro\b|\bpampa(?:-bbb)?\b|\bmdck(?:-mdr1)?\b|caco-?2|"
    r"transwell|monolayer|cell culture|hcmec|bcec|artificial membrane",
    re.IGNORECASE,
)
_NON_SYSTEMIC_ROUTE_PATTERN = re.compile(
    r"intrathecal|intracisternal|intracerebroventricular|intracerebral|"
    r"intraparenchymal|"
    r"direct(?:ly)? inject(?:ed|ion)? into (?:the )?(?:brain|csf)",
    re.IGNORECASE,
)
_CONDITIONABLE_ALTERED_CONTEXT_PATTERN = re.compile(
    r"focused ultrasound|hyperosmolar|osmotic (?:bbb )?(?:opening|disruption)|"
    r"breach(?:ing|ed)? (?:of )?(?:the )?blood.brain barrier|"
    r"brain tumor|intracranial tumou?r|glioma|meningitis|"
    r"ischemi(?:a|c)|irradiat(?:ed|ion)|radiation-induced|"
    r"brain lesion|intraventricular graft|injured (?:bbb|barrier)|"
    r"damaged (?:bbb|barrier)|nanoparticle|liposom(?:e|al)",
    re.IGNORECASE,
)
_INDIRECT_INFERENCE_PATTERN = re.compile(
    r"(?:probably|possibly|likely|presumably|perhaps) because.{0,80}"
    r"(?:bbb|blood.brain barrier)|"
    r"(?:absence|lack|loss) of (?:efficacy|effect|activity).{0,80}"
    r"(?:attributed to|explained by|due to).{0,40}"
    r"(?:bbb|blood.brain barrier)|"
    r"(?:no|lack|absence of).{0,50}(?:effect|inhibition|response|change)"
    r".{0,100}(?:cannot|does not|poor).{0,30}(?:cross|passage|bbb)|"
    r"(?:cross|penetrat|pass).{0,50}(?:demonstrated|inferred|supported)"
    r".{0,20}(?:by|from).{0,80}(?:efficacy|tumou?r growth|activity|ed50)|"
    r"(?:cross|penetrat|pass).{0,80}(?:based on|inferred from|demonstrated by)"
    r".{0,80}(?:in vivo )?activity",
    re.IGNORECASE,
)
_EX_VIVO_ONLY_PATTERN = re.compile(
    r"ex[ -]?vivo (?:human )?brain|postmortem (?:human )?brain|"
    r"autoradiography.{0,40}(?:tissue section|postmortem)|"
    r"(?:tissue section|postmortem).{0,40}autoradiography",
    re.IGNORECASE,
)
_UNSUPPORTED_METAL_SMILES_PATTERN = re.compile(
    r"\[(?:Al|As|Au|Cd|Co|Cr|Cu|Fe|Gd|Hg|Ir|Mn|Mo|Ni|Os|Pb|Pd|Pt|Rh|Ru|Sb|Sn|Ti|V|W|Zn)(?:[^A-Za-z]|\])"
)
_EXPERIMENTAL_MODEL_PATTERN = re.compile(
    r"\bin[ -]?vivo\b|\bpet\b|positron emission|microdialysis|"
    r"csf (?:measurement|sampling|analysis)|lumbar puncture|brain perfusion|"
    r"intravenous|oral administration|systemic administration|clinical stud|"
    r"pharmacokinetic stud|biodistribution|tissue distribution|autoradiograph|"
    r"lc-ms|hplc|animal (?:model|stud)|quantitative whole-body",
    re.IGNORECASE,
)
_MEASUREMENT_ACTION_PATTERN = re.compile(
    r"measur(?:ed|ement)|detect(?:ed|ion)|quantif(?:ied|ication)|determined|"
    r"observed|assessed|demonstrat(?:ed|ion)|show(?:ed|n)|sampling|sampled|"
    r"radioactiv|imaging|"
    r"concentration(?:s)? (?:was|were)|postmortem",
    re.IGNORECASE,
)
_SYSTEMIC_ROUTE_PATTERN = re.compile(
    r"intravenous|i\.v\.|oral(?:ly)?|systemic|intraperitoneal|subcutaneous|"
    r"after (?:administration|injection|dosing)",
    re.IGNORECASE,
)

_METRIC_ENDPOINT_PATTERNS = (
    (
        "brain_unbound",
        re.compile(
            r"kp\s*[,._-]?\s*uu(?:\s*[,._-]?\s*brain)?|unbound brain|"
            r"c[_ ]?u[,/_ ]?b|brain extracellular|brain ecf",
            re.IGNORECASE,
        ),
    ),
    (
        "brain_systemic_ratio",
        re.compile(
            r"brain\s*(?:/|-to-|to )\s*(?:plasma|blood|serum)|"
            r"(?:plasma|blood|serum)\s*(?:/|-to-|to )\s*brain|"
            r"brain.{0,20}(?:plasma|blood|serum).{0,10}ratio|"
            r"\bb[/ ]p\s*ratio\b|\blog\s*bb\b|\bkp\s*[,._-]?\s*brain\b",
            re.IGNORECASE,
        ),
    ),
    (
        "brain_tissue",
        re.compile(
            r"brain (?:tissue )?(?:concentration|level|uptake|accumulation|"
            r"distribution|penetration)|cerebr(?:um|al|ellum).{0,20}"
            r"(?:concentration|level|uptake|radioactiv)|brain.{0,20}radioactiv|"
            r"\bbrain uptake index\b|\bbui\b",
            re.IGNORECASE,
        ),
    ),
    (
        "csf",
        re.compile(
            r"(?:\bcsf\b|cerebrospinal fluid).{0,25}"
            r"(?:concentration|level|ratio|penetration|exposure|uptake|auc)|"
            r"(?:concentration|level|ratio|penetration|exposure|uptake|auc)"
            r".{0,25}(?:\bcsf\b|cerebrospinal fluid)",
            re.IGNORECASE,
        ),
    ),
    (
        "pet_or_autoradiography",
        re.compile(
            r"\bpet\b|positron emission|autoradiograph|\bsuv\b|%\s*id/g",
            re.IGNORECASE,
        ),
    ),
)

_TEXT_ENDPOINT_PATTERNS = (
    (
        "brain_unbound",
        re.compile(
            r"unbound brain|brain extracellular fluid|brain ecf", re.IGNORECASE
        ),
    ),
    (
        "brain_systemic_ratio",
        re.compile(
            r"brain\s*(?:/|-to-|to )\s*(?:plasma|blood|serum)|"
            r"(?:plasma|blood|serum)\s*(?:/|-to-|to )\s*brain|"
            r"\blog\s*bb\b|\bkp\s*[,._-]?\s*brain\b",
            re.IGNORECASE,
        ),
    ),
    (
        "brain_tissue",
        re.compile(
            r"(?:detected|measured|quantified|observed|radioactivity|"
            r"concentration|level|uptake|accumulation|distribution).{0,30}"
            r"(?:in|within|of) (?:the )?brain|brain (?:tissue )?"
            r"(?:concentration|level|uptake|accumulation|distribution)|"
            r"brain.{0,25}(?:detected|measured|radioactivity)",
            re.IGNORECASE,
        ),
    ),
    (
        "csf",
        re.compile(
            r"(?:csf|cerebrospinal fluid).{0,25}"
            r"(?:concentration|level|ratio|penetration|exposure|detected|measured)|"
            r"(?:concentration|level|detected|measured).{0,25}"
            r"(?:csf|cerebrospinal fluid)",
            re.IGNORECASE,
        ),
    ),
    (
        "pet_or_autoradiography",
        re.compile(r"\bpet\b|positron emission|autoradiograph", re.IGNORECASE),
    ),
)

_GENERIC_BBB_OUTCOME_PATTERN = re.compile(
    r"(?:cross(?:es|ed|ing)?|penetrat(?:e|es|ed|ion)|pass(?:es|ed|ing)? through)"
    r".{0,25}(?:bbb|blood.brain barrier)|"
    r"(?:bbb|blood.brain barrier).{0,25}"
    r"(?:cross(?:es|ed|ing)?|penetrat(?:e|es|ed|ion))|"
    r"brain penetrat(?:e|es|ed|ion)",
    re.IGNORECASE,
)


def load_label_decisions(
    *,
    revision: str = SOURCE_REVISION,
    max_rows: int = 0,
    label_decider: Callable[..., tuple[int | None, str]] | None = None,
    contract_version: str = CONTRACT_VERSION,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    """Load the frozen source and lazily apply the experimental outcome contract."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset = load_dataset(SOURCE_DATASET, split="train", revision=revision)
    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    resolved_revision = HfApi().dataset_info(SOURCE_DATASET, revision=revision).sha
    return label_decisions_from_rows(
        dataset,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        label_decider=label_decider,
        contract_version=contract_version,
    )


def label_decisions_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    requested_revision: str = SOURCE_REVISION,
    resolved_revision: str,
    label_decider: Callable[..., tuple[int | None, str]] | None = None,
    contract_version: str = CONTRACT_VERSION,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    """Apply the contract to an already opened copy of the frozen source."""

    metadata = {
        "dataset": SOURCE_DATASET,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "split": "train",
        "gold_contract": {
            "version": contract_version,
            "target": "experimentally supported meaningful CNS access after systemic administration",
            "positive_label": "reported meaningful or adequate experimental CNS access",
            "negative_label": "reported experimentally restricted or poor CNS access",
            "included_endpoint_families": [
                "brain_unbound",
                "brain_systemic_ratio",
                "brain_tissue",
                "csf",
                "pet_or_autoradiography",
                "experimental_bbb_outcome",
            ],
            "excluded_from_gold": [
                "computational predictions",
                "PAMPA and cell-based in-vitro permeability",
                "non-systemic CNS delivery",
                "artificially disrupted or disease-altered barriers",
                "indirect efficacy-only inference",
                "unresolved parent/metabolite or query/analyte attribution",
            ],
            "label_rule": (
                "use explicit qualitative bbb_permeability_label only after "
                "experimental outcome and provenance validation; low but nonzero "
                "exposure may remain restricted, and heterogeneous endpoints are "
                "not forced through one numeric threshold"
            ),
            "mechanism_policy": (
                "the outcome label is independent of passive diffusion, efflux, or "
                "influx mechanism; a mechanism-only proxy cannot vote without an "
                "eligible experimental CNS-access outcome"
            ),
            "distribution_policy": (
                "do not relabel, resample, or impose endpoint/mechanism quotas to "
                "make reasoning groups useful; audit downstream train-only evidence "
                "coverage separately after held-out exclusion"
            ),
        },
    }
    decision_fn = label_decider or label_record
    return (
        (
            _label_record(index, row, label_decider=decision_fn)
            for index, row in enumerate(rows)
        ),
        metadata,
    )


def classify_scope(
    record: Mapping[str, Any],
    *,
    allow_conditioned_context: bool = False,
) -> ScopeDecision:
    """Classify an eligible meaningful-CNS-access endpoint or reject it."""
    if has_reported_text(record.get("qualifying_conditions")) and not allow_conditioned_context:
        return ScopeDecision(None, None, "interpretation_altering_qualifying_conditions")

    assay_model = str(record.get("assay_model") or "")
    support_text = str(record.get("support_text") or "")
    extra_details = str(record.get("extra_details") or "")
    quant_metric = str(record.get("quant_metric") or "")
    quant_value = str(record.get("quant_value") or "")
    searchable = " | ".join((assay_model, support_text, extra_details, quant_metric))

    if (
        _PREDICTION_PATTERN.search(searchable)
        or _PREDICTION_MODEL_PATTERN.search(assay_model)
        or _CALCULATED_BBB_METRIC_PATTERN.search(searchable)
    ):
        return ScopeDecision(None, None, "computational_or_predicted_result")
    if _IN_VITRO_PATTERN.search(searchable):
        return ScopeDecision(None, None, "in_vitro_or_passive_permeability_result")
    if _EX_VIVO_ONLY_PATTERN.search(searchable):
        return ScopeDecision(None, None, "ex_vivo_only_result")
    if _NON_SYSTEMIC_ROUTE_PATTERN.search(searchable):
        return ScopeDecision(None, None, "non_systemic_or_altered_barrier_context")
    if (
        _CONDITIONABLE_ALTERED_CONTEXT_PATTERN.search(searchable)
        and not allow_conditioned_context
    ):
        return ScopeDecision(None, None, "non_systemic_or_altered_barrier_context")

    endpoint_family = _first_match(quant_metric, _METRIC_ENDPOINT_PATTERNS)
    basis = "direct_metric" if endpoint_family else None
    if endpoint_family is None:
        outcome_text = " | ".join((support_text, extra_details))
        endpoint_family = _first_match(outcome_text, _TEXT_ENDPOINT_PATTERNS)
        basis = "direct_measurement_text" if endpoint_family else None

    experimental_model = bool(_EXPERIMENTAL_MODEL_PATTERN.search(assay_model))
    measurement_action = bool(_MEASUREMENT_ACTION_PATTERN.search(support_text))
    systemic_procedure = bool(_SYSTEMIC_ROUTE_PATTERN.search(searchable))
    contextualized_quantitative_metric = (
        basis == "direct_metric"
        and has_reported_text(quant_value)
        and has_reported_text(record.get("pmid"))
        and has_reported_text(record.get("species"))
        and systemic_procedure
    )
    has_experimental_basis = (
        experimental_model or measurement_action or contextualized_quantitative_metric
    )

    if endpoint_family is None:
        generic_outcome = bool(_GENERIC_BBB_OUTCOME_PATTERN.search(support_text))
        if _INDIRECT_INFERENCE_PATTERN.search(support_text):
            return ScopeDecision(None, None, "indirect_outcome_inference")
        if (
            generic_outcome
            and experimental_model
            and systemic_procedure
            and measurement_action
        ):
            endpoint_family = "experimental_bbb_outcome"
            basis = "experimental_model_and_outcome_text"

    if endpoint_family is None:
        return ScopeDecision(None, None, "no_direct_cns_outcome")
    if not has_experimental_basis:
        return ScopeDecision(None, None, "no_explicit_experimental_basis")
    if basis == "direct_metric":
        basis = (
            "direct_metric_with_experimental_context"
            if experimental_model or measurement_action
            else "systemic_quantitative_direct_metric"
        )
    return ScopeDecision(endpoint_family, basis)


def label_record(
    record: Mapping[str, Any],
    *,
    source_index: int | None = None,
    allow_conditioned_context: bool = False,
) -> tuple[int | None, str]:
    """Return a label only for an eligible experimental CNS-access outcome."""
    if _UNSUPPORTED_METAL_SMILES_PATTERN.search(str(record.get("smiles") or "")):
        return None, "unsupported_metal_complex_identity"
    if source_index in MANUAL_SOURCE_EXCLUSIONS:
        return (
            None,
            f"manual_source_exclusion:{MANUAL_SOURCE_EXCLUSIONS[source_index]}",
        )
    scope = classify_scope(
        record,
        allow_conditioned_context=allow_conditioned_context,
    )
    if scope.rejection_reason:
        return None, scope.rejection_reason

    value = _normalized_label(record.get("bbb_permeability_label"))
    if value in POSITIVE_LABELS:
        label = 1
    elif value in NEGATIVE_LABELS:
        label = 0
    else:
        return None, "no_explicit_binary_permeability_label"
    if _has_obvious_direction_conflict(label, record.get("quant_value")):
        return None, "within_record_direction_conflict"
    return label, f"{CONTRACT_VERSION}:{scope.endpoint_family}:{scope.basis}"


def _label_record(
    index: int,
    row: Mapping[str, Any],
    *,
    label_decider: Callable[..., tuple[int | None, str]] = label_record,
) -> LabelDecision:
    index = int(row.get("source_index", index))
    label, method = label_decider(row, source_index=index)
    if label is None:
        return rejected(
            method,
            source_id=SOURCE_DATASET,
            source_index=index,
            permeability_label=row.get("bbb_permeability_label"),
            quant_metric=row.get("quant_metric"),
            assay_model=row.get("assay_model"),
            qualifying_conditions=row.get("qualifying_conditions"),
        )
    smiles = str(row.get("smiles") or "").strip()
    if not smiles:
        return rejected("missing_smiles", source_id=SOURCE_DATASET, source_index=index)
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=str(row.get("source_id") or SOURCE_DATASET),
            source_record_id=str(row.get("source_record_id") or f"row:{index}"),
            pmid=str(row.get("pmid") or ""),
            label_method=method,
            raw_value=" | ".join(
                str(value)
                for value in (
                    row.get("bbb_permeability_label"),
                    row.get("quant_metric"),
                    row.get("quant_value"),
                    row.get("quant_units"),
                )
                if value not in (None, "")
            ),
            context=" | ".join(
                str(value)
                for value in (row.get("assay_model"), row.get("species"))
                if value not in (None, "")
            ),
            source_row_uid=str(row.get("source_row_uid") or ""),
        )
    )


def _first_match(
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> str | None:
    for name, pattern in patterns:
        if pattern.search(text):
            return name
    return None


def _has_obvious_direction_conflict(label: int, quant_value: Any) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", str(quant_value or "").lower()).strip()
    negative_values = {
        "blq",
        "below limit of quantitation",
        "low",
        "lowest",
        "minimal",
        "negligible",
        "none",
        "not detected",
        "poor",
        "very low",
    }
    positive_values = {"good", "high", "rapid", "very high"}
    return (label == 1 and normalized in negative_values) or (
        label == 0 and normalized in positive_values
    )
