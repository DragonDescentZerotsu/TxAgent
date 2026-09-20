"""Load version-owned prose, tasks and cards; render two model-visible messages.

Callers assemble dynamic evidence and state using the prose map. Jinja renders
the system string; JSON serialization preserves historical user-message bytes.
Asset manifests identify every template, task, level and card input.
"""

from __future__ import annotations

from functools import lru_cache
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import yaml

from predict.harnesses.progressive.references import normalize_headings, validate_contract


TEMPLATE_DIR = Path(__file__).parent
PROMPT_DIR = TEMPLATE_DIR / 'prompts'
ACTIVE_PROMPT_VERSION = 'reranked_progressive_v8'
PROMPT_MODIFIER_SUFFIXES = (
    ('_no_scores_no_smiles_no_query_smiles_no_query_prior',
     ('no_scores', 'no_smiles', 'no_query_smiles', 'no_query_prior')),
    ('_no_scores_no_smiles_no_query_smiles',
     ('no_scores', 'no_smiles', 'no_query_smiles')),
    ('_no_scores_no_query_prior', ('no_scores', 'no_query_prior')),
    ('_no_scores', ('no_scores',)),
    ('_no_smiles_no_query_smiles_no_query_prior',
     ('no_smiles', 'no_query_smiles', 'no_query_prior')),
    ('_no_smiles_no_query_smiles', ('no_smiles', 'no_query_smiles')),
    ('_no_query_prior', ('no_query_prior',)),
    ('_no_query_smiles', ('no_query_smiles',)),
    ('_no_smiles', ('no_smiles',)),
)


def split_prompt_version(version: str) -> tuple[str, tuple[str, ...]]:
    """Split a composable variant name into its immutable base and modifiers."""
    if (PROMPT_DIR / version).is_dir():
        return version, ()
    for suffix, modifiers in PROMPT_MODIFIER_SUFFIXES:
        if version.endswith(suffix):
            base = version[:-len(suffix)]
            if (PROMPT_DIR / base).is_dir():
                return base, modifiers
    return version, ()


def prompt_directory(version: str) -> Path:
    """Resolve a bundle path without ever falling back to the active prompt."""
    if Path(version).name != version or version in {'.', '..', 'legacy'}:
        raise ValueError(f'invalid prompt version: {version!r}')
    base, _ = split_prompt_version(version)
    current = PROMPT_DIR / base
    return current if current.is_dir() else PROMPT_DIR / 'legacy' / version


def behavior_version(version: str) -> str:
    """Return the frozen assembly behavior inherited by a cloned bundle."""
    path = prompt_directory(version) / 'provenance.json'
    if not path.is_file():
        return version
    provenance = json.loads(path.read_text(encoding='utf-8'))
    return str(provenance.get('runtime_parent_version') or version)


def prompt_asset_path(version: str, relative: str) -> Path:
    """Resolve an asset through the bundle's explicitly pinned parent chain."""
    seen: set[str] = set()
    while True:
        if version in seen:
            raise ValueError(f'cyclic prompt asset inheritance at {version!r}')
        seen.add(version)
        directory = prompt_directory(version)
        local = directory / relative
        if local.is_file():
            return local
        provenance = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
        parent = provenance.get('asset_parent_version')
        if not parent:
            return local
        version = str(parent)


@lru_cache(maxsize=None)
def prompt_assets(version: str) -> dict[str, Any]:
    if behavior_version(version) != ACTIVE_PROMPT_VERSION:
        raise ValueError(
            f"inactive prompt version {version!r}; use or clone {ACTIVE_PROMPT_VERSION!r}"
    )
    directory = prompt_directory(version)
    provenance_path = directory / 'provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    _, applied_modifiers = split_prompt_version(version)
    settings = dict(provenance.get('runtime') or {})
    modifier_settings = settings.get('prompt_modifiers') or {}
    unknown = [name for name in applied_modifiers if name not in modifier_settings]
    if unknown:
        raise ValueError(f'{version!r} does not support prompt modifier {unknown[0]!r}')
    for name in applied_modifiers:
        settings.update(modifier_settings[name].get('runtime') or {})
    if settings.get('reasoning_reference_contract'):
        validate_contract(settings['reasoning_reference_contract'])
    molecule_metadata = settings.get('molecule_metadata_contract')
    if molecule_metadata is not None:
        fixed = {
            key: molecule_metadata.get(key)
            for key in ('schema_version', 'modes', 'query_visible')
        }
        if (
            fixed != {
                'schema_version': 'quotient_molecule_metadata.v1',
                'modes': ['none', 'raw', 'motif', 'coarse'],
                'query_visible': True,
            }
            or molecule_metadata.get('evidence_levels') not in ([1], [1, 2])
            or molecule_metadata.get('missing_policy') not in {'error', 'omit'}
        ):
            raise ValueError('invalid quotient molecule-metadata contract')
    texts = yaml.safe_load(prompt_asset_path(version, 'user_shared.yaml').read_text())
    tasks = yaml.safe_load(prompt_asset_path(version, 'tasks.yaml').read_text())
    card = yaml.safe_load(prompt_asset_path(version, 'card.yaml').read_text())
    levels: dict[str, dict[str, Any]] = {}
    context_levels: dict[str, list[dict[str, Any]]] = {}
    for level in range(1, 7):
        path = prompt_asset_path(version, f'levels/L{level}.yaml')
        for task, definition in yaml.safe_load(path.read_text()).items():
            context_level = definition.pop('context_level', None)
            if context_level is not None:
                context_levels.setdefault(task, []).append(context_level)
            if definition:
                levels.setdefault(task, {})[path.stem] = definition
    if (
        version.startswith('tianang_aligned')
        or version.startswith('reranked_progressive_')
        or str(settings.get('output_contract', '')).startswith('full_flat_progressive.')
    ):
        card['task_levels'] = levels
    bundle_path = prompt_asset_path(version, 'bundle.yaml')
    bundle = yaml.safe_load(bundle_path.read_text()) if bundle_path.exists() else None
    if bundle is not None:
        bundle['task_levels'] = levels
    descriptions_path = prompt_asset_path(version, 'level_descriptions.yaml')
    descriptions = yaml.safe_load(descriptions_path.read_text()) if descriptions_path.exists() else {}
    mode_names = ['morgan', 'assay-transfer', 'joint']
    for optional_mode in ('assay-transfer-contrastive', 'paired-order'):
        if prompt_asset_path(version, f'modes/{optional_mode}/user.yaml').is_file():
            mode_names.append(optional_mode)
    modes = {
        mode: yaml.safe_load(prompt_asset_path(version, f'modes/{mode}/user.yaml').read_text())
        for mode in mode_names
    }
    if not {'morgan', 'assay-transfer', 'joint'} <= set(modes):
        raise ValueError('Active prompt requires all three mode assets')
    return dict(user_shared=texts, tasks=tasks, card=card, levels=levels, bundle=bundle, modes=modes,
                context_levels=context_levels, level_descriptions=descriptions,
                settings=settings, applied_modifiers=applied_modifiers)


def prompt_asset_manifest(version: str) -> dict[str, Any]:
    assets = prompt_assets(version)
    directory = prompt_directory(version)
    hashes = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(directory.rglob('*')) if path.suffix in {'.yaml', '.jinja'}}
    if (directory / 'provenance.json').is_file():
        hashes['provenance.json'] = hashlib.sha256((directory / 'provenance.json').read_bytes()).hexdigest()
    parent = json.loads((directory / 'provenance.json').read_text()).get('asset_parent_version')
    if parent:
        hashes[f'@{parent}'] = prompt_asset_manifest(str(parent))['sha256']
    if assets['applied_modifiers']:
        definitions = assets['settings']['prompt_modifiers']
        applied = {name: definitions[name] for name in assets['applied_modifiers']}
        hashes['@applied_modifiers'] = hashlib.sha256(
            json.dumps(applied, sort_keys=True).encode()
        ).hexdigest()
    assembly_names = ['prompt.py', 'grammar.py', 'references.py', 'state.py', '_records.py']
    assembly_names.extend(['level_selection.py', '_visibility.py', 'cache_matched.py'])
    assembly = {
        name: hashlib.sha256(
            (
                TEMPLATE_DIR / name
                if name != 'cache_matched.py'
                else TEMPLATE_DIR.parents[1] / 'retrieval/assay_reranking/cache_matched.py'
            ).read_bytes()
        ).hexdigest()
        for name in assembly_names
    }
    return dict(version=version, directory=str(directory), files_sha256=hashes,
                assembly_files_sha256=assembly,
                sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


@lru_cache(maxsize=1)
def template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
    )


def render_progressive_messages(
    *,
    system_role: str,
    payload: Mapping[str, Any],
    template_name: str = "progressive.jinja",
    prompt_version: str = "standard_v1",
    compact: bool = False,
) -> list[dict[str, Any]]:
    # template_name remains the historical manifest identity, not an active input.
    assets = prompt_assets(prompt_version)
    template_path = prompt_asset_path(prompt_version, 'system.jinja').relative_to(TEMPLATE_DIR)
    mode = payload.get('protocol', {}).get('prompt_mode')
    if assets['modes'] and mode not in assets['modes']:
        raise ValueError('Mode-specific prompt requires a valid prompt_mode')
    rendered = template_environment().get_template(str(template_path)).render(
        system_role=system_role,
        task=payload['task_definition']['task'],
        task_definition=payload['task_definition'],
        prompt_version=prompt_version,
        prompt_settings=assets['settings'],
        mode=mode,
        current_level=int(payload.get('level_context', {}).get('current_level') or 0),
        mode_template=str(prompt_asset_path(prompt_version, f'modes/{mode}/system.jinja').relative_to(TEMPLATE_DIR)),
    )
    system = json.loads(rendered)
    if not isinstance(system, str):
        raise ValueError('system prompt must render a JSON string')
    payload = deepcopy(payload)
    modifier_settings = assets['settings'].get('prompt_modifiers') or {}
    for name in assets['applied_modifiers']:
        modifier = modifier_settings[name]
        system += '\n\n' + str(modifier['instruction'])
        if modifier.get('hide_query_smiles'):
            payload['query'].pop('canonical_smiles', None)
        if modifier.get('hide_evidence_smiles'):
            for molecule in payload['active_evidence']:
                molecule.pop('canonical_smiles', None)
        if modifier.get('hide_query_prior'):
            payload.pop('query_prior', None)
        if modifier.get('hide_retrieval_scores'):
            for molecule in payload['active_evidence']:
                for key in (
                    'morgan_similarity', 'transfer_likelihood',
                    'morgan_top5_rank', 'assay_transfer_top5_rank',
                    'morgan_panel_rank', 'assay_transfer_panel_rank',
                    'morgan_score_levels', 'transfer_score_level',
                ):
                    molecule.pop(key, None)
                for card in molecule['evidence_cards']:
                    card.pop('morgan_similarity', None)
                    card.pop('transfer_likelihood', None)
    for key in ('retrieval_by_level', 'prompt_mode', 'version'):
        payload['protocol'].pop(key, None)
    for molecule in payload['active_evidence']:
        molecule.pop('analog_id', None)
        for card in molecule['evidence_cards']:
            card.pop('retrieved_by', None)
            card.pop('evidence_family', None)
    user_template = assets['settings'].get('user_template')
    user = (
        template_environment().get_template(
            str(prompt_asset_path(prompt_version, str(user_template)).relative_to(TEMPLATE_DIR))
        ).render(payload=payload, mode=mode)
        if user_template
        else json.dumps(payload, ensure_ascii=False, separators=(',', ':') if compact else None)
    )
    if assets['settings'].get('reasoning_reference_contract'):
        user = normalize_headings(user)
    return [dict(role='system', content=system), dict(role='user', content=user)]


def build_level_messages(
    *,
    contract: Any,
    levels: list[dict[str, Any]],
    current_level: int,
    prepared: Mapping[str, Any],
    active: Mapping[str, Any],
    prior_state: Mapping[str, Any] | None,
    profile: str,
    prompt_version: str,
    record_limit: int,
    l2_record_limit: int,
    indirect_record_limit: int | Mapping[str, int],
    molecule_limit: int,
    include_indirect: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Turn prepared retrieval records and prior state into the exact request and reference index.

    Retrieval is already complete when this function is called.  Later prompts
    cannot be fully pre-rendered because ``prior_state`` is the preceding model
    response; all evidence selection itself remains prepared in advance.
    """
    if profile == "context_records":
        from predict.harnesses.progressive import _records as context_records

        common = dict(
            contract=contract,
            current_level=current_level,
            query_smiles=prepared["query_smiles"],
            query_molecule_description=prepared.get("query_molecule_description"),
            condition_sentence=prepared["condition_sentence"],
            query_prior=prepared["query_prior"] or None,
            query_tool_summary=prepared.get("query_tool_summary") or {},
            active=active,
            prior_state=prior_state,
            prompt_version=prompt_version,
            record_limit=record_limit,
            l2_record_limit=l2_record_limit,
            indirect_record_limit=indirect_record_limit,
        )
        if context_records.is_tianang_aligned(prompt_version):
            return context_records.build_tianang_aligned_messages(
                levels=levels,
                retrieval_policy=(prepared.get("retrieval_policy") or {}).get("stages"),
                molecule_limit=molecule_limit,
                return_reference_index=True,
                **common,
            )
        return context_records.build_messages(
            include_indirect=include_indirect,
            **common,
        ), None

    from predict.harnesses.progressive.state import build_progressive_messages

    return build_progressive_messages(
        contract=contract,
        levels=levels,
        current_level=current_level,
        query_smiles=prepared["query_smiles"],
        query_molecule_description=prepared.get("query_molecule_description"),
        condition_sentence=prepared["condition_sentence"],
        query_prior=prepared["query_prior"] or None,
        query_tool_summary=prepared.get("query_tool_summary") or {},
        active=active,
        prior_state=prior_state,
    ), None
