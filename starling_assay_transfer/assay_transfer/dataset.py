"""Source-safe prompts and a stable release schema."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import islice

import pyarrow as pa
from jinja2 import Environment, StrictUndefined

from .storage import PACKAGE
from .transporters import canonical_transporter_identifier

from typing import Any
import math, json
LABELS = {'endpoint_name':'Endpoint','measurement_text':'Known source measurement','unit_text':'Source unit'}

def _clean(value: Any) -> str | None:
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).strip()
    return text or None


def _label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").capitalize())


"""Task descriptions shared by molecule-level transfer families."""


from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    property_name: str
    description: str
    vote_value_label: str


TASKS = {
    'BBB_Martins': TaskSpec(
        'blood-brain barrier crossing',
        'You are comparing two molecules for blood-brain barrier (BBB) crossing. This property asks whether a systemically administered molecule crosses the intact blood-brain barrier and reaches brain tissue at meaningful exposure; predicted or in-vitro permeability alone does not establish the outcome.',
        'Known estimated likelihood of blood-brain barrier crossing',
    ),
    'Bioavailability_Ma': TaskSpec(
        'human absolute oral bioavailability',
        'You are comparing two molecules for human absolute oral bioavailability. Oral bioavailability is the fraction of an orally administered dose that reaches systemic circulation unchanged and reflects intestinal absorption together with first-pass gut and hepatic metabolism.',
        'Known estimated likelihood of meaningful human oral bioavailability',
    ),
    'Skin_Reaction': TaskSpec(
        'skin sensitization',
        'You are comparing two molecules for skin sensitization. Skin sensitization is the ability to induce immune-mediated contact allergy after dermal exposure; it is distinct from acute irritation, phototoxicity, and nonspecific local skin injury.',
        'Known estimated likelihood of skin sensitization',
    ),
    'DILI': TaskSpec(
        'drug-induced liver injury',
        'You are comparing two molecules for drug-induced liver injury (DILI). The outcome is clinically meaningful liver injury attributable to exposure to the molecule.',
        'Known estimated likelihood of drug-induced liver injury',
    ),
    'Ames': TaskSpec(
        'Ames mutagenicity',
        'You are comparing two molecules for bacterial reverse-mutation (Ames) mutagenicity. The reported assay condition, including strain and metabolic activation, is part of the outcome context.',
        'Known estimated likelihood of an Ames-positive outcome',
    ),
    'Carcinogens': TaskSpec(
        'carcinogenicity',
        'You are comparing two molecules for experimentally observed carcinogenicity under the reported species and exposure context.',
        'Known estimated likelihood of a carcinogenic outcome',
    ),
}


def display_value(value: float, label_mode: str) -> str:
    if label_mode == "vote_mean":
        return f"{100.0 * value:.1f}%"
    if label_mode == "hard_cap":
        return "positive" if value >= 0.5 else "negative"
    if label_mode == "oral_continuous":
        return f"{value:.1f}%"
    raise ValueError(f"unsupported label mode: {label_mode}")


def value_label(spec: TaskSpec, label_mode: str) -> str:
    if label_mode == "vote_mean":
        return spec.vote_value_label
    if label_mode == "hard_cap":
        return f"Known {spec.property_name} class"
    return "Known human absolute oral bioavailability"

HIDDEN = {'needs_more_context', 'smiles', 'source_smiles', 'confidence', 'extraction_id',
          'global_identifier', 'paragraph_idx', 'pmid', 'source_record_id', 'source_index'}
RESULTS = {'measurement_text', 'canonical_measurement_text', 'support_text', 'extra_details',
           'bbb_permeability_label', 'bbb_transport_label', 'interaction_conclusion',
           'passive_bbb_interpretation', 'metric_uncertainty', 'evidence_basis',
           'substrate_status', 'comparator_exposure', 'category_id', 'category_rank'}
PROJECTION = json.loads((PACKAGE/'assets/prompt_projection.json').read_text())
BBB_DISPLAY = json.loads((PACKAGE/'assets/bbb_display.json').read_text())

RESULT_TOKENS = ('result', 'outcome', 'conclusion', 'classification', 'effect_direction',
                 'response_value', 'readout', 'label', 'measurement_value')


@lru_cache(maxsize=1)
def template():
    return Environment(undefined=StrictUndefined, autoescape=False).from_string(
        (PACKAGE / 'templates/record.jinja').read_text())


def payload(row):
    value = json.loads(row['prompt_payload_json'])
    value.update(json.loads(row.get('pair_fields_json') or '{}'))
    value.update(endpoint_name=row['endpoint'], measurement_text=row['canonical_measurement_text'],
                 unit_text=row['canonical_unit_text'])
    if row['task_id'] == 'BBB_Martins':
        labels = BBB_DISPLAY
        scale = row['canonical_measurement_scale_id']
        if row['measurement_kind'] == 'binary':
            if scale not in labels or row['category_id'] not in ('negative','positive'):
                raise ValueError(f'Unsupported BBB display category: {scale}')
            names = labels[scale]
            value.update(measurement_text=names[row['category_id']=='positive'],unit_text=names[2])
        if 'transporter_identifier' in value:
            value['transporter_identifier'] = canonical_transporter_identifier(value['transporter_identifier'])
    return value


def fields(row, known=False):
    values = payload(row)
    if row['task_id'] == 'Skin_Reaction':
        projection = PROJECTION
        binding = projection['tasks']['skin_reaction'][row['source_id']]
        names = list(binding['query'])
        if known:
            names = names[:1] + ['measurement_text'] + names[1:]
            names += [key for key in binding['known_only'] if key != 'measurement_text']
    else:
        names = [key for key in values if key not in HIDDEN and (known or (
            key not in RESULTS and not any(token in key.lower() for token in RESULT_TOKENS)))]
    return [(_label(key), cleaned) for key in names if (cleaned := _clean(values.get(key))) is not None]


def prompt(reference, query):
    return template().render(known_smiles=reference['parent_smiles'], query_smiles=query['parent_smiles'],
                             known_fields=fields(reference, True), query_fields=fields(query)).strip()


@lru_cache(maxsize=1)
def context_template():
    return Environment(undefined=StrictUndefined,autoescape=False).from_string((PACKAGE/'templates/context.jinja').read_text())


def context_prompt(reference, query):
    spec = TASKS[query['task_id']]
    conditions = lambda row: '\n'.join('- '+str(atom).replace('=',': ',1).replace('_',' ') for atom in row['condition_atoms'])
    return context_template().render(description=spec.description, a_smiles=reference['parent_smiles'],
        b_smiles=query['parent_smiles'],a_context=conditions(reference),b_context=conditions(query),
        known_label=spec.vote_value_label,known_value=display_value(reference['geometry_value'],'vote_mean'),
        property_name=spec.property_name).strip()


@lru_cache(maxsize=1)
def release_schema():
    encoded = json.loads((PACKAGE/'assets/release_schema.json').read_text())['arrow_ipc_base64']
    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(encoded)))
    for name in ('family','effective_contract_sha256'):
        schema = schema.append(pa.field(name,pa.string()))
    return schema


def release_row(item, reference, query, policy, family):
    score, direct = item['probability'], item['component'] == 'direct'
    text = context_prompt(reference,query) if direct else prompt(reference,query)
    row = {'pair_id':item['pair_id'],'task_id':policy['task'],'family':family,'split':item['split'],
           'training_component':item['component'],'level':query['level'],'source_level':query['level'],
           'dataset_level':query['level'],'query_record_id':query['record_id'],'retrieval_record_id':reference['record_id'],
           'query_parent_id':query['parent_id'],'retrieval_parent_id':reference['parent_id'],
           'query_molecule_identity_key':query['parent_id'],'retrieval_molecule_identity_key':reference['parent_id'],
           'query_benchmark_row_id':query['record_id'],'retrieval_benchmark_row_id':reference['record_id'],
           'query_source_row_uid':query.get('source_row_uid'),'retrieval_source_row_uid':reference.get('source_row_uid'),
           'query_smiles':query['parent_smiles'],'retrieval_smiles':reference['parent_smiles'],
           'pair_bucket_key':query.get('pair_bucket_key'),'upstream_pair_bucket_key':query.get('upstream_pair_bucket_key'),
           'prompt':text,'completion':'(A)' if score >= .5 else '(B)','target_a':score,'target_b':1-score,
           'target_distribution':{'transfer':score,'nontransfer':1-score},'binary_label':int(score>=.5),
           'sd_logistic_similarity':score,'morgan_tanimoto_similarity':item['similarity'],
           'retrieval_parent_rank':item.get('retrieval_parent_rank'),'ranking_member_index':item.get('retrieval_parent_rank'),
           'ranking_query_id':query['record_id'] if item['split'] != 'train' else None,
           'evaluation_regime':query.get('evaluation_regime','gold'),'calibration_scope':item['calibration_scope'],
           'logit_geometry_transform':item['transform'],'effective_contract_sha256':policy['contract_sha256'],
           'query_source_payload_json':query.get('prompt_payload_json',query.get('raw_json')),
           'retrieval_source_payload_json':reference.get('prompt_payload_json',reference.get('raw_json')),
           'representation':'direct_context' if direct else 'indirect_record','measurement_kind':query['measurement_kind'],
           'retrieval_strategy':'top_morgan_balanced_no_reverse' if direct else policy['record_sampling_policy'],
           'canonical_endpoint_key':policy['task'] if direct else query['pair_bucket_key'],
           'source_id':'gold_v1' if direct else query['source_id'],'assay_concept':policy['task'] if direct else query['source_id'],
           'query_geometry_value':query['geometry_value'],'retrieval_geometry_value':reference['geometry_value'],
           'query_scaffold':query['scaffold'],'retrieval_scaffold':reference['scaffold'],
           'calibration_sample_standard_deviation':item.get('sd'),
           'target_train_sample_standard_deviation':None if query.get('evaluation_regime')=='ood' else item.get('sd'),
           'absolute_canonical_value_difference':abs(query['geometry_value']-reference['geometry_value']),
           'absolute_geometry_difference':item.get('effective_distance'),
           'standardized_canonical_value_difference':item.get('standardized_distance'),
           'is_decisive':score>.6 or score<.4,'ranking_balance_label':'near' if score>=.5 else 'far',
           'label_mode':'vote_mean' if direct else 'mixed_record_level'}
    for role, source in (('query',query),('retrieval',reference)):
        row[f'{role}_condition_atoms'] = source['condition_atoms'] if direct else [f'{key}={value}' for key,value in fields(source)]
        row[f'{role}_condition_group'] = source['condition_group'] if direct else source['pair_bucket_key']
        row[f'{role}_condition_scope'] = 'external' if direct else 'exact_pair_bucket'
        row[f'{role}_source_record_count'] = source.get('source_record_count',1)
        row[f'{role}_gold_Y'] = source.get('Y') if direct else int(source['geometry_value']) if source['measurement_kind']=='binary' else None
    for key in ('bucket_total_record_count','bucket_train_record_count','bucket_train_parent_count'):
        row[key] = query.get(key)
    row['metadata'] = {'soft_targets':{'(A)':score,'(B)':1-score},
                       'calibration_sha256':item.get('calibration_sha256'),'source_snapshot_sha256':item.get('source_manifest_sha256'),
                       'target_contract':json.dumps(policy['common']['numeric_target'],sort_keys=True),
                       'groups':{'task_id':policy['task'],'source_id':row['source_id'],'level':query['level'],
                                 'measurement_kind':query['measurement_kind'],'assay_concept':row['assay_concept']}}
    return {field:row.get(field) for field in release_schema().names}


def rendered_rows(pairs, records, policy, family, workers=1):
    def render(item):
        row = release_row(item,records[item['a_id']],records[item['b_id']],policy,family)
        structure = release_schema().field('metadata').type
        row['metadata']['groups'] = {field.name:row['metadata']['groups'].get(field.name) for field in structure.field('groups').type}
        row['metadata'] = {field.name:row['metadata'].get(field.name) for field in structure}
        return row
    if workers == 1:
        yield from map(render, pairs)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pairs = iter(pairs)
            while batch := list(islice(pairs, 2000)):
                yield from executor.map(render, batch)


def create_dataset(stage_root,task_root,policy,family,workers):
    from .storage import read_json,read_parquet,verify_manifest,write_json,write_parquet
    source=task_root/'targets'
    records=({r['record_id']:r for r in read_parquet(task_root/'pool/records.parquet')}
             if family=='record' else read_json(task_root/'sampling/records.json'))
    counts={}
    for split in ('train','validation_ranking','test_ranking'):
        pairs=read_parquet(source/f'{split}.parquet')
        counts[split]=write_parquet(stage_root/split/'data.parquet',
            rendered_rows(pairs,records,policy,family,workers),release_schema())
    write_json(stage_root/'effective_contract.json',policy)
    write_json(stage_root/'calibration.json',read_json(source/'calibration.json'))
    scope=verify_manifest(task_root/'pool')['scope']
    config=['---','configs:','- config_name: default','  data_files:']
    for split in counts:config += [f'  - split: {split}',f'    path: {split}/*.parquet']
    config += ['---',f'# Canonical {family}: {policy["task"]}','',f'Scope: {scope}.',
               'See manifest.json and provenance/verification.json for hashes, coverage and verification.','']
    (stage_root/'README.md').write_text('\n'.join(config))
    return {'scope':scope,'task':policy['task'],'family':family,'coverage':counts}


def upload_verified(task_root,policy,family,repo_id,revision=None,private=False):
    from .storage import digest,read_json,verify_manifest,write_json
    from .verify import verify_task
    release=task_root/'dataset'/family
    receipt=read_json(task_root/f'verification/{family}/receipt.json')
    manifests={name:verify_manifest(task_root/name) for name in
               ('pool','sampling','augmentation','targets',f'dataset/{family}',f'verification/{family}')}
    if receipt.get('status')!='verified' or receipt.get('scope')!='full':
        raise ValueError('HF upload requires a verified full release')
    current=verify_task(task_root,policy,family)
    if current!=receipt:raise ValueError('Verification receipt does not match the current release')
    dataset_manifest=manifests[f'dataset/{family}']
    revision=revision or f'canonical-{dataset_manifest["build_signature"][:16]}'
    from huggingface_hub import HfApi,CommitOperationAdd
    api=HfApi()
    api.create_repo(repo_id=repo_id,repo_type='dataset',private=private,exist_ok=True)
    api.create_branch(repo_id=repo_id,repo_type='dataset',branch=revision,exist_ok=True)
    operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(release/name))
                for name in sorted(dataset_manifest['artifacts'])+['manifest.json']]
    operations.append(CommitOperationAdd(path_in_repo='provenance/verification.json',
        path_or_fileobj=str(task_root/f'verification/{family}/receipt.json')))
    commit=api.create_commit(repo_id=repo_id,repo_type='dataset',revision=revision,operations=operations,
                             commit_message=f'Canonical {family}: {policy["task"]}')
    result={'repo_id':repo_id,'revision':revision,'commit_sha':commit.oid,'commit_url':commit.commit_url,
            'dataset_manifest_sha256':digest(dataset_manifest),'verification_sha256':digest(receipt)}
    write_json(task_root/'uploads'/f'{commit.oid}.json',result)
    return result
