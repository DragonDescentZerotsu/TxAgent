"""Gold memberships and task-wide parent/scaffold disjointness."""

from collections import Counter

def assign(rows, gold, policy):
    parents = {split: {row['parent_id'] for row in group} for split, group in gold.items()}
    for row in rows:
        testing = row['parent_id'] in parents['test']
        validation = row['parent_id'] in parents['valid']
        split = 'test_gate_only' if testing else 'validation' if validation else 'train'
        if not policy['evaluation_levels'] and split != 'train':
            split = 'gate_only'
        row.update(dataset_split=split, validation_parent_eligible=validation and not testing)
    return dict(Counter(f'{row["level"]}/{row["dataset_split"]}' for row in rows))

def filter_training(rows, heldout, policy):
    parents = {row['parent_id'] for row in heldout}
    scaffolds = {row['scaffold'] for row in heldout}
    retained, excluded = [], []
    for row in rows:
        reason = ('heldout_parent' if row['parent_id'] in parents else
                  'heldout_scaffold' if policy['split_disjointness'] == 'scaffold' and row['scaffold'] in scaffolds else None)
        if row.get('dataset_split', 'train') == 'train' and reason:
            excluded.append({**row, 'exclusion_reason': reason})
        else:
            retained.append(row)
    return retained, excluded


def filter_gold(groups, policy):
    heldout = groups['valid'] + groups['test']
    train, excluded = filter_training(groups['train'], heldout, policy)
    retained = {**groups, 'train': train}
    ids = [row['record_id'] for rows in retained.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Retained Gold identities repeat across splits')
    if not train:
        raise ValueError('No Gold training records remain after disjointness filtering')
    return retained, excluded


def check_disjointness(training, heldout, policy):
    _, excluded = filter_training(training, heldout, policy)
    if excluded:
        raise ValueError(f'Task-wide {policy["split_disjointness"]} overlap: {excluded[0]["record_id"]}')
    return {'mode': policy['split_disjointness'], 'scope': 'task',
            'training_records_checked': len(training), 'heldout_records_checked': len(heldout)}
