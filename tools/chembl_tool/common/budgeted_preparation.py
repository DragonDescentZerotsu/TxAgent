"""Reusable, source-bound candidate pools for simple baseline preparation.

Each worker keeps at most one partitioned index resident, loaded on cache miss. Disk entries share the
expensive, selector-independent retrieval across settings and conditions. Tool
prefetch and rendering stay in the existing runner.
"""
import fcntl
import json
from pathlib import Path
import threading
import zlib

from .budgeted_evidence import content_hash
from .build_runtime import local_input
from .budgeted_retrieval import build_evidence_view, retrieve_candidates
from .json_utils import atomic_output_path
from .retrieval_features import load_retrieval_index


def evidence_view(index, task, manifest):
    if manifest.get('benchmark_profile'):
        from .tdc_evidence import merge_train_labels
        from .json_utils import read_jsonl
        inputs = manifest['inputs'][task]
        index = merge_train_labels(index, task, read_jsonl(Path(inputs['train']['path'])),
            read_jsonl(Path(inputs['heldout']['path'])))
    return build_evidence_view(index)


class PreparationCache:
    def __init__(self, root):
        self.root = Path(root)
        self._view = None
        self._view_key = None
        self._guard = threading.Lock()
        self.hits = self.misses = self.index_loads = 0

    def view(self, task, manifest):
        source = manifest['inputs'][task]['index']
        key = content_hash([manifest['inputs'][task], manifest['code']])
        with self._guard:
            if self._view_key != key:
                # Release the previous task before allocating another large index.
                self._view = self._view_key = None
                path = Path(source['path'])
                # Keep descriptor-relative paths intact; large pickles reuse verified NVMe copies.
                path = path if path.suffix == '.json' else local_input(path)
                self._view = evidence_view(load_retrieval_index(path), task, manifest)
                self._view_key = key
                self.index_loads += 1
            return self._view

    def candidates(self, smiles, view, *, task, manifest, split_scheme,
                   candidate_molecules, min_similarity, indirect_identity_policy='parent_disjoint'):
        inputs = manifest['inputs'][task]
        binding = {'version': 1, 'task': task, 'smiles': smiles,
                   'sources': {k: v['sha256'] for k, v in inputs.items() if k != 'input'},
                   'code': manifest['code'], 'split_scheme': split_scheme,
                   'candidate_molecules': candidate_molecules, 'min_similarity': min_similarity,
                   'indirect_identity_policy': indirect_identity_policy}
        key = content_hash(binding)
        path = self.root / key[:2] / f'{key}.json.zlib'
        path.parent.mkdir(parents=True, exist_ok=True)
        # Separate processes/condition rows may ask for the same molecule.
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                saved = json.loads(zlib.decompress(path.read_bytes()))
                if saved['binding'] != binding:
                    raise ValueError('candidate cache binding mismatch')
                with self._guard:
                    self.hits += 1
                return saved['pool']
            if view is None:
                view = self.view(task, manifest)
            pool = retrieve_candidates(smiles, view, split_scheme=split_scheme,
                                       candidate_molecules=candidate_molecules,
                                       min_similarity=min_similarity,
                                       indirect_identity_policy=indirect_identity_policy)
            data = json.dumps({'binding': binding, 'pool': pool},
                              ensure_ascii=False, separators=(',', ':')).encode()
            with atomic_output_path(path) as temporary:
                temporary.write_bytes(zlib.compress(data, level=1))
            with self._guard:
                self.misses += 1
            return pool

    def snapshot(self):
        with self._guard:
            return dict(hits=self.hits, misses=self.misses, index_loads=self.index_loads)
