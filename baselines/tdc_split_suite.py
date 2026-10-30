"""Run the fixed Skin/Carcinogens ML protocol on prepared TDC training pools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = None
AUXILIARY = None
MINIMOL_RUNTIME_ROOT = Path(os.environ.get(
    'TXAGENT_MINIMOL_RUNTIME_ROOT', str(Path.home() / '.cache' / 'txagent' / 'minimol')))
ENV = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
       'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
       'PYTHONPATH': os.pathsep.join(map(str, (
           '.', MINIMOL_RUNTIME_ROOT / 'pydeps', MINIMOL_RUNTIME_ROOT / 'runtime_41642328'))),
       'MPLCONFIGDIR': '/tmp/tdc-suite-matplotlib'}
GPU = queue.Queue()
STOP = threading.Event()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def call(module, args, output, complete, gpu=False):
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', module, *map(str, args)]
    receipt = output / 'command.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['argv'] == command
        if complete.exists():
            return
    env = dict(ENV)
    device = GPU.get() if gpu else None
    if gpu:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
    write(receipt, {'argv': command, 'gpu': device})
    print('START', str(output.relative_to(ROOT)), flush=True)
    try:
        with (output / 'run.log').open('a') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert complete.exists(), complete
        print('DONE', str(output.relative_to(ROOT)), flush=True)
    finally:
        if gpu:
            GPU.put(device)


def monitor():
    while not STOP.is_set():
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
        with (ROOT / 'resources.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.time(), 'gpus': p.stdout, 'error': p.stderr,
                                'load': os.getloadavg()}) + '\n')
        STOP.wait(15)


def prepare(pool):
    out = ROOT / pool
    call('baselines.minimol.run_multitask', ['prepare', '--prepared', out/'prepared', '--data-dir',out/'data',
         '--auxiliary',AUXILIARY,
         '--condition-field', 'none' if pool=='tdc_only' else 'model_condition', '--preserve-main-split',
         '--prepare-workers',32], out/'prepare_job', out/'prepared/graphs.json')
    call('baselines.minimol.run_multitask', ['embed','--prepared',out/'prepared','--output',out/'frozen',
         '--workers',2], out/'embed_job', out/'frozen/embeddings/test.pt', gpu=True)


def head_knn(pool):
    out = ROOT/pool
    common = ['--data-dir',out/'data','--embedding-cache-dir',out/'frozen/embeddings']
    args = common + ['--output-dir',out/'frozen','--device','cpu','--epochs',25,'--ensemble-size',5,
                     '--threshold-strategy','valid_macro_f1','--hidden-dim',512,'--depth',3,'--lr',0.0003]
    if pool=='mixed': args += ['--condition-field','model_condition']
    call('baselines.minimol.run_bioavailability_ma', args, out/'head_job', out/'frozen/metrics.json')
    for method,module in [('minimol_knn','baselines.minimol.run_embedding_knn'),('morgan_knn','baselines.structure_knn.run')]:
        for policy in ['all_train_unique_molecules','same_condition_then_null']:
            target=out/method/policy
            args=common if method=='minimol_knn' else ['--data-dir',out/'data']
            call(module,args+['--output-dir',target,'--condition-policy',policy,'--k',3],target,target/'metrics.json')


def train(job):
    pool,weight,seed=job;out=ROOT/pool
    target=out/f'aux_{weight:g}'/f'seed_{seed}'
    call('baselines.minimol.run_multitask',['train','--prepared',out/'prepared','--output',target,
         '--aux-weight',weight,'--seed',seed],target,target/'training.json',gpu=True)


def forest(pool):
    out=ROOT/pool;prepared=out/'rf/prepared'
    call('baselines.retrieval_forest',['prepare','--prepared',prepared,'--data-dir',out/'data',
         '--source-prepared',out/'prepared','--workers',24],out/'rf/prepare_job',prepared/'manifest.json')
    jobs=[]
    for policy in [0,1]:
        target=out/'rf'/f'policy_{policy}'
        for features in ['sqrt','0.3']:
            for leaf in [1,5,20]:
                candidate=target/'candidates'/f'features_{features}_leaf_{leaf}'
                jobs.append((policy,features,leaf,candidate))
    def fit(job):
        policy,features,leaf,target=job
        call('baselines.retrieval_forest',['fit','--prepared',prepared,'--policy',policy,
             '--max-features',features,'--min-leaf',leaf,'--output',target,'--workers',8],target,target/'training.json')
    with ThreadPoolExecutor(max_workers=3) as ex: list(ex.map(fit,jobs))
    for policy in [0,1]:
        target=out/'rf'/f'policy_{policy}'
        call('baselines.retrieval_forest',['finalize','--prepared',prepared,'--policy',policy,'--output',target],
             target,target/'metrics.json')


def main():
    global ROOT, AUXILIARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs-per-gpu', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    if manifest['task'] not in ('skin_reaction', 'carcinogens'):
        raise ValueError('This fixed head profile applies only to Skin and Carcinogens')
    AUXILIARY = manifest['auxiliary']['path']
    # A matching command and completion file alone do not establish safe resume.
    inputs = {**manifest['inputs'], AUXILIARY: manifest['auxiliary']['sha256']}
    code = {name: digest for name, digest in json.loads((ROOT/'code_hashes.json').read_text()).items()
            if name.startswith('baselines/')}
    for name, expected in {**inputs, **code}.items():
        if Path(name).resolve() == Path(__file__).resolve():
            continue  # Orchestration maintenance does not alter model artifacts.
        with Path(name).open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Input or model code changed; use a fresh experiment root: {name}')
    for _ in range(args.jobs_per_gpu):
        for device in range(8):
            GPU.put(device)
    os.environ.update(ENV)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as ex: list(ex.map(prepare,['tdc_only','mixed']))
        with ThreadPoolExecutor(max_workers=4) as cpu:
            futures=[cpu.submit(fn,pool) for fn in [head_knn,forest] for pool in ['tdc_only','mixed']]
            jobs=[(pool,w,s) for w in [0,0.1,0.6,1] for s in range(1,6) for pool in ['tdc_only','mixed']]
            with ThreadPoolExecutor(max_workers=8*args.jobs_per_gpu) as ex: list(ex.map(train,jobs))
            for pool in ['tdc_only','mixed']:
                out=ROOT/pool
                call('baselines.minimol.run_multitask',['finalize','--prepared',out/'prepared','--output',out],
                     out/'finalize_job',out/'indirect/metrics.json',gpu=True)
            for future in futures: future.result()
        write(ROOT/'completion.json',{'status':'training_complete','time':time.time()})
    finally:
        STOP.set();thread.join()


if __name__ == "__main__":
    main()
