"""The new index-based encoders retain the shared training and resume contract."""

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from predict.baselines.minimol import run_multitask as runner
from predict.utils.json import sha256_file


@pytest.mark.parametrize("frozen", [False, True])
def test_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch, frozen):
    from predict.baselines import foundation_models

    class Encoder(nn.Module):
        output_dim = 3

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.table = nn.Embedding(12, 3)
            self.table.weight.requires_grad_(not frozen)

        def forward(self, indices):
            return self.table(indices)

    monkeypatch.setattr(foundation_models, "FoundationEncoder", Encoder)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = dict(indices=torch.arange(8), labels=torch.tensor([0., 1.] * 4),
                conditions=torch.eye(2).repeat(4, 1))
    data = dict(smiles=[str(i) for i in range(12)], conditions=["a", "b"], buckets=[{}, {}],
                splits=dict(train=rows, valid=rows, test=rows),
                auxiliary=dict(indices=torch.arange(8, 12), buckets=torch.tensor([0, 1, 0, 1]),
                               targets=torch.tensor([0., 2., 1., -1.]), binary=torch.tensor([True, False] * 2)))
    torch.save(data, prepared / "data.pt")
    (prepared / "manifest.json").write_text(json.dumps(dict(data_sha256=sha256_file(prepared / "data.pt"))))
    (prepared / "receipt.json").write_text("{}")
    args = Namespace(prepared=prepared, encoder="fake", runtime_root=tmp_path, feature_cache=prepared,
                     frozen=frozen, hidden_dim=8, depth=3, device="cpu", seed=2, epochs=6,
                     batch_size=4, eval_batch_size=4, workers=0, threads=1,
                     aux_weight=0 if frozen else .6, lr=1e-3, encoder_lr=1e-3, output=tmp_path / "full")
    runner.train(args)
    reference = torch.load(args.output / "best.pt", weights_only=False)
    args.output = tmp_path / "resumed"
    write = runner.write_json_atomic

    def interrupted(path, payload):
        write(path, payload)
        if path.name == "progress.json" and len(payload["history"]) == 2:
            raise InterruptedError("simulated process interruption")

    monkeypatch.setattr(runner, "write_json_atomic", interrupted)
    with pytest.raises(InterruptedError):
        runner.train(args)
    monkeypatch.setattr(runner, "write_json_atomic", write)
    runner.train(args)
    resumed = torch.load(args.output / "best.pt", weights_only=False)
    assert reference["epoch"] == resumed["epoch"]
    assert reference["valid_bce"] == resumed["valid_bce"]
    for key, value in reference["state_dict"].items():
        torch.testing.assert_close(value, resumed["state_dict"][key], rtol=0, atol=0)
    receipt = json.loads((args.output / "training.json").read_text())
    assert bool(receipt["gradient_check"]) != frozen


def test_checkpoint_rejects_wrong_encoder_even_with_same_data(tmp_path):
    import numpy as np

    labels = np.array([0., 1.])
    scores = np.array([.2, .8])
    np.save(tmp_path / "valid_scores.npy", scores)
    torch.save(dict(args=dict(seed=1, aux_weight=0, epochs=1, encoder="grover", frozen=False),
                    binding=dict(features_sha256="cache"), epoch=1, prepared_sha256="input",
                    valid_bce=runner.ensemble_bce(labels, scores)), tmp_path / "best.pt")
    (tmp_path / "training.json").write_text(json.dumps(dict(
        checkpoint_sha256=sha256_file(tmp_path / "best.pt"), test_used=False,
        best_epoch=1, history=[dict(epoch=1, valid_bce=.2)])))
    with pytest.raises(ValueError, match="encoder/feature/setting"):
        runner.validated_member(tmp_path, prepared_sha256="input", seed=1, weight=0, labels=labels,
                                features_sha256="cache", encoder="kermt", frozen=False)


def test_result_audit_recomputes_ensemble_and_rejects_changed_conditions(tmp_path):
    import numpy as np
    from predict.baselines import foundation_suite as suite

    cell = dict(input="inputs", training_pool="conditioned_train", task="ames", benchmark="gold")
    manifest = dict(models={"chemberta": {}}, cells=[cell], settings=["frozen"],
                    seeds=[1, 2, 3, 4, 5], epochs=1, auxiliary_weights=[.1, .6, 1.])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    prepared = tmp_path / "inputs"
    prepared.mkdir()
    labels, scores = np.array([0., 1.]), np.array([.2, .8])
    rows = dict(indices=torch.arange(2), labels=torch.tensor(labels), condition_groups=["a", "b"])
    torch.save(dict(smiles=["CCO", "CCN"], splits=dict(valid=rows, test=rows)), prepared / "data.pt")
    root = tmp_path / "runs/chemberta/conditioned_train/ames/frozen"
    for seed in manifest["seeds"]:
        member = root / "aux_0" / f"seed_{seed}"
        member.mkdir(parents=True)
        loss = runner.ensemble_bce(labels, scores)
        np.save(member / "valid_scores.npy", scores)
        (member / "training.json").write_text(json.dumps(dict(test_used=False, best_epoch=1,
                                                             history=[dict(epoch=1, valid_bce=loss)])))
    out = root / "frozen"
    out.mkdir()
    threshold, _ = suite.choose_threshold(labels.tolist(), scores, "valid_macro_f1")
    metrics = dict(aux_weight=0, threshold=threshold,
                   selection=dict(test_used=False, seeds=manifest["seeds"], selected_aux_weight=0),
                   prediction_sha256={})
    for split in ("valid", "test"):
        predictions = [dict(index=i, drug=s, condition_group=c, label=int(y), probability=float(p),
                            prediction=int(p >= threshold))
                       for i, (s, c, y, p) in enumerate(zip(["CCO", "CCN"], ["a", "b"], labels, scores))]
        file = out / f"{split}_predictions.jsonl"
        file.write_text("".join(json.dumps(r) + "\n" for r in predictions))
        metrics["prediction_sha256"][split] = sha256_file(file)
        metrics[split] = suite.evaluate_metrics(labels.tolist(), scores, threshold)
    (out / "metrics.json").write_text(json.dumps(metrics))
    suite.verify(Namespace(output=tmp_path))
    assert len(json.loads((tmp_path / "results.json").read_text())) == 1
    assert not json.loads((tmp_path / "verification.json").read_text())["complete"]
    file = out / "test_predictions.jsonl"
    file.write_text(file.read_text().replace('"condition_group": "a"', '"condition_group": "wrong"'))
    metrics["prediction_sha256"]["test"] = sha256_file(file)
    (out / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="condition identity"):
        suite.verify(Namespace(output=tmp_path))


def test_full_matrix_has_all_fits_and_resolvable_dependencies(tmp_path):
    from predict.baselines import foundation_suite as suite
    from predict.baselines.foundation_models import MODELS

    source = Path(__file__).resolve().parents[3] / "experiments/baselines/foundation_sources.json"
    cells = [dict(c, input=f"inputs/{c['training_pool']}/{c['task']}", hidden_dim=512, depth=3, lr=1e-4)
             for c in json.loads(source.read_text())["datasets"]]
    manifest = dict(cells=cells, epochs=25, seeds=[1, 2, 3, 4, 5], auxiliary_weights=[.1, .6, 1.])
    args = Namespace(models=MODELS, output=tmp_path, runtime_root=tmp_path, scratch_root=tmp_path / "ssd",
                     workers=16, conformer_workers=64)
    jobs = suite.jobs(args, manifest)
    ids = {j["id"] for j in jobs}
    assert len(ids) == len(jobs) == 3528
    assert sum(j["marker"].name == "training.json" for j in jobs) == 3150
    assert sum(bool(j.get("exclusive")) for j in jobs) == 18 * 21
    assert all(set(j["dependencies"]) <= ids for j in jobs)
    assert {j["id"].split("/")[1] for j in jobs[:7]} == set(MODELS)


def test_ssd_publication_retains_conflicting_completed_work(tmp_path):
    from predict.baselines.foundation_suite import publish_fit

    work, archive = tmp_path / "ssd", tmp_path / "archive"
    work.mkdir()
    (work / "best.pt").write_bytes(b"checkpoint")
    (work / "training.json").write_text('{"checkpoint_sha256": "one"}')
    job = dict(work=work, marker=archive / "training.json")
    publish_fit(job)
    assert not work.exists() and (archive / "best.pt").read_bytes() == b"checkpoint"
    work.mkdir()
    (work / "training.json").write_text('{"checkpoint_sha256": "different"}')
    with pytest.raises(ValueError, match="Conflicting completed"):
        publish_fit(job)
    assert work.exists()


def test_local_input_cache_checks_publication_and_detects_changes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from predict.baselines.foundation_cache import cached_input

    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable scientific input")
    digest = sha256_file(source)
    monkeypatch.setenv("TXAGENT_FOUNDATION_CACHE", str(tmp_path / "local"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        copies = list(pool.map(lambda _: cached_input(source, digest), range(4)))
    assert len(set(copies)) == 1 and copies[0].read_bytes() == source.read_bytes()
    copies[0].chmod(0o644)
    copies[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cached input changed"):
        cached_input(source, digest)
    with pytest.raises(ValueError, match="changed during local caching"):
        cached_input(source, "wrong-checksum")


def test_prefetch_changes_dependencies_not_training_commands(tmp_path):
    from predict.baselines import foundation_suite as suite
    from predict.baselines.foundation_models import MODELS

    cells = [dict(input="inputs/a", training_pool="conditioned_train", task="ames", hidden_dim=512, depth=3, lr=1e-4)]
    manifest = dict(cells=cells, epochs=25, seeds=[1, 2, 3, 4, 5], auxiliary_weights=[.1, .6, 1.])
    args = Namespace(models=MODELS, output=tmp_path, runtime_root=tmp_path, scratch_root=tmp_path / "ssd",
                     workers=16, conformer_workers=96)
    original = {j["id"]: j for j in suite.jobs(args, manifest)}
    args.prefetch_features = True
    updated = {j["id"]: j for j in suite.jobs(args, manifest)}
    assert len(updated) == len(original) + 7
    for key, job in original.items():
        assert job["command"] == updated[key]["command"]
        assert job["marker"] == updated[key]["marker"]
        assert set(updated[key]["dependencies"]) <= set(updated)
    preparation = updated["featurize/unimol/conditioned_train/ames"]
    assert preparation["cpu"] and preparation["preprocessing"]
    assert preparation["command"][preparation["command"].index("--workers") + 1] == "32"
    heavy = dict(id="runs/unimol/pool/task/trainable_bf16/aux_0/seed_1")
    light = dict(id="runs/grover/pool/task/trainable/aux_0/seed_1")
    assert suite.gpu_slots(heavy, 4) == 2
    assert suite.gpu_slots(light, 4) == 1
    assert suite.gpu_slots(dict(heavy, exclusive=True), 4) == 4


def test_background_audit_does_not_wait_for_other_controller(tmp_path):
    import fcntl
    from predict.baselines.foundation_suite import audit_reports

    with (tmp_path / "verify.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert audit_reports(Namespace(output=tmp_path), blocking=False) is False


def test_cpu_feature_stage_matches_combined_embedding_path(tmp_path, monkeypatch):
    from predict.baselines import foundation_models as module

    calls = []

    class Encoder(nn.Module):
        output_dim = 2

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.tokenizer = lambda s: {"input_ids": [1, 2]}

        def make_feature(self, smiles):
            return [len(smiles), sum(map(ord, smiles))]

        def forward(self, indices):
            calls.append(indices.tolist())
            return torch.tensor([self.features[i] for i in indices], dtype=torch.float32)

    monkeypatch.setattr(module, "FoundationEncoder", Encoder)
    monkeypatch.delenv("TXAGENT_FOUNDATION_CACHE", raising=False)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    torch.save(dict(smiles=["CC", "CO", "CN"], splits={"train": dict(indices=torch.tensor([0, 1])),
                                                       "test": dict(indices=torch.tensor([2]))}), prepared / "data.pt")
    (prepared / "manifest.json").write_text(json.dumps(dict(data_sha256=sha256_file(prepared / "data.pt"))))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(models={"chemberta": {}})))
    combined, staged = tmp_path / "combined", tmp_path / "staged"
    module.prepare_features("chemberta", tmp_path, prepared, combined, device="cpu")
    calls.clear()
    module.prepare_features("chemberta", tmp_path, prepared, staged, device="cpu", features_only=True)
    assert not calls and not (staged / "embeddings.pt").exists()
    module.prepare_features("chemberta", tmp_path, prepared, staged, device="cpu")
    assert calls
    a = torch.load(combined / "embeddings.pt", weights_only=False)
    b = torch.load(staged / "embeddings.pt", weights_only=False)
    torch.testing.assert_close(a["embeddings"], b["embeddings"], rtol=0, atol=0)
    assert a["smiles"] == b["smiles"] and a["main_indices"] == b["main_indices"]
