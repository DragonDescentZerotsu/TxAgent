import json
from argparse import Namespace

import numpy as np
import pytest

from predict.baselines import repeated_runs as runs
from predict.baselines.foundation_suite import claim_job, jobs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_threshold_matches_original_search_including_ties(dtype):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(19)
    for size in (1, 2, 10, 51):
        labels = rng.integers(0, 2, size).tolist()
        values = np.round(rng.random(size), 2).astype(dtype)
        if size > 2:
            values[:3] = [.5, .5, np.nextafter(dtype(.5), dtype(1.))]
        candidates = sorted(set(float(v) for v in values))
        thresholds = [.5, *candidates, *((a+b)/2 for a, b in zip(candidates, candidates[1:]))]
        expected = min([(float(v), float(f1_score(labels, values >= v, average="macro"))) for v in thresholds],
                       key=lambda pair: (-pair[1], abs(pair[0]-.5)))
        assert runs.choose_threshold(labels, values, "valid_macro_f1") == expected


def test_seed_selects_own_lambda_and_threshold():
    labels = np.array([0., 1., 0., 1.])
    good, poor = np.array([.1, .8, .2, .9]), np.array([.7, .4, .6, .3])
    first = {.1: dict(scores=good), .6: dict(scores=poor)}
    second = {.1: dict(scores=poor), .6: dict(scores=good)}
    assert runs.select_member(first, labels)[0] == .1
    assert runs.select_member(second, labels)[0] == .6
    assert runs.select_member(first, labels)[1] == runs.choose_threshold(labels.tolist(), good, "valid_macro_f1")[0]


def test_sample_sd_and_small_sample_t_interval():
    result = runs.statistics([.6, .7, .8, .9, 1.])
    assert result["mean"] == pytest.approx(.8)
    assert result["sd"] == pytest.approx(np.sqrt(.025))
    assert result["ci95_high"] - .8 == pytest.approx(2.776445105 * np.sqrt(.025 / 5))
    with pytest.raises(ValueError):
        runs.statistics([.8])


def test_task_lock_excludes_a_second_controller(tmp_path):
    task = dict(id="one")
    first = claim_job(tmp_path, task)
    assert first is not None and claim_job(tmp_path, task) is None
    first.close()
    second = claim_job(tmp_path, task)
    assert second is not None
    second.close()


def test_remote_owner_retains_staged_work_until_publication(tmp_path):
    task = dict(id="staged")
    handle = claim_job(tmp_path, task)
    handle.write(json.dumps(dict(host="another-compute-node", state="running")))
    handle.flush()
    handle.close()
    assert claim_job(tmp_path, task) is None


def test_repeated_summary_audits_selection_and_preserves_ensemble(tmp_path):
    labels = [0, 1, 0, 1]
    rows = {split: [dict(index=i, drug=f"molecule_{i}", label=y, condition_group="none")
                   for i, y in enumerate(labels)] for split in ("valid", "test")}
    folder = tmp_path / "model/pool/task/indirect"
    for seed in runs.PROTOCOL["seeds"]:
        scores = np.array([.1, .7, .3, .9]) + seed * .001
        threshold, _ = runs.choose_threshold(labels, scores, "valid_macro_f1")
        runs.save_evaluation(folder / f"seed_{seed}", rows, dict(valid=scores, test=scores), seed=seed,
            weight=.1, threshold=threshold, candidates={.1: runs.training.ensemble_bce(np.array(labels), scores), .6: 1.},
            provenance=dict(precision="fp32"))
    runs.summarize(tmp_path, 1)
    result = json.loads((tmp_path / "results.json").read_text())[0]
    assert result["macro_f1_mean"] == 1. and result["macro_f1_sd"] == 0.
    with pytest.raises(ValueError, match="Duplicate result cell"):
        runs.summarize(tmp_path, 1, sources=[tmp_path, tmp_path])
    predictions = folder / "seed_2/test_predictions.jsonl"
    original = predictions.read_text()
    predictions.write_text(original.replace("molecule_0", "different_molecule"))
    receipt = folder / "seed_2/metrics.json"
    record = json.loads(receipt.read_text())
    record["prediction_sha256"]["test"] = runs.sha256_file(predictions)
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Different evaluation cohort"):
        runs.summarize(tmp_path, 1)
    predictions.write_text(original)
    record["prediction_sha256"]["test"] = runs.sha256_file(predictions)
    receipt.write_text(json.dumps(record))
    path = folder / "seed_1/metrics.json"
    changed = json.loads(path.read_text()); changed["selected_aux_weight"] = .6
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Lambda selection"):
        runs.summarize(tmp_path, 1)


def test_forest_replay_selects_each_seed_before_test_inference(tmp_path, monkeypatch):
    from predict.baselines import retrieval_forest as forest
    import joblib
    labels = [0, 1, 0, 1]
    good, poor = np.array([.1, .8, .2, .9]), np.array([.7, .4, .6, .3])
    rows = {s: [dict(drug=str(i), Y=y) for i, y in enumerate(labels)] for s in ("valid", "test")}
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}")
    monkeypatch.setattr(forest, "load_features", lambda *a, **k: ({}, dict(valid="valid", test="test"), rows))
    candidates = []
    probabilities = {}
    for name in ("a", "b"):
        scores = [good if (seed == 1) == (name == "a") else poor for seed in range(1, 6)]
        members = [dict(seed=seed, path=f"{name}/{seed}") for seed in range(1, 6)]
        probabilities.update({m["path"]: p for m, p in zip(members, scores)})
        candidates.append((0., name, dict(members=members), np.array(scores)))
    monkeypatch.setattr(forest, "validated_candidates", lambda *a: candidates)
    inferred = []
    class SavedForest:
        def __init__(self, path):
            self.path = path
        def predict_proba(self, split):
            inferred.append((self.path, split))
            p = probabilities[self.path]
            return np.column_stack((1-p, p))
    monkeypatch.setattr(joblib, "load", SavedForest)
    for seed, expected in ((1, "a"), (2, "b")):
        output = tmp_path / f"seed_{seed}"
        runs.legacy_forest(Namespace(output=output, prepared=prepared, policy=0,
            feature_set="direct", data_dir=None, training_root=tmp_path, seed=seed))
        record = json.loads((output / "metrics.json").read_text())
        assert record["provenance"]["selected_candidate"] == expected
        assert record["protocol"] == runs.FOREST_PROTOCOL
        assert inferred[-2:] == [(f"{expected}/{seed}", "valid"), (f"{expected}/{seed}", "test")]


def test_precision_and_evaluation_jobs_have_distinct_dependencies(tmp_path):
    cells = [dict(input=f"inputs/{i}", training_pool="pool", task=str(i), hidden_dim=512, depth=3, lr=1e-4)
             for i in range(2)]
    manifest = dict(cells=cells, epochs=25, seeds=[1, 2, 3, 4, 5], auxiliary_weights=[.1, .6, 1.])
    args = Namespace(models=["unimol"], output=tmp_path, runtime_root=tmp_path, scratch_root=tmp_path / "ssd",
                     workers=16, conformer_workers=64, repeated_runs=True, unimol_precision="bf16")
    tasks = jobs(args, manifest)
    second = [j for j in tasks if j["id"].startswith("runs/unimol/pool/1/trainable_bf16/aux")]
    assert len(second) == 20 and all("bf16" in j["command"] for j in second)
    evaluation = next(j for j in tasks if j["id"] == "repeated_runs/unimol/pool/1/indirect/seed_3")
    assert len(evaluation["dependencies"]) == 3
    assert all(d.endswith("seed_3") and "trainable_bf16" in d for d in evaluation["dependencies"])


def test_fixed_auxiliary_reuses_exact_fits_and_changes_evaluation_identity(tmp_path):
    import copy
    cell = dict(input="inputs/pool/task", training_pool="pool", task="task", hidden_dim=512, depth=3, lr=1e-4)
    manifest = dict(cells=[cell], epochs=25, seeds=[1, 2, 3, 4, 5], auxiliary_weights=[.1, .6, 1.])
    args = Namespace(models=["unimol", "grover"], output=tmp_path / "old", runtime_root=tmp_path,
                     scratch_root=tmp_path / "ssd", workers=16, conformer_workers=64,
                     repeated_runs=True, unimol_precision="bf16")
    old = {j["id"]: j for j in jobs(args, manifest)}
    policy = dict(schema="minimol_fixed_auxiliary.v1", test_used=False, seeds=[1, 2, 3, 4, 5],
                  selection="MiniMol validation mean", cells={"pool/task": dict(weight=.6,
                      mean_validation_bce={"0.1": .4, "0.6": .3, "1.0": .5})})
    manifest.update(artifact_root=str(args.output), auxiliary_policy=policy)
    args.output = tmp_path / "new"
    new = jobs(args, manifest)
    fits = [j for j in new if j["marker"].name == "training.json"]
    assert len(fits) == 30
    assert all(j["command"] == old[j["id"]]["command"] and j["marker"] == old[j["id"]]["marker"] for j in fits)
    assert not any(j["id"].endswith("/finalize") for j in new)
    for j in new:
        if "/indirect/seed_" in j["id"]:
            assert len(j["dependencies"]) == 1 and "/aux_0.6/" in j["dependencies"][0]
            assert "--auxiliary-policy" in j["command"] and j["marker"].is_relative_to(args.output)
    assert runs.fixed_auxiliary_protocol(policy, "pool/task")["fixed_auxiliary_weight"] == .6
    wrong = copy.deepcopy(policy)
    wrong["cells"]["pool/task"]["weight"] = .1
    with pytest.raises(ValueError, match="Invalid fixed"):
        runs.fixed_auxiliary_protocol(wrong, "pool/task")


def test_fixed_policy_uses_mean_validation_loss_not_vote_or_test_score(tmp_path):
    for task in range(18):
        for seed in range(1, 6):
            path = tmp_path / f"minimol/pool/task_{task}/indirect/seed_{seed}/metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(protocol=runs.PROTOCOL, seed=seed,
                candidate_validation_bce={"0.1": .2 if seed < 4 else .9, "0.6": .3, "1.0": .8},
                test={"macro_f1": 1.})))
    output = tmp_path / "policy.json"
    runs.select_auxiliary_policy(tmp_path, output)
    policy = json.loads(output.read_text())
    assert all(c["weight"] == .6 for c in policy["cells"].values())
    assert all(len(c["sources"]) == 5 for c in policy["cells"].values())
    assert not policy["test_used"]


def test_saturated_probabilities_require_hash_bound_validation_replay(tmp_path):
    import torch
    from predict.utils.json import sha256_file
    labels = np.array([0., 1.])
    logits = torch.tensor([20., 0.])
    scores = logits.sigmoid().numpy()
    loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(labels).float()))
    np.save(tmp_path / "valid_scores.npy", scores)
    checkpoint = dict(args=dict(seed=1, aux_weight=0., epochs=1), epoch=1,
                      prepared_sha256="input", valid_bce=loss,
                      valid_scores_sha256=sha256_file(tmp_path / "valid_scores.npy"))
    for hashed in [True, False]:
        if not hashed:
            checkpoint.pop("valid_scores_sha256")
        torch.save(checkpoint, tmp_path / "best.pt")
        (tmp_path / "training.json").write_text(json.dumps(dict(test_used=False, best_epoch=1,
            checkpoint_sha256=sha256_file(tmp_path / "best.pt"), history=[dict(epoch=1, valid_bce=loss)])))
        kwargs = dict(prepared_sha256="input", seed=1, weight=0., labels=labels)
        with pytest.raises(ValueError, match="Validation scores"):
            runs.training.validated_member(tmp_path, **kwargs)
        if hashed:
            member = runs.training.validated_member(tmp_path, allow_bce_replay=True, **kwargs)
            assert member["validation_replay_required"] and member["checkpoint_validation_bce"] == loss
        else:
            with pytest.raises(ValueError, match="Validation scores"):
                runs.training.validated_member(tmp_path, allow_bce_replay=True, **kwargs)
