from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agentad import SeriesData, SeriesItem
from agentad.benchmark import (
    DatasetSplit,
    RunMismatchError,
    atomic_write_file,
    atomic_write_json,
    completion_path,
    has_score,
    is_complete,
    prepare_run,
    save_score,
    score_path,
    scores_dir,
    unit_dir,
    write_metrics,
)
from agentad.benchmark.tsbad_reference import (
    TSB_AD_LEADERBOARD_COMMIT,
    leaderboard_hp,
    official_hp,
    resolve_protocol,
    validate_reference,
)


def _collection(item: SeriesItem) -> SeriesData:
    return SeriesData.from_items([item], features=pd.DataFrame({"name": ["x"]}))


def _split(tmp_path: Path) -> DatasetSplit:
    root = tmp_path / "data"
    dataset_dir = root / "source" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "artifact.train.zarr.zip").write_bytes(b"train")
    (dataset_dir / "artifact.test.zarr.zip").write_bytes(b"test")
    train = _collection(
        SeriesItem(
            "s",
            np.array([[1.0], [2.0]]),
            np.array([0, 1]),
            pd.DataFrame({"label": [0, 0]}),
        )
    )
    test = _collection(
        SeriesItem(
            "s",
            np.array([[3.0], [4.0], [5.0]]),
            np.array([2, 3, 4]),
            pd.DataFrame({"label": [0, 1, 1]}),
        )
    )
    return DatasetSplit(root, "source", "dataset", "artifact", train, test)


def _prepare(unit: Path, split: DatasetSplit, *, hp=None, resume=False):
    return prepare_run(
        unit,
        split,
        method="Toy",
        partition="Eva",
        seed=7,
        hp=hp,
        resume=resume,
        implementation_file=__file__,
        device="cpu",
    )


def test_run_fingerprint_isolates_score_generations(tmp_path):
    split = _split(tmp_path)
    unit = unit_dir(tmp_path / "outputs", "Toy", split)
    first = _prepare(unit, split)
    save_score(unit, "s", np.arange(5, dtype=float))
    assert has_score(unit, "s")
    assert score_path(unit, "s").parent.name == first["fingerprint"]

    with pytest.raises(RunMismatchError, match="run contract changed"):
        _prepare(unit, split, hp={"window": 3}, resume=True)

    second = _prepare(unit, split, hp={"window": 3}, resume=False)
    assert second["fingerprint"] != first["fingerprint"]
    assert not has_score(unit, "s")


def test_legacy_scores_cannot_be_silently_adopted(tmp_path):
    split = _split(tmp_path)
    unit = unit_dir(tmp_path / "outputs", "Toy", split)
    scores_dir(unit).mkdir(parents=True)
    np.save(scores_dir(unit) / "s.npy", np.arange(5, dtype=float))

    with pytest.raises(RunMismatchError, match="legacy outputs"):
        _prepare(unit, split, resume=True)


def test_metrics_publish_completion_only_for_a_full_run(tmp_path):
    split = _split(tmp_path)
    unit = unit_dir(tmp_path / "outputs", "Toy", split)
    results = unit_dir(tmp_path / "results", "Toy", split)
    _prepare(unit, split)
    with pytest.raises(ValueError, match="no score"):
        write_metrics(split, unit, results, metrics=("AUC-ROC", "Standard-F1"))
    assert not completion_path(results).exists()

    save_score(unit, "s", np.array([0.0, 0.1, 0.2, 0.8, 0.9]))
    frame = write_metrics(
        split,
        unit,
        results,
        metrics=("AUC-ROC", "Standard-F1"),
        sliding_window=0,
    )
    assert np.isnan(frame.loc[0, "score_seconds"])
    assert frame.loc[0, "score_load_seconds"] >= 0
    assert is_complete(unit, results)

    with (results / "metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert not is_complete(unit, results)


def test_atomic_binary_write_preserves_previous_target_on_failure(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"old")

    def fail(handle):
        handle.write(b"new")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        atomic_write_file(target, fail)
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))

    json_target = tmp_path / "artifact.json"
    atomic_write_json(json_target, {"ok": True})
    assert json_target.read_text(encoding="utf-8") == '{"ok":true}'


def test_tsb_ad_protocol_profiles_distinguish_legacy_and_submissions():
    pca_u = resolve_protocol("PCA", "U")
    pca_m = resolve_protocol("PCA", "M")
    time_rcd = resolve_protocol("Time-RCD", "U")
    tspulse = resolve_protocol("TSPulse (FT)", "M")

    assert pca_u.official_name == "Sub_PCA"
    assert pca_m.official_name == "PCA"
    assert pca_u.provenance == "tsb-ad-2024"
    assert time_rcd.official_name == "Time_RCD"
    assert time_rcd.score_scope == "test"
    assert time_rcd.provenance == "community-submission"
    assert tspulse.official_name == "TSPulse_FT"
    assert tspulse.score_scope == "full"


def test_tsb_ad_pinned_archive_and_hp_are_validated(tmp_path):
    reference = tmp_path / "TSB-AD"
    (reference / "TSB_AD").mkdir(parents=True)
    (reference / "benchmark_exp").mkdir()
    (reference / "TSB_AD" / "model_wrapper.py").write_text("", encoding="utf-8")
    (reference / "TSB_AD" / "HP_list.py").write_text(
        "Optimal_Uni_algo_HP_dict = {'Sub_PCA': {'periodicity': 1}}\n"
        "Optimal_Multi_algo_HP_dict = {'PCA': {'n_components': 0.25}}\n",
        encoding="utf-8",
    )
    (reference / "benchmark_exp" / "Run_Detector_U.py").write_text(
        "", encoding="utf-8"
    )
    (reference / "benchmark_exp" / "Run_Detector_M.py").write_text(
        "", encoding="utf-8"
    )
    (reference / ".agentad-reference.json").write_text(
        '{"commit":"' + TSB_AD_LEADERBOARD_COMMIT + '"}', encoding="utf-8"
    )

    identity = validate_reference(reference)
    assert identity["kind"] == "pinned-archive"
    assert identity["commit"] == TSB_AD_LEADERBOARD_COMMIT
    assert official_hp(reference, "U", "Sub_PCA") == {"periodicity": 1}
    assert leaderboard_hp(reference, "M", "TSPulse_ZS") == {
        "win_size": 96,
        "prediction_mode": "time",
    }
    assert leaderboard_hp(reference, "M", "Time_RCD")["win_size"] == (
        "min(10000, floor(400000 / n_features))"
    )


def test_write_metrics_accepts_per_series_vus_windows(tmp_path):
    split = _split(tmp_path)
    unit = unit_dir(tmp_path / "outputs", "Toy", split)
    results = unit_dir(tmp_path / "results", "Toy", split)
    _prepare(unit, split)
    save_score(unit, "s", np.array([0.0, 0.1, 0.2, 0.8, 0.9]))
    frame = write_metrics(
        split,
        unit,
        results,
        metrics=("VUS-PR",),
        sliding_window={"s": 2},
    )
    assert frame.loc[0, "sliding_window"] == 2


def test_write_metrics_can_evaluate_submission_test_suffix(tmp_path):
    split = _split(tmp_path)
    unit = unit_dir(tmp_path / "outputs", "SuffixToy", split)
    results = unit_dir(tmp_path / "results", "SuffixToy", split)
    prepare_run(
        unit,
        split,
        method="SuffixToy",
        partition="Eva",
        seed=7,
        hp={},
        resume=False,
        implementation_file=__file__,
    )
    save_score(unit, "s", np.array([0.0, 0.8, 0.9]))
    frame = write_metrics(
        split,
        unit,
        results,
        metrics=("VUS-PR",),
        sliding_window={"s": 2},
        score_scope="test",
    )
    assert frame.loc[0, "n_points"] == 3
    assert not frame.loc[0, "full_eval"]


def test_baseline_driver_batches_every_task_once():
    from tmp.run_all_baselines import _batches

    tasks = [
        ("module", "dataset", str(index), "Eva", "cpu", True) for index in range(29)
    ]
    batches = _batches(tasks, 12)
    assert [start for start, _ in batches] == [0, 12, 24]
    assert [task for _, batch in batches for task in batch] == tasks


def test_tspulse_finetune_moves_in_memory_model_before_scoring(tmp_path, monkeypatch):
    from agentad.methods.TSPulse import eval as tspulse_eval

    split = _split(tmp_path)
    moves: list[str] = []

    class FakeModel:
        def to(self, device):
            moves.append(str(device))
            return self

        def score(self, series, *, batch_size):
            assert moves
            return series.new_zeros(series.shape[:2])

    monkeypatch.setattr(tspulse_eval, "load_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(
        tspulse_eval,
        "_resolve_finetune_config",
        lambda *args, **kwargs: SimpleNamespace(
            context_length=2, finetune_batch_size=1
        ),
    )
    monkeypatch.setattr(tspulse_eval, "_load_finetuned", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tspulse_eval, "train_series", lambda *args, **kwargs: FakeModel()
    )

    frame = tspulse_eval.evaluate_finetune(
        tmp_path / "unused",
        output_root=tmp_path / "outputs",
        results_root=tmp_path / "results",
        device="cpu",
        resume=False,
    )
    assert len(frame) == 1
    assert moves == ["cpu"]
