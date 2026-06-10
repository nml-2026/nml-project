from __future__ import annotations

import pytest

from nba_forecasting.runner.io import _format_duration, _progress_disabled, log_epoch_progress


def test_progress_disabled_when_multiple_training_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NBA_FORECASTING_DISABLE_TQDM", raising=False)
    monkeypatch.setenv("NBA_TRAINING_SLOTS", "1")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    assert _progress_disabled() is False

    monkeypatch.setenv("NBA_TRAINING_SLOTS", "4")
    assert _progress_disabled() is True


def test_format_duration() -> None:
    assert _format_duration(45) == "45s"
    assert _format_duration(125) == "2m05s"
    assert _format_duration(3725) == "1h02m"


def test_log_epoch_progress_when_parallel(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBA_TRAINING_SLOTS", "4")
    log_epoch_progress(
        experiment_name="graph_transformer",
        epoch=3,
        total_epochs=80,
        start_epoch=1,
        started_at=0.0,
        train_loss=0.0123,
        val_metric=2.5,
        best_epoch=2,
        best_metric=2.4,
        checkpoint_metric="ade",
        patience=1,
        patience_limit=80,
    )
    out = capsys.readouterr().out
    assert "epochs:graph_transformer 3/80" in out
    assert "val_ade=2.50" in out
    assert "eta=" in out


def test_log_epoch_progress_skipped_when_tqdm_enabled(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBA_TRAINING_SLOTS", "1")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    log_epoch_progress(
        experiment_name="graph_transformer",
        epoch=3,
        total_epochs=80,
        start_epoch=1,
        started_at=0.0,
        train_loss=0.0123,
        val_metric=2.5,
        best_epoch=2,
        best_metric=2.4,
        checkpoint_metric="ade",
        patience=1,
        patience_limit=80,
    )
    assert capsys.readouterr().out == ""


def test_progress_disabled_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBA_FORECASTING_DISABLE_TQDM", "1")
    monkeypatch.setenv("NBA_TRAINING_SLOTS", "1")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    assert _progress_disabled() is True
