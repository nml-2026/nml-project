from __future__ import annotations

import os

import pytest

from nba_forecasting.runtime_env import (
    AUTO_NUM_WORKERS,
    configure_training_cpu_runtime,
    cpus_per_slot,
    resolve_num_workers,
    training_slots,
)


def test_resolve_num_workers_explicit_and_disabled() -> None:
    assert resolve_num_workers(0, slots=1) == 0
    assert resolve_num_workers(4, slots=1, worker_cap=8) == 4


def test_resolve_num_workers_auto_scales_with_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.delenv("NBA_TRAINING_SLOTS", raising=False)
    assert resolve_num_workers(AUTO_NUM_WORKERS, slots=1) == 8
    assert resolve_num_workers(AUTO_NUM_WORKERS, slots=4) == 8
    assert resolve_num_workers(AUTO_NUM_WORKERS, slots=32) == 2


def test_training_slots_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setenv("NBA_TRAINING_SLOTS", "4")
    assert training_slots() == 4
    assert cpus_per_slot(4) == 32


def test_configure_training_cpu_runtime_sets_thread_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    runtime = configure_training_cpu_runtime(AUTO_NUM_WORKERS, slots=4)
    assert runtime.num_workers == 6
    assert runtime.omp_num_threads == 2
    assert os.environ["OMP_NUM_THREADS"] == "2"
