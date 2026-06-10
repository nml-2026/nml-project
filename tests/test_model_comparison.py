from __future__ import annotations

from pathlib import Path

import pytest

from nba_forecasting.model_comparison import (
    ALL_JOBS,
    DEFAULT_COMPARISON_ROOT,
    DEFAULT_CONNECTIVITY_ROOT,
    DEFAULT_SMOKE_ROOT,
    ComparisonJob,
    latest_run_dir,
    resolve_jobs,
)


def test_all_jobs_count_and_names() -> None:
    names = [job.name for job in ALL_JOBS]
    assert len(names) == 10
    assert names == [
        "baseline",
        "joint_gru",
        "static_graph",
        "dynamic_graph_autoregressive",
        "dynamic_graph_direct",
        "graph_transformer",
        "graph_transformer_mdn",
        "mart",
        "connectivity_no_graph",
        "connectivity_shuffled_graph",
    ]


def test_mart_epochs_and_connectivity_overrides() -> None:
    mart = next(job for job in ALL_JOBS if job.name == "mart")
    assert mart.epochs == 100
    assert mart.resolved_patience() == 100

    no_graph = next(job for job in ALL_JOBS if job.name == "connectivity_no_graph")
    assert no_graph.config == "graph_transformer"
    assert no_graph.name_suffix == "graph_ablation_no_graph"
    assert no_graph.run_folder_name() == "graph_transformer_graph_ablation_no_graph"
    assert no_graph.extra_overrides["model.graph_transformer.graph_ablation_mode"] == "no_graph"
    assert (
        no_graph.output_root(
            comparison_root=DEFAULT_COMPARISON_ROOT,
            connectivity_root=DEFAULT_CONNECTIVITY_ROOT,
            smoke=False,
            smoke_root=DEFAULT_SMOKE_ROOT,
        )
        == DEFAULT_CONNECTIVITY_ROOT
    )


def test_resolve_jobs_default_subset_and_skip() -> None:
    assert len(resolve_jobs()) == 10
    assert [job.name for job in resolve_jobs(jobs=["baseline", "mart"])] == ["baseline", "mart"]
    skipped = resolve_jobs(skip_jobs=["baseline", "joint_gru", "mart"])
    assert len(skipped) == 7
    assert "baseline" not in {job.name for job in skipped}


def test_resolve_jobs_single_job() -> None:
    jobs = resolve_jobs(job="graph_transformer")
    assert len(jobs) == 1
    assert jobs[0].name == "graph_transformer"


def test_resolve_jobs_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown job"):
        resolve_jobs(job="missing")
    with pytest.raises(ValueError, match="Unknown job"):
        resolve_jobs(jobs=["baseline", "missing"])


def test_mart_aug_jobs_optional_not_in_default_batch() -> None:
    default = {job.name for job in resolve_jobs()}
    assert "mart_aug_none" not in default
    aug = resolve_jobs(jobs=["mart_aug_none", "mart_aug_reflect", "mart_aug_perm"])
    assert [job.config for job in aug] == ["mart_aug_none", "mart_aug_reflect", "mart_aug_perm"]


def test_with_name_suffix_only_for_comparison_jobs() -> None:
    job = ComparisonJob(
        name="static_graph",
        report_label="Static graph MPNN",
        config="static_graph",
    )
    updated = job.with_name_suffix("suffix_a")
    assert updated.name_suffix == "suffix_a"

    connectivity = next(item for item in ALL_JOBS if item.name == "connectivity_no_graph")
    assert connectivity.with_name_suffix("ignored").name_suffix == "graph_ablation_no_graph"


def test_run_folder_name_is_unique_for_graph_transformer_variants() -> None:
    delta = next(job for job in ALL_JOBS if job.name == "graph_transformer")
    mdn = next(job for job in ALL_JOBS if job.name == "graph_transformer_mdn")
    assert delta.run_folder_name() == "graph_transformer"
    assert mdn.run_folder_name() == "graph_transformer_mdn"
    assert delta.run_folder_name() != mdn.run_folder_name()


def test_latest_run_dir_uses_exact_folder_name(tmp_path: Path) -> None:
    root = tmp_path / "model_comparison"
    delta_run = root / "graph_transformer" / "20260101_000000_000001"
    mdn_run = root / "graph_transformer_mdn" / "20260102_000000_000002"
    for run_dir in (delta_run, mdn_run):
        run_dir.mkdir(parents=True)
        (run_dir / "history.csv").write_text("epoch,val_ade\n1,1.0\n", encoding="utf-8")
        (run_dir / "best.pt").write_bytes(b"checkpoint")

    assert latest_run_dir(root, "graph_transformer") == delta_run
    assert latest_run_dir(root, "graph_transformer_mdn") == mdn_run


def test_latest_run_dir_prefers_complete_checkpoint(tmp_path: Path) -> None:
    import os
    import time

    root = tmp_path / "connectivity_ablations"
    folder = root / "graph_transformer_graph_ablation_no_graph"
    incomplete = folder / "20260610_000000_000001"
    complete = folder / "20260610_000000_000002"
    for run_dir in (incomplete, complete):
        run_dir.mkdir(parents=True)
        (run_dir / "history.csv").write_text("epoch\n1\n", encoding="utf-8")
    (complete / "best.pt").write_bytes(b"checkpoint")
    os.utime(incomplete, (time.time() + 10, time.time() + 10))

    assert latest_run_dir(root, "graph_transformer_graph_ablation_no_graph") == complete


def test_smoke_output_root_for_comparison_jobs() -> None:
    job = next(item for item in ALL_JOBS if item.name == "baseline")
    assert job.output_root(
        comparison_root=Path("runs/model_comparison"),
        connectivity_root=Path("runs/connectivity_ablations"),
        smoke=True,
        smoke_root=Path("runs/model_comparison_smoke"),
    ) == Path("runs/model_comparison_smoke")
