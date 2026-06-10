"""Resolve DataLoader workers and BLAS thread counts from host CPU budget."""

from __future__ import annotations

import os
from dataclasses import dataclass

AUTO_NUM_WORKERS = -1
DEFAULT_WORKER_CAP = 8
DEFAULT_OMP_CAP = 8
_TRAINING_SLOTS_ENV = "NBA_TRAINING_SLOTS"


@dataclass(frozen=True)
class TrainingCpuRuntime:
    requested_num_workers: int
    num_workers: int
    training_slots: int
    cpus_per_slot: int
    omp_num_threads: int


def training_slots() -> int:
    raw = os.environ.get(_TRAINING_SLOTS_ENV, "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def cpus_per_slot(slots: int | None = None) -> int:
    slots = max(1, slots or training_slots())
    return max(1, (os.cpu_count() or 1) // slots)


def resolve_num_workers(
    requested: int,
    *,
    slots: int | None = None,
    worker_cap: int = DEFAULT_WORKER_CAP,
) -> int:
    """Map config ``training.num_workers`` to a concrete worker count.

    - ``> 0``: explicit worker count (capped per slot)
    - ``0``: disable workers (tests / debug)
    - ``< 0`` (default ``-1``): auto from CPU budget
    """
    if requested == 0:
        return 0
    per_slot = cpus_per_slot(slots)
    if requested > 0:
        return max(0, min(requested, per_slot - 1))
    if per_slot <= 2:
        return 1
    return max(1, min(worker_cap, per_slot - 2))


def configure_training_cpu_runtime(
    requested_num_workers: int,
    *,
    slots: int | None = None,
    worker_cap: int = DEFAULT_WORKER_CAP,
    omp_cap: int = DEFAULT_OMP_CAP,
) -> TrainingCpuRuntime:
    slots = max(1, slots or training_slots())
    per_slot = cpus_per_slot(slots)
    workers = resolve_num_workers(requested_num_workers, slots=slots, worker_cap=worker_cap)
    if workers > 0:
        omp = min(omp_cap, max(1, per_slot - workers))
    else:
        omp = min(omp_cap, max(1, per_slot // 2))

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(omp))
    try:
        import torch

        torch.set_num_threads(omp)
    except ImportError:
        pass

    return TrainingCpuRuntime(
        requested_num_workers=requested_num_workers,
        num_workers=workers,
        training_slots=slots,
        cpus_per_slot=per_slot,
        omp_num_threads=omp,
    )


def dataloader_worker_kwargs(num_workers: int) -> dict[str, int | bool]:
    if num_workers <= 0:
        return {}
    return {"persistent_workers": True, "prefetch_factor": 2}


def format_training_cpu_runtime(runtime: TrainingCpuRuntime) -> str:
    requested = "auto" if runtime.requested_num_workers < 0 else str(runtime.requested_num_workers)
    return (
        f"DataLoader workers={runtime.num_workers} (config={requested}), "
        f"training_slots={runtime.training_slots}, cpus_per_slot={runtime.cpus_per_slot}, "
        f"omp_num_threads={runtime.omp_num_threads}"
    )
