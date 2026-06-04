from __future__ import annotations

import importlib.metadata as metadata
import platform
import re
import shutil
import subprocess
import sys
import traceback


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _extract_nvidia_smi_cuda_version(output: str | None) -> str | None:
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def _extract_nvcc_cuda_version(output: str | None) -> str | None:
    if not output:
        return None
    match = re.search(r"release\s+([0-9.]+)", output)
    return match.group(1) if match else None


def _check_inductor(device: str) -> tuple[bool, str]:
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            return False, "CUDA requested for Inductor check, but torch.cuda.is_available() is false."
        resolved = torch.device(device)
        module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(resolved)
        sample = torch.randn(4, 8, device=resolved)
        compiled = torch.compile(module, mode="reduce-overhead")
        compiled(sample).sum().backward()
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - this is a diagnostics command.
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return False, detail


def run_doctor(*, expect_cuda: bool = False, check_inductor: bool = False, inductor_device: str = "cuda") -> int:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")
    print(f"numpy: {_version('numpy')}")
    print(f"pandas: {_version('pandas')}")
    print(f"ray: {_version('ray')}")
    print(f"triton: {_version('triton')}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on host environment
        print(f"torch: import failed: {exc}")
        return 1

    cuda_available = torch.cuda.is_available()
    print(f"torch: {torch.__version__} ({torch.__file__})")
    print(f"torch cuda runtime: {torch.version.cuda}")
    print(f"cuda available: {cuda_available}")
    print(f"cuda device count: {torch.cuda.device_count()}")
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            capability = ".".join(str(part) for part in torch.cuda.get_device_capability(index))
            memory_gb = props.total_memory / 1024**3
            print(f"gpu {index}: {props.name}, capability {capability}, {memory_gb:.1f} GiB")

    nvidia_smi = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    nvidia_smi_banner = _run(["nvidia-smi"])
    driver_cuda_version = _extract_nvidia_smi_cuda_version(nvidia_smi_banner)
    nvcc_version = _extract_nvcc_cuda_version(_run(["nvcc", "--version"]))
    if nvidia_smi:
        print("nvidia-smi:")
        for line in nvidia_smi.splitlines():
            print(f"  {line}")
    else:
        print("nvidia-smi: unavailable")
    print(f"nvidia-smi cuda version: {driver_cuda_version or 'unavailable'}")
    print(f"nvcc cuda version: {nvcc_version or 'unavailable'}")

    if expect_cuda and not cuda_available:
        print("error: CUDA was expected but torch.cuda.is_available() is false")
        return 1
    if check_inductor:
        import torch._dynamo

        from .runner import _configure_torch_dynamo_for_compile

        _configure_torch_dynamo_for_compile()
        print(f"torch._dynamo.config.cache_size_limit: {torch._dynamo.config.cache_size_limit}")
        accumulated = getattr(torch._dynamo.config, "accumulated_cache_size_limit", "unavailable")
        print(f"torch._dynamo.config.accumulated_cache_size_limit: {accumulated}")
        ok, detail = _check_inductor(inductor_device)
        print(f"inductor check ({inductor_device}): {'ok' if ok else 'failed'}")
        if not ok:
            print(f"inductor error: {detail}")
            return 1
    return 0
