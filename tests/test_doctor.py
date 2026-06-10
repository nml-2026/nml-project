from __future__ import annotations

from nba_forecasting.doctor import _extract_nvcc_cuda_version, _extract_nvidia_smi_cuda_version


def test_extract_nvidia_smi_cuda_version_from_banner() -> None:
    output = "| NVIDIA-SMI 535.54.03 Driver Version: 535.54.03 CUDA Version: 12.2 |"
    assert _extract_nvidia_smi_cuda_version(output) == "12.2"


def test_extract_nvcc_cuda_version_from_release_line() -> None:
    output = "Cuda compilation tools, release 12.1, V12.1.105"
    assert _extract_nvcc_cuda_version(output) == "12.1"


def test_extract_cuda_versions_return_none_when_missing() -> None:
    assert _extract_nvidia_smi_cuda_version("no cuda field") is None
    assert _extract_nvcc_cuda_version("no release field") is None
