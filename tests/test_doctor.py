from __future__ import annotations

import unittest

from nba_forecasting.doctor import _extract_nvcc_cuda_version, _extract_nvidia_smi_cuda_version


class DoctorTests(unittest.TestCase):
    def test_extract_nvidia_smi_cuda_version_from_banner(self) -> None:
        output = "| NVIDIA-SMI 535.54.03 Driver Version: 535.54.03 CUDA Version: 12.2 |"
        self.assertEqual(_extract_nvidia_smi_cuda_version(output), "12.2")

    def test_extract_nvcc_cuda_version_from_release_line(self) -> None:
        output = "Cuda compilation tools, release 12.1, V12.1.105"
        self.assertEqual(_extract_nvcc_cuda_version(output), "12.1")

    def test_extract_cuda_versions_return_none_when_missing(self) -> None:
        self.assertIsNone(_extract_nvidia_smi_cuda_version("no cuda field"))
        self.assertIsNone(_extract_nvcc_cuda_version("no release field"))


if __name__ == "__main__":
    unittest.main()
