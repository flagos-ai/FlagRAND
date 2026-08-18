from __future__ import annotations

import os
from pathlib import Path

import torch

from benchmark.ppu_matrix.curand import PpuCurandGenerator


def corex_curand_library_path() -> str:
    candidates = [
        os.environ.get("FLAGRAND_COREX_CURAND_PATH"),
        os.environ.get("CURAND_LIBRARY_PATH"),
        "/usr/local/corex/lib64/libcurand.so",
        "/usr/local/corex-4.4.0/lib64/libcurand.so",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("CoreX libcurand.so was not found.")


class CorexCurandGenerator:
    def __init__(self, *, seed: int) -> None:
        self._generator = PpuCurandGenerator(
            seed=seed,
            library_path=corex_curand_library_path(),
        )

    def generate(self, distribution: str, output: torch.Tensor) -> None:
        self._generator.generate(distribution, output)

    def close(self) -> None:
        self._generator.close()

    def __enter__(self) -> CorexCurandGenerator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
