from __future__ import annotations

import os
from importlib.metadata import version
from functools import lru_cache

from triton.runtime.driver import driver


@lru_cache(maxsize=1)
def current_backend() -> str:
    target = driver.active.get_current_target()
    backend = str(getattr(target, "backend", "unknown"))
    if backend == "cuda" and "corex" in version("triton"):
        backend = "corex"
    if backend == "ppu":
        from flagrand.runtime.ppu_compiler import install_ppu_compiler_compatibility

        install_ppu_compiler_compatibility()
    return backend


def force_portable_backend() -> bool:
    value = os.environ.get("FLAGRAND_FORCE_PORTABLE", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def supports_cached_binary_launch(backend: str | None = None) -> bool:
    if force_portable_backend():
        return False
    selected = current_backend() if backend is None else backend
    return selected in {"cuda", "ppu", "npu"}


def uses_portable_f32_math(backend: str | None = None) -> bool:
    if force_portable_backend():
        return True
    selected = current_backend() if backend is None else backend
    return selected != "cuda"
