from __future__ import annotations

import os
from functools import lru_cache

from triton.runtime.driver import driver


@lru_cache(maxsize=1)
def current_backend() -> str:
    target = driver.active.get_current_target()
    return str(getattr(target, "backend", "unknown"))


def force_portable_backend() -> bool:
    value = os.environ.get("FLAGRAND_FORCE_PORTABLE", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def supports_cached_binary_launch(backend: str | None = None) -> bool:
    if force_portable_backend():
        return False
    return (current_backend() if backend is None else backend) == "cuda"


def uses_portable_f32_math(backend: str | None = None) -> bool:
    return not supports_cached_binary_launch(backend)
