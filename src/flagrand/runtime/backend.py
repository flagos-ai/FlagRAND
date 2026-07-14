from __future__ import annotations

from functools import lru_cache

from triton.runtime.driver import driver


@lru_cache(maxsize=1)
def current_backend() -> str:
    target = driver.active.get_current_target()
    return str(getattr(target, "backend", "unknown"))


def supports_cached_binary_launch(backend: str | None = None) -> bool:
    return (current_backend() if backend is None else backend) == "cuda"


def uses_portable_f32_math(backend: str | None = None) -> bool:
    return not supports_cached_binary_launch(backend)
