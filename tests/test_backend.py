from __future__ import annotations

from flagrand.runtime import supports_cached_binary_launch, uses_portable_f32_math


def test_backend_policy_keeps_cuda_fast_path() -> None:
    assert supports_cached_binary_launch("cuda")
    assert not uses_portable_f32_math("cuda")


def test_backend_policy_uses_portable_maca_path() -> None:
    assert not supports_cached_binary_launch("maca")
    assert uses_portable_f32_math("maca")
