"""Backend policy tests."""

from __future__ import annotations

import pytest

from flagrand.runtime import supports_cached_binary_launch, uses_portable_f32_math

pytestmark = [pytest.mark.accuracy, pytest.mark.runtime]


def test_backend_policy_keeps_cuda_fast_path() -> None:
    assert supports_cached_binary_launch("cuda")
    assert not uses_portable_f32_math("cuda")


def test_backend_policy_uses_portable_maca_path() -> None:
    assert not supports_cached_binary_launch("maca")
    assert uses_portable_f32_math("maca")


def test_backend_policy_uses_portable_ascend_path() -> None:
    assert not supports_cached_binary_launch("ascend")
    assert uses_portable_f32_math("ascend")


def test_backend_policy_caches_npu_launch_with_portable_math() -> None:
    assert supports_cached_binary_launch("npu")
    assert uses_portable_f32_math("npu")


def test_backend_policy_caches_ppu_launch_with_portable_math() -> None:
    assert supports_cached_binary_launch("ppu")
    assert uses_portable_f32_math("ppu")
