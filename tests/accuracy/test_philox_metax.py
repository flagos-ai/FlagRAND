"""MetaX MACA dispatch checks for the public Philox API."""

from __future__ import annotations

import pytest
import torch
from triton.runtime import driver

from flagrand import curand
from flagrand.metax.launcher import MetaXKernelLauncher
from flagrand.runtime import current_backend, current_vendor

METAX_AVAILABLE = torch.cuda.is_available() and current_vendor() == "metax"

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.cuda,
    pytest.mark.metax,
    pytest.mark.skipif(not METAX_AVAILABLE, reason="MetaX MACA is required"),
]


@pytest.mark.raw32
@pytest.mark.parametrize("offset", [0, 4, 60, 64, 1025, 2**33-4, 2**33, 2**34+1])
def test_metax_generic_counter_carries_match_cpu_philox(offset: int) -> None:
    from benchmark.hygon_matrix.reference import words

    seed=0x1234567887654321
    actual=torch.empty(1025,device="cuda",dtype=torch.int32)
    curand.generate(curand.create_generator(seed=seed,offset=offset),actual)
    expected=torch.from_numpy(words(seed,offset,actual.numel()).view("int32").copy())
    assert torch.equal(actual.cpu(),expected)


@pytest.mark.raw32
def test_metax_backend_is_active() -> None:
    assert current_vendor() == "metax"
    assert current_backend() == "maca"
    assert driver.active.get_current_target().backend == "maca"


@pytest.mark.raw32
@pytest.mark.uniform_f32
@pytest.mark.uniform_f64
@pytest.mark.normal_f32
@pytest.mark.normal_f64
@pytest.mark.lognormal_f32
@pytest.mark.lognormal_f64
def test_metax_public_api_uses_vendor_dispatch() -> None:
    expected = "flagrand.runtime.backend._metax.ops"
    assert curand.try_generate_raw.__module__ == expected
    assert curand.try_generate_uniform.__module__ == expected
    assert curand.try_generate_normal.__module__ == expected
    for name in (
        "generate",
        "generate_uniform",
        "generate_uniform_double",
        "generate_normal",
        "generate_normal_double",
        "generate_lognormal",
        "generate_lognormal_double",
    ):
        assert getattr(curand, name).__module__ == expected


@pytest.mark.uniform_f32
def test_metax_public_api_uses_current_stream() -> None:
    expected = torch.empty(4096, device="cuda", dtype=torch.float32)
    actual = torch.empty_like(expected)
    curand.generate_uniform(curand.create_generator(seed=83), expected)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        curand.generate_uniform(curand.create_generator(seed=83), actual)
    stream.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.raw32
def test_metax_launcher_supports_jit_without_legacy_binder() -> None:
    calls = []

    class ModernJITFunction:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append((grid, args, kwargs))
                return "compiled"

            return run

    launcher = MetaXKernelLauncher(
        ModernJITFunction(),
        constexpr_names=("BLOCK",),
    )
    result = launcher.launch(
        3,
        ("output", 17),
        (128,),
        (4,),
    )

    assert result == "compiled"
    assert calls == [
        (
            (3, 1, 1),
            ("output", 17),
            {"BLOCK": 128, "num_warps": 4},
        )
    ]
