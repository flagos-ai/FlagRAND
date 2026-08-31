"""NVIDIA cuRAND compatibility tests for the public Philox API."""

from __future__ import annotations

import pytest
import torch

from benchmark.nvidia_matrix.curand import NvidiaCurandGenerator
from flagrand import curand
from flagrand.runtime import current_backend, current_vendor

NVIDIA_AVAILABLE = torch.cuda.is_available() and current_vendor() == "nvidia"

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.cuda,
    pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA is required"),
]


@pytest.mark.raw32
def test_nvidia_backend_is_active() -> None:
    assert current_vendor() == "nvidia"
    assert current_backend() == "cuda"
    properties = torch.cuda.get_device_properties(0)
    assert properties.major >= 8


@pytest.mark.raw32
@pytest.mark.parametrize("seed", [0, 1, 17, (1 << 40) + 29])
@pytest.mark.parametrize("offset", range(8))
def test_raw32_matches_native_curand(seed: int, offset: int) -> None:
    expected = torch.empty(37, device="cuda", dtype=torch.int32)
    actual = torch.empty_like(expected)
    with NvidiaCurandGenerator(seed=seed, offset=offset) as generator:
        generator.generate("raw32", expected)
    curand.generate(curand.create_generator(seed=seed, offset=offset), actual)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.raw32
def test_raw32_aligned_fast_path_matches_native_curand() -> None:
    expected = torch.empty(64, device="cuda", dtype=torch.int32)
    actual = torch.empty_like(expected)
    with NvidiaCurandGenerator(seed=43) as generator:
        generator.generate("raw32", expected)
    curand.generate(curand.create_generator(seed=43), actual)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("distribution", "operation", "dtype"),
    (
        pytest.param("uniform_f32", curand.generate_uniform, torch.float32, marks=pytest.mark.uniform_f32),
        pytest.param("uniform_f64", curand.generate_uniform_double, torch.float64, marks=pytest.mark.uniform_f64),
    ),
)
@pytest.mark.parametrize("offset", range(4))
def test_uniform_matches_native_curand(distribution, operation, dtype: torch.dtype, offset: int) -> None:
    expected = torch.empty(34, device="cuda", dtype=dtype)
    actual = torch.empty_like(expected)
    with NvidiaCurandGenerator(seed=71, offset=offset) as generator:
        generator.generate(distribution, expected)
    operation(curand.create_generator(seed=71, offset=offset), actual)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("distribution", "operation", "dtype"),
    (
        pytest.param("uniform_f32", curand.generate_uniform, torch.float32, marks=pytest.mark.uniform_f32),
        pytest.param("uniform_f64", curand.generate_uniform_double, torch.float64, marks=pytest.mark.uniform_f64),
    ),
)
def test_uniform_aligned_fast_path_matches_native_curand(distribution, operation, dtype: torch.dtype) -> None:
    expected = torch.empty(64, device="cuda", dtype=dtype)
    actual = torch.empty_like(expected)
    with NvidiaCurandGenerator(seed=73) as generator:
        generator.generate(distribution, expected)
    operation(curand.create_generator(seed=73), actual)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("distribution", "operation", "dtype", "lognormal"),
    (
        pytest.param(
            "normal_f32",
            curand.generate_normal,
            torch.float32,
            False,
            marks=pytest.mark.normal_f32,
        ),
        pytest.param(
            "normal_f64",
            curand.generate_normal_double,
            torch.float64,
            False,
            marks=pytest.mark.normal_f64,
        ),
        pytest.param(
            "lognormal_f32",
            curand.generate_lognormal,
            torch.float32,
            True,
            marks=pytest.mark.lognormal_f32,
        ),
        pytest.param(
            "lognormal_f64",
            curand.generate_lognormal_double,
            torch.float64,
            True,
            marks=pytest.mark.lognormal_f64,
        ),
    ),
)
def test_normal_family_matches_native_mathematical_contract(
    distribution,
    operation,
    dtype: torch.dtype,
    lognormal: bool,
) -> None:
    n = 262144
    mean = 0.25
    stddev = 0.75
    native = torch.empty(n, device="cuda", dtype=dtype)
    actual = torch.empty_like(native)
    with NvidiaCurandGenerator(seed=79) as generator:
        generator.generate(distribution, native, mean=mean, stddev=stddev)
    operation(curand.create_generator(seed=79), actual, mean=mean, stddev=stddev)
    torch.cuda.synchronize()

    for output in (native, actual):
        values = output.log() if lognormal else output
        assert bool(torch.isfinite(values).all())
        assert abs(float(values.mean()) - mean) < 0.015
        assert abs(float(values.std(unbiased=False)) - stddev) < 0.015


@pytest.mark.raw32
def test_native_baseline_uses_the_current_cuda_stream() -> None:
    expected = torch.empty(4096, device="cuda", dtype=torch.int32)
    actual = torch.empty_like(expected)
    with NvidiaCurandGenerator(seed=83) as generator:
        generator.generate("raw32", expected)

    stream = torch.cuda.Stream()
    with NvidiaCurandGenerator(seed=83) as generator, torch.cuda.stream(stream):
        generator.generate("raw32", actual)
    stream.synchronize()
    assert torch.equal(actual, expected)
