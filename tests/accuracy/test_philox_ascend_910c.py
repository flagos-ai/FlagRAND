"""Ascend 910C replay, stream-continuity, and statistical tests."""

from __future__ import annotations

import pytest
import torch

from flagrand import curand

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

NPU_AVAILABLE = torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()
OPERATOR_CASES = (
    pytest.param("raw32", torch.int32, marks=pytest.mark.raw32),
    pytest.param("uniform_f32", torch.float32, marks=pytest.mark.uniform_f32),
    pytest.param("uniform_f64", torch.float64, marks=pytest.mark.uniform_f64),
    pytest.param("normal_f32", torch.float32, marks=pytest.mark.normal_f32),
    pytest.param("normal_f64", torch.float64, marks=pytest.mark.normal_f64),
    pytest.param("lognormal_f32", torch.float32, marks=pytest.mark.lognormal_f32),
    pytest.param("lognormal_f64", torch.float64, marks=pytest.mark.lognormal_f64),
)
pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.ascend_910c,
    pytest.mark.skipif(not NPU_AVAILABLE, reason="Huawei Ascend NPU with torch_npu is required"),
]


def _generate(distribution: str, generator: object, output: torch.Tensor) -> None:
    if distribution == "raw32":
        curand.generate(generator, output)
    elif distribution == "uniform_f32":
        curand.generate_uniform(generator, output)
    elif distribution == "uniform_f64":
        curand.generate_uniform_double(generator, output)
    elif distribution == "normal_f32":
        curand.generate_normal(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "normal_f64":
        curand.generate_normal_double(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "lognormal_f32":
        curand.generate_lognormal(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "lognormal_f64":
        curand.generate_lognormal_double(generator, output, mean=0.0, stddev=1.0)
    else:
        raise AssertionError(f"unregistered distribution: {distribution}")


@pytest.mark.parametrize(("distribution", "dtype"), OPERATOR_CASES)
def test_ascend_910c_distribution_contract(distribution: str, dtype: torch.dtype) -> None:
    n = 65536
    output = torch.empty(n, device="npu:0", dtype=dtype)
    replay = torch.empty_like(output)

    _generate(distribution, curand.create_generator(seed=2026), output)
    _generate(distribution, curand.create_generator(seed=2026), replay)
    torch.npu.synchronize()

    assert torch.equal(output, replay)
    values = output.cpu().to(torch.float64)
    assert bool(torch.isfinite(values).all())

    if distribution == "raw32":
        negative_fraction = float((values < 0).double().mean())
        assert 0.47 < negative_fraction < 0.53
    elif distribution.startswith("uniform"):
        assert 0.0 < float(values.min()) <= float(values.max()) <= 1.0
        assert abs(float(values.mean()) - 0.5) < 0.01
        assert abs(float(values.var(unbiased=False)) - (1.0 / 12.0)) < 0.005
    else:
        normal_values = values.log() if distribution.startswith("lognormal") else values
        if distribution.startswith("lognormal"):
            assert bool((values > 0).all())
        assert abs(float(normal_values.mean())) < 0.03
        assert abs(float(normal_values.std(unbiased=False)) - 1.0) < 0.03


@pytest.mark.parametrize(
    ("distribution", "dtype"),
    (
        pytest.param("raw32", torch.int32, marks=pytest.mark.raw32),
        pytest.param("uniform_f32", torch.float32, marks=pytest.mark.uniform_f32),
        pytest.param("normal_f32", torch.float32, marks=pytest.mark.normal_f32),
        pytest.param("lognormal_f32", torch.float32, marks=pytest.mark.lognormal_f32),
    ),
)
def test_ascend_910c_f32_bulk_matches_split(distribution: str, dtype: torch.dtype) -> None:
    first_n = 6 if "normal" in distribution else 5
    bulk = torch.empty(18, device="npu:0", dtype=dtype)
    first = torch.empty(first_n, device="npu:0", dtype=dtype)
    second = torch.empty(18 - first_n, device="npu:0", dtype=dtype)

    _generate(distribution, curand.create_generator(seed=2027), bulk)
    split_generator = curand.create_generator(seed=2027)
    _generate(distribution, split_generator, first)
    _generate(distribution, split_generator, second)
    torch.npu.synchronize()

    assert torch.equal(bulk, torch.cat((first, second)))


@pytest.mark.parametrize(
    "distribution",
    (
        pytest.param("normal_f32", marks=pytest.mark.normal_f32),
        pytest.param("lognormal_f32", marks=pytest.mark.lognormal_f32),
    ),
)
def test_ascend_910c_f32_normal_crosses_counter_epoch(distribution: str) -> None:
    start = (1 << 32) - 8
    output = torch.empty(18, device="npu:0", dtype=torch.float32)
    replay = torch.empty_like(output)
    first_generator = curand.create_generator(seed=2028, offset=start)
    replay_generator = curand.create_generator(seed=2028, offset=start)

    _generate(distribution, first_generator, output)
    _generate(distribution, replay_generator, replay)
    torch.npu.synchronize()

    assert torch.equal(output, replay)
    assert first_generator.offset == start + output.numel()
    assert replay_generator.offset == start + replay.numel()
