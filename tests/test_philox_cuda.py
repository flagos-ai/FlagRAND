from __future__ import annotations

import pytest
import torch

from flagrand import curand
from flagrand.fused._internal.philox_direct import (
    generate_philox_lognormal_f32,
    generate_philox_normal_f32,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_raw_matches_known_seed_zero_vector() -> None:
    expected = [
        1713891541,
        -1135104948,
        -513161843,
        -1694442536,
        -2075847199,
        253345875,
        -259166550,
        -2080868880,
        107075191,
        639715526,
        -1232967847,
        2000445424,
        -257357124,
        -1092682108,
        291018424,
        -1200667194,
    ]
    actual = torch.empty(len(expected), device="cuda", dtype=torch.int32)
    curand.generate(curand.create_generator(seed=0), actual)
    assert actual.cpu().tolist() == expected


def test_raw_split_matches_bulk_across_launcher_specializations() -> None:
    bulk = torch.empty(136, device="cuda", dtype=torch.int32)
    first = torch.empty(68, device="cuda", dtype=torch.int32)
    second = torch.empty(68, device="cuda", dtype=torch.int32)

    curand.generate(curand.create_generator(seed=17), bulk)
    split_generator = curand.create_generator(seed=17)
    curand.generate(split_generator, first)
    curand.generate(split_generator, second)
    torch.cuda.synchronize()

    assert torch.equal(bulk, torch.cat((first, second)))


def test_offset_selects_the_same_raw_subsequence() -> None:
    bulk = torch.empty(128, device="cuda", dtype=torch.int32)
    tail = torch.empty(64, device="cuda", dtype=torch.int32)

    curand.generate(curand.create_generator(seed=23), bulk)
    curand.generate(curand.create_generator(seed=23, offset=64), tail)
    torch.cuda.synchronize()

    assert torch.equal(tail, bulk[64:])


def test_large_seed_and_unaligned_contiguous_output() -> None:
    expected = torch.empty(68, device="cuda", dtype=torch.int32)
    storage = torch.empty(69, device="cuda", dtype=torch.int32)
    unaligned = storage[1:]
    seed = (1 << 40) + 29

    curand.generate(curand.create_generator(seed=seed), expected)
    curand.generate(curand.create_generator(seed=seed), unaligned)
    torch.cuda.synchronize()

    assert unaligned.is_contiguous()
    assert unaligned.data_ptr() % 16 != 0
    assert torch.equal(expected, unaligned)


def test_facade_uses_the_current_cuda_stream() -> None:
    expected = torch.empty(4096, device="cuda", dtype=torch.float32)
    actual = torch.empty_like(expected)
    curand.generate_uniform(curand.create_generator(seed=31), expected)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        curand.generate_uniform(curand.create_generator(seed=31), actual)
    stream.synchronize()

    assert torch.equal(expected, actual)


def test_uniform_f32_and_f64_statistics_and_endpoints() -> None:
    n = 262144
    uniform32 = torch.empty(n, device="cuda", dtype=torch.float32)
    uniform64 = torch.empty(n, device="cuda", dtype=torch.float64)

    curand.generate_uniform(curand.create_generator(seed=41), uniform32)
    curand.generate_uniform_double(curand.create_generator(seed=43), uniform64)
    torch.cuda.synchronize()

    for values in (uniform32, uniform64):
        assert bool(torch.isfinite(values).all())
        assert 0.0 < float(values.min()) <= float(values.max()) <= 1.0
        assert abs(float(values.mean()) - 0.5) < 0.004
        assert abs(float(values.var()) - (1.0 / 12.0)) < 0.002


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_split_matches_bulk(dtype: torch.dtype) -> None:
    operation = (
        curand.generate_uniform_double
        if dtype == torch.float64
        else curand.generate_uniform
    )
    bulk = torch.empty(8192, device="cuda", dtype=dtype)
    first = torch.empty(4096, device="cuda", dtype=dtype)
    second = torch.empty_like(first)

    operation(curand.create_generator(seed=45), bulk)
    split_generator = curand.create_generator(seed=45)
    operation(split_generator, first)
    operation(split_generator, second)
    assert torch.equal(bulk, torch.cat((first, second)))


def test_distribution_offsets_count_consumed_raw_uint32_words() -> None:
    cases = (
        (curand.generate_uniform, torch.float32, {}, 8),
        (curand.generate_uniform_double, torch.float64, {}, 16),
        (curand.generate_normal, torch.float32, {"mean": 0.0, "stddev": 1.0}, 8),
        (
            curand.generate_normal_double,
            torch.float64,
            {"mean": 0.0, "stddev": 1.0},
            16,
        ),
    )
    for operation, dtype, kwargs, expected_offset in cases:
        generator = curand.create_generator(seed=46)
        out = torch.empty(8, device="cuda", dtype=dtype)
        operation(generator, out, **kwargs)
        assert generator.offset == expected_offset


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normal_and_lognormal_statistics(dtype: torch.dtype) -> None:
    n = 262144
    normal = torch.empty(n, device="cuda", dtype=dtype)
    lognormal = torch.empty_like(normal)
    normal_op = curand.generate_normal_double if dtype == torch.float64 else curand.generate_normal
    lognormal_op = (
        curand.generate_lognormal_double
        if dtype == torch.float64
        else curand.generate_lognormal
    )

    normal_op(curand.create_generator(seed=47), normal, mean=0.25, stddev=1.5)
    lognormal_op(curand.create_generator(seed=53), lognormal, mean=0.2, stddev=0.5)
    torch.cuda.synchronize()

    assert bool(torch.isfinite(normal).all())
    assert abs(float(normal.mean()) - 0.25) < 0.015
    assert abs(float(normal.std()) - 1.5) < 0.015
    assert bool((lognormal > 0).all())
    log_values = lognormal.log()
    assert abs(float(log_values.mean()) - 0.2) < 0.01
    assert abs(float(log_values.std()) - 0.5) < 0.01


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normal_and_lognormal_split_match_bulk(dtype: torch.dtype) -> None:
    n = 8192
    normal_op = curand.generate_normal_double if dtype == torch.float64 else curand.generate_normal
    lognormal_op = (
        curand.generate_lognormal_double
        if dtype == torch.float64
        else curand.generate_lognormal
    )

    for operation, seed in ((normal_op, 59), (lognormal_op, 61)):
        bulk = torch.empty(n, device="cuda", dtype=dtype)
        first = torch.empty(n // 2, device="cuda", dtype=dtype)
        second = torch.empty_like(first)
        operation(curand.create_generator(seed=seed), bulk, mean=0.2, stddev=0.5)
        split_generator = curand.create_generator(seed=seed)
        operation(split_generator, first, mean=0.2, stddev=0.5)
        operation(split_generator, second, mean=0.2, stddev=0.5)
        assert torch.equal(bulk, torch.cat((first, second)))


@pytest.mark.parametrize(
    ("operation", "lognormal"),
    (
        (generate_philox_normal_f32, False),
        (generate_philox_lognormal_f32, True),
    ),
)
def test_portable_f32_math_path(operation, lognormal: bool) -> None:
    n = 65536
    bulk = torch.empty(n, device="cuda", dtype=torch.float32)
    first = torch.empty(n // 2, device="cuda", dtype=torch.float32)
    second = torch.empty_like(first)
    kwargs = {
        "mean": 0.0,
        "stddev": 1.0,
        "block_size": 512,
        "num_warps": 4,
        "_portable_math": True,
    }

    operation(bulk, curand.create_generator(seed=63).engine, **kwargs)
    split_generator = curand.create_generator(seed=63).engine
    operation(first, split_generator, **kwargs)
    operation(second, split_generator, **kwargs)
    torch.cuda.synchronize()

    assert torch.equal(bulk, torch.cat((first, second)))
    values = bulk.log() if lognormal else bulk
    assert bool(torch.isfinite(values).all())
    assert abs(float(values.mean())) < 0.02
    assert abs(float(values.std()) - 1.0) < 0.02


def test_shape_and_layout_contract_errors() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        curand.generate(
            curand.create_generator(),
            torch.empty(6, device="cuda", dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="even element count"):
        curand.generate_normal_double(
            curand.create_generator(),
            torch.empty(3, device="cuda", dtype=torch.float64),
            mean=0.0,
            stddev=1.0,
        )
    storage = torch.empty(16, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous CUDA output"):
        curand.generate_uniform(curand.create_generator(), storage[::2])


def test_empty_cuda_outputs_are_noops() -> None:
    generator = curand.create_generator(seed=67, offset=8)
    curand.generate(generator, torch.empty(0, device="cuda", dtype=torch.int32))
    curand.generate_uniform(
        generator,
        torch.empty(0, device="cuda", dtype=torch.float32),
    )
    assert generator.offset == 8
