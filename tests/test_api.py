from __future__ import annotations

import pytest
import torch

import flagrand
from flagrand import curand
from flagrand.rng import PhiloxGenerator


def test_public_surface_is_philox_only() -> None:
    expected = {
        "CURAND_RNG_PSEUDO_PHILOX4_32_10",
        "CurandGenerator",
        "create_generator",
        "set_pseudo_random_generator_seed",
        "set_generator_offset",
        "generate",
        "generate_uniform",
        "generate_uniform_double",
        "generate_normal",
        "generate_normal_double",
        "generate_lognormal",
        "generate_lognormal_double",
    }
    assert set(curand.__all__) == expected
    assert set(flagrand.__all__) == {
        "curand",
        "CURAND_RNG_PSEUDO_PHILOX4_32_10",
        "CurandGenerator",
        "PhiloxGenerator",
        "create_generator",
        "set_pseudo_random_generator_seed",
        "set_generator_offset",
    }


@pytest.mark.parametrize("name", ["philox", "philox4_32_10", curand.CURAND_RNG_PSEUDO_PHILOX4_32_10])
def test_create_generator_accepts_philox_names(name: str) -> None:
    generator = curand.create_generator(name, seed=7, offset=8)
    assert isinstance(generator.engine, PhiloxGenerator)
    assert generator.seed == 7
    assert generator.offset == 8


def test_create_generator_rejects_unshipped_generator() -> None:
    with pytest.raises(ValueError, match="Unsupported cuRAND rng_type"):
        curand.create_generator("xorwow")


def test_seed_and_offset_mutators_validate_and_update_state() -> None:
    generator = curand.create_generator(seed=1, offset=8)
    curand.set_pseudo_random_generator_seed(generator, 99)
    assert generator.seed == 99
    assert generator.offset == 0

    curand.set_generator_offset(generator, 32)
    assert generator.offset == 32

    with pytest.raises(ValueError, match="multiple of 4"):
        curand.set_generator_offset(generator, 3)
    with pytest.raises(ValueError, match=r"2\^64-1"):
        curand.set_pseudo_random_generator_seed(generator, 1 << 64)


def test_facade_checks_dtype_before_device() -> None:
    generator = curand.create_generator()
    with pytest.raises(TypeError, match="generate_uniform_double"):
        curand.generate_uniform_double(generator, torch.empty(8, dtype=torch.float32))
    with pytest.raises(TypeError, match="generate_normal"):
        curand.generate_normal(
            generator,
            torch.empty(8, dtype=torch.float64),
            mean=0.0,
            stddev=1.0,
        )


def test_facade_rejects_cpu_output_and_unknown_options() -> None:
    generator = curand.create_generator()
    out = torch.empty(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous CUDA output"):
        curand.generate_uniform(generator, out)
    with pytest.raises(ValueError, match="contiguous CUDA output"):
        curand.generate_uniform(generator, out, unsupported=True)


def test_launch_config_validation_happens_before_launch() -> None:
    generator = curand.create_generator()
    out = torch.empty(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="block_size"):
        curand.generate_uniform(generator, out, block_size=0)
    with pytest.raises(ValueError, match="num_warps"):
        curand.generate_uniform(generator, out, num_warps=0)
