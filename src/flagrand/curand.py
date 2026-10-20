"""cuRAND-shaped host APIs with generator-specific state and backend dispatch."""

import torch

from flagrand import _curand_philox_api as _philox
from flagrand._curand_handle import (
    CURAND_RNG_PSEUDO_MRG32K3A,
    CURAND_RNG_PSEUDO_MTGP32,
    CURAND_RNG_PSEUDO_PHILOX4_32_10,
    CURAND_RNG_PSEUDO_XORWOW,
    CurandGenerator,
    create_generator,
    set_generator_offset,
    set_pseudo_random_generator_seed,
)
from flagrand.fused.poisson import generate_poisson as _poisson


def generate(generator: CurandGenerator, out: torch.Tensor, **kwargs):
    if generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return _philox.generate(generator, out, **kwargs)
    return generator.engine.generate(out, **kwargs)


def generate_uniform(generator: CurandGenerator, out: torch.Tensor, **kwargs):
    if generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return _philox.generate_uniform(generator, out, **kwargs)
    return generator.engine.generate_uniform(out, **kwargs)


def generate_uniform_double(generator: CurandGenerator, out: torch.Tensor, **kwargs):
    if generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return _philox.generate_uniform_double(generator, out, **kwargs)
    return generator.engine.generate_uniform(out, **kwargs)


def generate_normal(generator: CurandGenerator, out: torch.Tensor, *, mean=0.0, stddev=1.0, **kwargs):
    if generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return _philox.generate_normal(generator, out, mean=mean, stddev=stddev, **kwargs)
    return generator.engine.generate_normal(out, mean=mean, stddev=stddev, **kwargs)


def generate_normal_double(generator: CurandGenerator, out: torch.Tensor, *, mean=0.0, stddev=1.0, **kwargs):
    if generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return _philox.generate_normal_double(generator, out, mean=mean, stddev=stddev, **kwargs)
    return generator.engine.generate_normal(out, mean=mean, stddev=stddev, **kwargs)


def generate_lognormal(generator: CurandGenerator, out: torch.Tensor, *, mean=0.0, stddev=1.0, **kwargs):
    if generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return _philox.generate_lognormal(generator, out, mean=mean, stddev=stddev, **kwargs)
    return generator.engine.generate_normal(out, mean=mean, stddev=stddev, lognormal=True, **kwargs)


def generate_lognormal_double(generator: CurandGenerator, out: torch.Tensor, *, mean=0.0, stddev=1.0, **kwargs):
    if generator.rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return _philox.generate_lognormal_double(generator, out, mean=mean, stddev=stddev, **kwargs)
    return generator.engine.generate_normal(out, mean=mean, stddev=stddev, lognormal=True, **kwargs)


def generate_poisson(generator: CurandGenerator, out: torch.Tensor, lam: float, **kwargs):
    return _poisson(generator.engine, out, lam, **kwargs)


__all__ = [
    "CURAND_RNG_PSEUDO_PHILOX4_32_10",
    "CURAND_RNG_PSEUDO_XORWOW",
    "CURAND_RNG_PSEUDO_MRG32K3A",
    "CURAND_RNG_PSEUDO_MTGP32",
    "CurandGenerator",
    "create_generator",
    "set_generator_offset",
    "set_pseudo_random_generator_seed",
    "generate",
    "generate_uniform",
    "generate_uniform_double",
    "generate_normal",
    "generate_normal_double",
    "generate_lognormal",
    "generate_lognormal_double",
    "generate_poisson",
]
