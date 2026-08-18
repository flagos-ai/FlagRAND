from flagrand import curand
from flagrand._curand_handle import (
    CURAND_RNG_PSEUDO_PHILOX4_32_10,
    CurandGenerator,
    create_generator,
    set_generator_offset,
    set_pseudo_random_generator_seed,
)
from flagrand.rng import PhiloxGenerator

__version__ = "0.1.0"

__all__ = [
    "curand",
    "CURAND_RNG_PSEUDO_PHILOX4_32_10",
    "CurandGenerator",
    "PhiloxGenerator",
    "create_generator",
    "set_pseudo_random_generator_seed",
    "set_generator_offset",
]
