from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from flagrand import curand
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
from flagrand.rng import Mrg32k3aGenerator, Mtgp32Generator, PhiloxGenerator, XorwowGenerator

__version__ = "0.1.0"

__all__ = [
    "curand",
    "CURAND_RNG_PSEUDO_PHILOX4_32_10",
    "CURAND_RNG_PSEUDO_XORWOW",
    "CURAND_RNG_PSEUDO_MRG32K3A",
    "CURAND_RNG_PSEUDO_MTGP32",
    "CurandGenerator",
    "PhiloxGenerator",
    "XorwowGenerator",
    "Mrg32k3aGenerator",
    "Mtgp32Generator",
    "create_generator",
    "set_pseudo_random_generator_seed",
    "set_generator_offset",
]
