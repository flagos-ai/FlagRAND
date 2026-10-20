from __future__ import annotations

from dataclasses import dataclass

from flagrand.rng import PhiloxGenerator

CURAND_RNG_PSEUDO_PHILOX4_32_10 = "pseudo_philox4_32_10"
CURAND_RNG_PSEUDO_XORWOW = "pseudo_xorwow"
CURAND_RNG_PSEUDO_MRG32K3A = "pseudo_mrg32k3a"
CURAND_RNG_PSEUDO_MTGP32 = "pseudo_mtgp32"

_ALIASES = {
    "philox": CURAND_RNG_PSEUDO_PHILOX4_32_10,
    "philox4_32_10": CURAND_RNG_PSEUDO_PHILOX4_32_10,
    "xorwow": CURAND_RNG_PSEUDO_XORWOW,
    "mrg32k3a": CURAND_RNG_PSEUDO_MRG32K3A,
    "mtgp32": CURAND_RNG_PSEUDO_MTGP32,
}
_UINT64_MAX = (1 << 64) - 1


@dataclass
class CurandGenerator:
    rng_type: str
    engine: object

    @property
    def seed(self) -> int:
        return int(self.engine.seed)

    @property
    def offset(self) -> int:
        return int(self.engine.offset)


def create_generator(
    rng_type: str = CURAND_RNG_PSEUDO_PHILOX4_32_10,
    *,
    seed: int = 0,
    offset: int = 0,
) -> CurandGenerator:
    resolved = resolve_rng_type(rng_type)
    if resolved == CURAND_RNG_PSEUDO_PHILOX4_32_10:
        engine_type = PhiloxGenerator
    elif resolved == CURAND_RNG_PSEUDO_XORWOW:
        from flagrand.rng.xorwow import XorwowGenerator

        engine_type = XorwowGenerator
    elif resolved == CURAND_RNG_PSEUDO_MRG32K3A:
        from flagrand.rng.mrg32k3a import Mrg32k3aGenerator

        engine_type = Mrg32k3aGenerator
    elif resolved == CURAND_RNG_PSEUDO_MTGP32:
        from flagrand.rng.mtgp32 import Mtgp32Generator

        engine_type = Mtgp32Generator
    else:
        raise ValueError(f"Unsupported cuRAND rng_type: {rng_type}")
    seed_value = _validate_seed(seed)
    offset_value = _validate_offset(offset)
    return CurandGenerator(
        rng_type=resolved,
        engine=engine_type(seed=seed_value, offset=offset_value),
    )


def set_pseudo_random_generator_seed(
    generator: CurandGenerator,
    seed: int,
) -> CurandGenerator:
    if generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        generator.engine.reset(seed=_validate_seed(seed), offset=0)
        return generator
    generator.engine.seed = _validate_seed(seed)
    generator.engine.offset = 0
    return generator


def set_generator_offset(
    generator: CurandGenerator,
    offset: int,
) -> CurandGenerator:
    if generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        generator.engine.reset(seed=generator.seed, offset=_validate_offset(offset))
        return generator
    generator.engine.offset = _validate_offset(offset)
    return generator


def resolve_rng_type(rng_type: str) -> str:
    return _ALIASES.get(rng_type, rng_type)


def _require_philox(generator: CurandGenerator) -> None:
    if not isinstance(generator, CurandGenerator):
        raise TypeError(f"expected CurandGenerator, got {type(generator).__name__}")
    if generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        raise ValueError(f"unsupported generator type: {generator.rng_type}")


def _validate_seed(seed: int) -> int:
    value = int(seed)
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"seed must be in [0, 2^64-1], got {seed!r}")
    return value


def _validate_offset(offset: int) -> int:
    value = int(offset)
    if value < 0:
        raise ValueError(f"Philox offset must be non-negative, got {offset!r}.")
    return value
