from __future__ import annotations

from dataclasses import dataclass

from flagrand.rng import PhiloxGenerator


CURAND_RNG_PSEUDO_PHILOX4_32_10 = "pseudo_philox4_32_10"

_ALIASES = {
    "philox": CURAND_RNG_PSEUDO_PHILOX4_32_10,
    "philox4_32_10": CURAND_RNG_PSEUDO_PHILOX4_32_10,
}
_UINT64_MAX = (1 << 64) - 1


@dataclass
class CurandGenerator:
    rng_type: str
    engine: PhiloxGenerator

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
    if resolved != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        raise ValueError(f"Unsupported cuRAND rng_type in this preview: {rng_type}")
    seed_value = _validate_seed(seed)
    offset_value = _validate_offset(offset)
    return CurandGenerator(
        rng_type=resolved,
        engine=PhiloxGenerator(seed=seed_value, offset=offset_value),
    )


def set_pseudo_random_generator_seed(
    generator: CurandGenerator,
    seed: int,
) -> CurandGenerator:
    _require_philox(generator)
    generator.engine.seed = _validate_seed(seed)
    generator.engine.offset = 0
    return generator


def set_generator_offset(
    generator: CurandGenerator,
    offset: int,
) -> CurandGenerator:
    _require_philox(generator)
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
    if value < 0 or value % 4:
        raise ValueError(
            "Philox offset is measured in uint32 outputs and must be a "
            f"non-negative multiple of 4, got {offset!r}."
        )
    return value
