from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ALL_DISTRIBUTIONS = (
    "raw32",
    "uniform_f32",
    "normal_f32",
    "lognormal_f32",
)

COREX_CURAND_DISTRIBUTIONS = (
    "raw32",
    "uniform_f32",
    "normal_f32",
    "lognormal_f32",
)

RAW_UNIFORM_MATRIX_SHAPES = (
    1,
    2,
    3,
    4,
    5,
    7,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    4095,
    4096,
    4097,
    16383,
    16384,
    16385,
    65535,
    65536,
    65537,
    262143,
    262144,
    262145,
    1048575,
    1048576,
    1048577,
    4194304,
    16777216,
)

NORMAL_MATRIX_SHAPES = (
    2,
    4,
    6,
    8,
    14,
    16,
    18,
    30,
    32,
    34,
    62,
    64,
    66,
    126,
    128,
    130,
    254,
    256,
    258,
    510,
    512,
    514,
    1022,
    1024,
    1026,
    4094,
    4096,
    4098,
    16382,
    16384,
    16386,
    65534,
    65536,
    65538,
    262142,
    262144,
    262146,
    1048574,
    1048576,
    1048578,
    4194304,
    16777216,
)

MATRIX_SHAPE_MAP = {
    "raw32": RAW_UNIFORM_MATRIX_SHAPES,
    "uniform_f32": RAW_UNIFORM_MATRIX_SHAPES,
    "uniform_f64": RAW_UNIFORM_MATRIX_SHAPES,
    "normal_f32": NORMAL_MATRIX_SHAPES,
    "lognormal_f32": NORMAL_MATRIX_SHAPES,
}


@dataclass(frozen=True)
class Profile:
    name: str
    sizes: tuple[int, ...]
    warmup: int
    repeats: int
    timing_budget_ms: float


PROFILES = {
    "smoke": Profile(
        name="smoke",
        sizes=(1 << 20, 1 << 24),
        warmup=2,
        repeats=5,
        timing_budget_ms=80.0,
    ),
    "matrix": Profile(
        name="matrix",
        sizes=(),
        warmup=5,
        repeats=11,
        timing_budget_ms=200.0,
    ),
}


def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and time FlagRAND on BI-V150 CoreX.")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument(
        "--distributions",
        default=",".join(ALL_DISTRIBUTIONS),
        help="Comma-separated subset of the implemented CoreX distributions.",
    )
    parser.add_argument("--sizes", default=None, help="Optional comma-separated positive element counts.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--results-dir", type=Path, default=project_root / "corex_results")
    args = parser.parse_args()
    args.distributions = _parse_distributions(args.distributions)
    args.profile_config = _resolve_profile(args.profile, args.sizes)
    if args.profile == "matrix" and args.sizes is None:
        args.shape_map = {distribution: MATRIX_SHAPE_MAP[distribution] for distribution in args.distributions}
    else:
        args.shape_map = {distribution: args.profile_config.sizes for distribution in args.distributions}
    return args


def _parse_distributions(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if any(name.endswith("_f64") for name in names):
        raise SystemExit("BI-V150 current scope excludes FP64 APIs")
    unknown = sorted(set(names) - set(ALL_DISTRIBUTIONS))
    if unknown:
        raise SystemExit(f"Unsupported CoreX distributions: {', '.join(unknown)}")
    if not names:
        raise SystemExit("At least one distribution is required.")
    return names


def _resolve_profile(name: str, sizes_value: str | None) -> Profile:
    profile = PROFILES[name]
    if sizes_value is None:
        return profile
    try:
        sizes = tuple(int(part.strip()) for part in sizes_value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit("--sizes must contain only integers.") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise SystemExit("Every --sizes value must be positive.")
    return Profile(
        name=f"{name}_custom_sizes",
        sizes=sizes,
        warmup=profile.warmup,
        repeats=profile.repeats,
        timing_budget_ms=profile.timing_budget_ms,
    )
