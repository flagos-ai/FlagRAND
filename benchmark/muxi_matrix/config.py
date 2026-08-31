"""Muxi benchmark configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ALL_DISTRIBUTIONS = (
    "raw32",
    "uniform_f32",
    "uniform_f64",
    "normal_f32",
    "normal_f64",
    "lognormal_f32",
    "lognormal_f64",
)

ACCEPTANCE_DISTRIBUTION_SIZES = {
    "raw32": (
        128,
        256,
        1023,
        1024,
        1025,
        4095,
        4096,
        4097,
        16384,
        65535,
        65536,
        65537,
        262144,
        1048575,
        1048576,
        1048577,
    ),
    "uniform_f32": (
        128,
        256,
        1023,
        1024,
        1025,
        4095,
        4096,
        4097,
        16384,
        65535,
        65536,
        65537,
        262144,
        1048575,
        1048576,
        1048577,
    ),
    "uniform_f64": (
        128,
        256,
        1023,
        1024,
        1025,
        4095,
        4096,
        4097,
        16384,
        65535,
        65536,
        65537,
        262144,
        1048575,
        1048576,
        1048577,
    ),
    "normal_f32": (
        126,
        128,
        130,
        1022,
        1024,
        1026,
        4094,
        4096,
        4098,
        16384,
        65534,
        65536,
        65538,
        262144,
        1048574,
        1048576,
        1048578,
    ),
    "normal_f64": (
        126,
        128,
        1024,
        1026,
        4096,
        4098,
        16384,
        65534,
        65536,
        65538,
        262144,
    ),
    "lognormal_f32": (
        126,
        128,
        130,
        1022,
        1024,
        1026,
        4094,
        4096,
        4098,
        16384,
        65534,
        65536,
        65538,
        262144,
        1048574,
        1048576,
        1048578,
    ),
    "lognormal_f64": (
        18,
        126,
        128,
        130,
        1022,
        1024,
        1026,
        4094,
        4096,
        4098,
        16384,
        65534,
        65536,
        65538,
        262144,
    ),
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
        sizes=(4096, 65536, 1048576),
        warmup=2,
        repeats=5,
        timing_budget_ms=80.0,
    ),
    "full": Profile(
        name="full",
        sizes=(4096, 16384, 65536, 262144, 1048576, 4194304, 16777216),
        warmup=5,
        repeats=11,
        timing_budget_ms=200.0,
    ),
}


def parse_args(
    project_root: Path,
    *,
    default_results_dir_name: str = "muxi_results",
    default_skip_mcrand: bool = False,
    default_skip_ppu_curand: bool = True,
    default_skip_ppu_acrand: bool = True,
    default_acceptance_baseline: str = "mcrand",
    platform_name: str = "muxi",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Validate and time the preview's Philox Host API on {platform_name} GPUs."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument(
        "--distributions",
        default=",".join(ALL_DISTRIBUTIONS),
        help="Comma-separated subset of the seven supported distributions.",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help=(
            "Optional comma-separated element counts. raw/uniform accepts any "
            "positive count; normal/lognormal requires even counts."
        ),
    )
    parser.add_argument(
        "--acceptance-profile",
        action="store_true",
        help="Run the fixed acceptance shape set and print per-API speedups against the selected native baseline.",
    )
    parser.add_argument(
        "--acceptance-baseline",
        choices=("mcrand", "ppu_acrand", "ppu_curand"),
        default=default_acceptance_baseline,
        help="Baseline used by --acceptance-profile speedup reporting.",
    )
    parser.add_argument(
        "--acceptance-min",
        type=float,
        default=0.95,
        help="Lower bound for acceptance speedup reporting.",
    )
    parser.add_argument(
        "--acceptance-max",
        type=float,
        default=1.50,
        help="Upper bound for acceptance speedup reporting.",
    )
    parser.add_argument(
        "--strict-acceptance",
        action="store_true",
        help="Return a nonzero status when an acceptance API misses the configured range.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--skip-torch-reference",
        action="store_true",
        help="Skip PyTorch RNG and zero-write references.",
    )
    parser.add_argument(
        "--skip-mcrand",
        action="store_true",
        help="Skip the MetaX native mcRAND baseline.",
    )
    parser.add_argument(
        "--with-mcrand",
        action="store_false",
        dest="skip_mcrand",
        help="Run the MetaX native mcRAND baseline.",
    )
    parser.add_argument(
        "--skip-ppu-curand",
        action="store_true",
        help="Skip the PPU cuRAND baseline.",
    )
    parser.add_argument(
        "--with-ppu-curand",
        action="store_false",
        dest="skip_ppu_curand",
        help="Run the PPU cuRAND baseline.",
    )
    parser.add_argument(
        "--skip-ppu-acrand",
        action="store_true",
        help="Skip the PPU native acrand baseline.",
    )
    parser.add_argument(
        "--with-ppu-acrand",
        action="store_false",
        dest="skip_ppu_acrand",
        help="Run the PPU native acrand baseline.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run source, compile, golden-vector, and split-stream checks only.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / default_results_dir_name,
    )
    parser.set_defaults(skip_mcrand=default_skip_mcrand)
    parser.set_defaults(skip_ppu_curand=default_skip_ppu_curand)
    parser.set_defaults(skip_ppu_acrand=default_skip_ppu_acrand)
    args = parser.parse_args()
    args.distributions = _parse_distributions(args.distributions)
    if args.acceptance_min <= 0.0 or args.acceptance_max <= 0.0:
        raise SystemExit("--acceptance-min and --acceptance-max must be positive.")
    if args.acceptance_min > args.acceptance_max:
        raise SystemExit("--acceptance-min must be <= --acceptance-max.")
    if args.strict_acceptance and not args.acceptance_profile:
        raise SystemExit("--strict-acceptance requires --acceptance-profile.")
    if args.acceptance_profile:
        args.profile_config = _resolve_acceptance_profile(
            args.profile,
            args.sizes,
            args.distributions,
        )
        args.acceptance_shape_map = {
            distribution: ACCEPTANCE_DISTRIBUTION_SIZES[distribution] for distribution in args.distributions
        }
        _validate_acceptance_baseline(args)
    else:
        args.profile_config = _resolve_profile(args.profile, args.sizes)
        args.acceptance_shape_map = {}
    return args


def _parse_distributions(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(names) - set(ALL_DISTRIBUTIONS))
    if unknown:
        raise SystemExit(f"Unsupported distributions: {', '.join(unknown)}")
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


def _resolve_acceptance_profile(
    name: str,
    sizes_value: str | None,
    distributions: tuple[str, ...],
) -> Profile:
    if sizes_value is not None:
        raise SystemExit("--sizes cannot be combined with --acceptance-profile.")
    missing = sorted(set(distributions) - set(ACCEPTANCE_DISTRIBUTION_SIZES))
    if missing:
        raise SystemExit(f"No acceptance shapes configured for: {', '.join(missing)}")
    profile = PROFILES[name]
    sizes = tuple(
        sorted({size for distribution in distributions for size in ACCEPTANCE_DISTRIBUTION_SIZES[distribution]})
    )
    return Profile(
        name=f"acceptance_{profile.name}",
        sizes=sizes,
        warmup=profile.warmup,
        repeats=profile.repeats,
        timing_budget_ms=profile.timing_budget_ms,
    )


def _validate_acceptance_baseline(args: argparse.Namespace) -> None:
    if args.acceptance_baseline == "mcrand" and args.skip_mcrand:
        raise SystemExit("--acceptance-profile requires --with-mcrand for mcrand reporting.")
    if args.acceptance_baseline == "ppu_acrand" and args.skip_ppu_acrand:
        raise SystemExit("--acceptance-profile requires --with-ppu-acrand for ppu_acrand reporting.")
    if args.acceptance_baseline == "ppu_curand" and args.skip_ppu_curand:
        raise SystemExit("--acceptance-profile requires --with-ppu-curand for ppu_curand reporting.")
