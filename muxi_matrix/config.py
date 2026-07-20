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
    default_skip_ppu_curand: bool = True,
    default_skip_ppu_acrand: bool = True,
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--skip-torch-reference",
        action="store_true",
        help="Skip PyTorch RNG and zero-write references.",
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
    parser.set_defaults(skip_ppu_curand=default_skip_ppu_curand)
    parser.set_defaults(skip_ppu_acrand=default_skip_ppu_acrand)
    args = parser.parse_args()
    args.distributions = _parse_distributions(args.distributions)
    args.profile_config = _resolve_profile(args.profile, args.sizes)
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
