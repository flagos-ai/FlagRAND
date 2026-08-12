from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from benchmark.muxi_matrix.config import ACCEPTANCE_DISTRIBUTION_SIZES

ALL_DISTRIBUTIONS = (
    "raw32",
    "uniform_f32",
    "uniform_f64",
    "normal_f32",
    "normal_f64",
    "lognormal_f32",
    "lognormal_f64",
)

ASCEND_910C_SUPPORTED_DISTRIBUTIONS = (
    "raw32",
    "uniform_f32",
    "uniform_f64",
    "normal_f32",
    "normal_f64",
    "lognormal_f32",
    "lognormal_f64",
)

ASCEND_910C_UNSUPPORTED_DISTRIBUTIONS: dict[str, str] = {}


@dataclass(frozen=True)
class Profile:
    name: str
    sizes: tuple[int, ...]
    warmup: int
    repeats: int
    timing_budget_ms: float


PROFILES = {
    "curve": Profile(
        name="curve",
        sizes=(1,),
        warmup=2,
        repeats=5,
        timing_budget_ms=10.0,
    ),
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


def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Validate and benchmark FlagRAND against the CANN 9.0 ACLNN RNG " "path on Huawei Ascend 910C.")
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument(
        "--distributions",
        default=",".join(ASCEND_910C_SUPPORTED_DISTRIBUTIONS),
        help="Comma-separated subset of the seven Huawei-baseline APIs.",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Optional comma-separated element counts (normal-like APIs require even N).",
    )
    parser.add_argument(
        "--acceptance-profile",
        action="store_true",
        help="Use the fixed wide shape matrix used by the existing delivery rules.",
    )
    parser.add_argument(
        "--speedup-min",
        type=float,
        default=None,
        help="Optional per-API arithmetic-mean speedup lower bound.",
    )
    parser.add_argument(
        "--speedup-max",
        type=float,
        default=None,
        help="Optional per-API arithmetic-mean speedup upper bound.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0, help="Logical NPU device index.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Ascend UB-safe default; explicitly sweep 64/128 after preflight.",
    )
    parser.add_argument(
        "--num-warps",
        type=int,
        default=4,
        help="Triton launch compatibility option; not an Ascend autotune dimension.",
    )
    parser.add_argument(
        "--skip-cann-baseline",
        action="store_true",
        help="Only validate/time FlagRAND; no ACLNN speedup is produced.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run environment, compile, golden-vector and split-stream checks only.",
    )
    parser.add_argument(
        "--allow-non-cann9",
        action="store_true",
        help="Diagnostic escape hatch; formal runs must leave this disabled.",
    )
    parser.add_argument(
        "--allow-unverified-910c",
        action="store_true",
        help="Diagnostic escape hatch when model evidence cannot identify this as 910C.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / "ascend_910c_results",
    )
    args = parser.parse_args()
    args.distributions = _parse_distributions(args.distributions)
    if args.device < 0:
        raise SystemExit("--device must be non-negative.")
    if args.block_size <= 0 or args.num_warps <= 0:
        raise SystemExit("--block-size and --num-warps must be positive.")
    if args.speedup_min is not None and args.speedup_min <= 0.0:
        raise SystemExit("--speedup-min must be positive.")
    if args.speedup_max is not None and args.speedup_max <= 0.0:
        raise SystemExit("--speedup-max must be positive.")
    if args.speedup_min is not None and args.speedup_max is not None and args.speedup_min > args.speedup_max:
        raise SystemExit("--speedup-min must be <= --speedup-max.")
    if args.acceptance_profile and args.sizes is not None:
        raise SystemExit("--sizes cannot be combined with --acceptance-profile.")
    if args.skip_cann_baseline and (args.speedup_min is not None or args.speedup_max is not None):
        raise SystemExit("Speedup bounds require the CANN baseline.")

    base_profile = PROFILES[args.profile]
    if args.acceptance_profile:
        args.shape_map = {
            distribution: ACCEPTANCE_DISTRIBUTION_SIZES[distribution] for distribution in args.distributions
        }
        sizes = tuple(sorted({n for values in args.shape_map.values() for n in values}))
        args.profile_config = Profile(
            name=f"acceptance_{base_profile.name}",
            sizes=sizes,
            warmup=base_profile.warmup,
            repeats=base_profile.repeats,
            timing_budget_ms=base_profile.timing_budget_ms,
        )
    else:
        sizes = _parse_sizes(args.sizes) if args.sizes is not None else base_profile.sizes
        args.shape_map = {distribution: sizes for distribution in args.distributions}
        profile_name = base_profile.name if args.sizes is None else f"{base_profile.name}_custom_sizes"
        args.profile_config = Profile(
            name=profile_name,
            sizes=sizes,
            warmup=base_profile.warmup,
            repeats=base_profile.repeats,
            timing_budget_ms=base_profile.timing_budget_ms,
        )
    return args


def iter_cases(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.acceptance_profile:
        return [(distribution, n) for distribution in args.distributions for n in args.shape_map[distribution]]
    return [(distribution, n) for n in args.profile_config.sizes for distribution in args.distributions]


def supports_size(distribution: str, n: int) -> bool:
    # The 910C fast normal-score kernel is one-output-per-counter and CANN's
    # native FP64 path also accepts odd lengths.
    return True


def _parse_distributions(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(names) - set(ALL_DISTRIBUTIONS))
    if unknown:
        raise SystemExit(f"Unsupported distributions: {', '.join(unknown)}")
    if not names:
        raise SystemExit("At least one distribution is required.")
    return names


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit("--sizes must contain only integers.") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise SystemExit("Every --sizes value must be positive.")
    return sizes
