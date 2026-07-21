from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import torch

from muxi_matrix.compat import (
    apply_maca_compatibility,
    collect_environment,
    verify_preview_source,
)
from muxi_matrix.config import parse_args
from muxi_matrix.reporting import (
    build_acceptance_summary,
    create_run_directory,
    print_acceptance_summary,
    print_record,
    write_results,
)


def main(
    *,
    default_results_dir_name: str = "muxi_results",
    default_skip_ppu_curand: bool = True,
    default_skip_ppu_acrand: bool = True,
    platform_name: str = "muxi",
) -> int:
    args = parse_args(
        PROJECT_ROOT,
        default_results_dir_name=default_results_dir_name,
        default_skip_ppu_curand=default_skip_ppu_curand,
        default_skip_ppu_acrand=default_skip_ppu_acrand,
        platform_name=platform_name,
    )
    run_directory = create_run_directory(args.results_dir, args.profile_config.name)
    manifest: dict[str, Any] = {
        "profile": args.profile_config.name,
        "sizes": args.profile_config.sizes,
        "distributions": args.distributions,
        "seed": args.seed,
        "preflight_only": args.preflight_only,
        "skip_ppu_curand": args.skip_ppu_curand,
        "skip_ppu_acrand": args.skip_ppu_acrand,
        "skip_torch_reference": args.skip_torch_reference,
        "platform": platform_name,
        "acceptance_profile": args.acceptance_profile,
        "acceptance_baseline": args.acceptance_baseline,
        "acceptance_min": args.acceptance_min,
        "acceptance_max": args.acceptance_max,
        "acceptance_shape_map": args.acceptance_shape_map,
    }
    records: list[dict[str, Any]] = []

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false; no Muxi/CUDA device is visible.")
        manifest["source"] = verify_preview_source(PROJECT_ROOT)
        manifest["compatibility"] = apply_maca_compatibility()
        manifest["environment"] = collect_environment()

        from flagrand.runtime import libtuner_status
        from muxi_matrix.cases import (
            run_flagrand_case,
            run_ppu_acrand_case,
            run_ppu_curand_case,
            run_preflight,
            run_torch_references,
        )
        from ppu_matrix.curand import ppu_curand_available
        from ppu_matrix.acrand import ppu_acrand_available

        manifest["libtuner"] = libtuner_status()
        manifest["ppu_curand_available"] = ppu_curand_available()
        manifest["ppu_acrand_available"] = ppu_acrand_available()

        manifest["preflight"] = run_preflight(args.seed)
        print(f"Preflight: {manifest['preflight']['status']}")
        if manifest["preflight"]["status"] != "pass":
            write_results(run_directory, manifest, records)
            print(f"Results: {run_directory}")
            return 2
        if args.preflight_only:
            write_results(run_directory, manifest, records)
            print(f"Results: {run_directory}")
            return 0

        if args.acceptance_profile:
            _print_acceptance_shape_plan(args)

        reference_sizes: set[int] = set()
        for distribution, n in _iter_matrix_cases(args):
            if not _supports_size(distribution, n):
                print(f"SKIP {distribution:<15} N={n:<9} requires an even element count")
                continue
            record = run_flagrand_case(distribution, n, args.seed, args.profile_config)
            ppu_record = None
            acrand_record = None
            if not args.skip_ppu_curand:
                ppu_record = run_ppu_curand_case(
                    distribution,
                    n,
                    args.seed,
                    args.profile_config,
                )
                _annotate_ppu_speedup(record, ppu_record)
            if not args.skip_ppu_acrand:
                acrand_record = run_ppu_acrand_case(
                    distribution,
                    n,
                    args.seed,
                    args.profile_config,
                )
                _annotate_ppu_acrand_speedup(record, acrand_record)
            records.append(record)
            print_record(record)
            if ppu_record is not None:
                records.append(ppu_record)
                print_record(ppu_record)
            if acrand_record is not None:
                records.append(acrand_record)
                print_record(acrand_record)
            if not args.skip_torch_reference:
                if n in reference_sizes:
                    continue
                reference_sizes.add(n)
                for record in run_torch_references(n, args.profile_config):
                    records.append(record)
                    print_record(record)

        if args.acceptance_profile:
            manifest["acceptance_summary"] = build_acceptance_summary(
                records,
                distributions=args.distributions,
                shape_map=args.acceptance_shape_map,
                baseline=args.acceptance_baseline,
                min_speedup=args.acceptance_min,
                max_speedup=args.acceptance_max,
            )
            print_acceptance_summary(manifest["acceptance_summary"])
    except Exception as exc:
        manifest["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
        write_results(run_directory, manifest, records)
        print(f"Fatal error: {type(exc).__name__}: {exc}")
        print(f"Results: {run_directory}")
        return 3

    write_results(run_directory, manifest, records)
    failures = sum(record["status"] != "pass" for record in records)
    print(f"Completed {len(records)} records with {failures} failure(s).")
    print(f"Results: {run_directory}")
    return 1 if failures else 0


def _supports_size(distribution: str, n: int) -> bool:
    if distribution.startswith("normal") or distribution.startswith("lognormal"):
        return n % 2 == 0
    return True


def _iter_matrix_cases(args: Any) -> list[tuple[str, int]]:
    if args.acceptance_profile:
        return [
            (distribution, n)
            for distribution in args.distributions
            for n in args.acceptance_shape_map[distribution]
        ]
    return [
        (distribution, n)
        for n in args.profile_config.sizes
        for distribution in args.distributions
    ]


def _print_acceptance_shape_plan(args: Any) -> None:
    print(
        "Acceptance shape plan "
        f"baseline={args.acceptance_baseline} "
        f"target=[{args.acceptance_min:.2f}, {args.acceptance_max:.2f}]"
    )
    for distribution in args.distributions:
        shapes = ",".join(str(n) for n in args.acceptance_shape_map[distribution])
        print(f"ACCEPTANCE_SHAPE {distribution:<15} N={shapes}")


def _annotate_ppu_speedup(flagrand_record: dict[str, Any], ppu_record: dict[str, Any]) -> None:
    if flagrand_record["status"] != "pass" or ppu_record["status"] != "pass":
        return
    flagrand_record["speedup_vs_ppu_curand"] = (
        float(ppu_record["total_ms_median"]) / float(flagrand_record["total_ms_median"])
    )
    if "device_ms_median" in flagrand_record and "device_ms_median" in ppu_record:
        flagrand_record["device_speedup_vs_ppu_curand"] = (
            float(ppu_record["device_ms_median"]) / float(flagrand_record["device_ms_median"])
        )
    ppu_record["speedup_reference_for"] = "flagrand_public"


def _annotate_ppu_acrand_speedup(flagrand_record: dict[str, Any], ppu_record: dict[str, Any]) -> None:
    if flagrand_record["status"] != "pass" or ppu_record["status"] != "pass":
        return
    flagrand_record["speedup_vs_ppu_acrand"] = (
        float(ppu_record["total_ms_median"]) / float(flagrand_record["total_ms_median"])
    )
    if "device_ms_median" in flagrand_record and "device_ms_median" in ppu_record:
        flagrand_record["device_speedup_vs_ppu_acrand"] = (
            float(ppu_record["device_ms_median"]) / float(flagrand_record["device_ms_median"])
        )
    ppu_record["speedup_reference_for"] = "flagrand_public"


if __name__ == "__main__":
    raise SystemExit(main())
