from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import triton
from triton.runtime.driver import driver

from benchmark.nvidia_matrix.cases import (
    annotate_speedup,
    run_flagrand_case,
    run_native_compatibility_preflight,
    run_nvidia_curand_case,
    warm_flagrand_compilation_paths,
)
from benchmark.nvidia_matrix.config import parse_args
from benchmark.nvidia_matrix.curand import nvidia_curand_library_path, nvidia_curand_version
from benchmark.nvidia_matrix.reporting import (
    build_api_summary,
    create_run_directory,
    print_api_summary,
    print_record,
    write_results,
)
from benchmark.nvidia_matrix.timing import TIMING_SCOPE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args(PROJECT_ROOT)
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    run_directory = create_run_directory(args.results_dir, args.profile_config.name)
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "platform": "nvidia_h20_cuda",
        "acceptance_baseline": "nvidia_curand",
        "profile": args.profile_config.name,
        "shape_map": args.shape_map,
        "distributions": args.distributions,
        "seed": args.seed,
        "device_index": args.device,
        "execution_mode": "preflight_only" if args.preflight_only else "benchmark",
        "timing": {
            "scope": TIMING_SCOPE,
            "formal_metric": "total_ms_median",
            "speedup_formula": "nvidia_curand total_ms_median / FlagRAND total_ms_median",
            "api_aggregation": "arithmetic mean over configured shapes",
            "cross_api_average": None,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "triton_target": str(driver.active.get_current_target()),
            "device": {
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory": properties.total_memory,
            },
            "device_count": torch.cuda.device_count(),
            "curand_library": nvidia_curand_library_path(),
            "curand_version": nvidia_curand_version(),
        },
    }

    try:
        manifest["compilation_preflight"] = warm_flagrand_compilation_paths(args.seed)
        manifest["native_compatibility_preflight"] = run_native_compatibility_preflight(args.seed)
        LOGGER.info("Native compatibility preflight: %s", manifest["native_compatibility_preflight"]["status"])
        native_compatibility_failed = manifest["native_compatibility_preflight"]["status"] != "pass"
        if args.preflight_only:
            write_results(run_directory, manifest, records)
            LOGGER.info("Results: %s", run_directory)
            return 2 if native_compatibility_failed else 0

        for distribution in args.distributions:
            for n in args.shape_map[distribution]:
                if distribution.startswith(("normal", "lognormal")) and n % 2:
                    LOGGER.info("SKIP %-15s N=%-9d requires an even element count", distribution, n)
                    continue
                flagrand_record = run_flagrand_case(distribution, n, args.seed, args.profile_config)
                baseline_record = run_nvidia_curand_case(distribution, n, args.seed, args.profile_config)
                annotate_speedup(flagrand_record, baseline_record)
                records.extend((flagrand_record, baseline_record))
                print_record(flagrand_record)
                print_record(baseline_record)

        manifest["api_speedup_summary"] = build_api_summary(records, args.distributions)
        print_api_summary(manifest["api_speedup_summary"])
    except Exception as exc:
        manifest["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
        write_results(run_directory, manifest, records)
        LOGGER.exception("Fatal error: %s: %s", type(exc).__name__, exc)
        LOGGER.info("Results: %s", run_directory)
        return 3

    write_results(run_directory, manifest, records)
    failures = [record for record in records if record["status"] != "pass"]
    LOGGER.info("Completed %d records with %d failure(s).", len(records), len(failures))
    LOGGER.info("Results: %s", run_directory)
    if native_compatibility_failed:
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
