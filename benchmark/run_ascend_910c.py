from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from benchmark.ascend_910c.compat import (
    collect_and_validate_environment,
    import_torch_npu,
    select_device,
    verify_source,
)
from benchmark.ascend_910c.config import ASCEND_910C_UNSUPPORTED_DISTRIBUTIONS, iter_cases, parse_args, supports_size
from benchmark.ascend_910c.reporting import (
    build_api_summary,
    create_run_directory,
    print_api_summary,
    print_record,
    write_diagnostics,
    write_results,
)
from benchmark.ascend_910c.timing import TIMING_SCOPE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args(PROJECT_ROOT)
    run_directory = create_run_directory(args.results_dir, args.profile_config.name)
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "platform": "huawei_ascend_910c",
        "profile": args.profile_config.name,
        "distributions": args.distributions,
        "shape_map": args.shape_map,
        "seed": args.seed,
        "device": args.device,
        "block_size": args.block_size,
        "num_warps": args.num_warps,
        "acceptance_profile": args.acceptance_profile,
        "skip_cann_baseline": args.skip_cann_baseline,
        "preflight_only": args.preflight_only,
        "diagnostic_overrides": {
            "allow_non_cann9": args.allow_non_cann9,
            "allow_unverified_910c": args.allow_unverified_910c,
        },
        "formal_run_eligible": not (args.allow_non_cann9 or args.allow_unverified_910c),
        "hardware_excluded_distributions": ASCEND_910C_UNSUPPORTED_DISTRIBUTIONS,
        "huawei_f64_contract": {
            "uniform_f64": "FP32 random resolution stored in binary64",
            "normal_f64": "FP32 random resolution stored in binary64",
            "lognormal_f64": "FP32-resolution Normal followed by FP64 Exp",
            "comparison_note": (
                "FP64 uses the CANN-native compatibility path and is not an "
                "independent implementation speed comparison"
            ),
        },
        "timing": {
            "scope": TIMING_SCOPE,
            "formal_metric": "total_ms_median",
            "speedup_formula": ("cann_aclnn total_ms_median / FlagRAND total_ms_median"),
            "api_aggregation": "arithmetic mean over configured shapes",
            "cross_api_average": None,
        },
    }

    try:
        torch_npu = import_torch_npu()
        manifest["torch_npu_module_version"] = getattr(torch_npu, "__version__", "unknown")
        select_device(args.device)
        manifest["source"] = verify_source(PROJECT_ROOT)
        manifest["environment"] = collect_and_validate_environment(
            args.device,
            allow_non_cann9=args.allow_non_cann9,
            allow_unverified_910c=args.allow_unverified_910c,
            require_baseline=not args.skip_cann_baseline,
        )
        for warning in manifest["environment"].get("warnings", ()):
            LOGGER.warning("ENV WARNING: %s", warning)
        for note in manifest["environment"].get("notes", ()):
            LOGGER.info("ENV NOTE: %s", note)
        if manifest["environment"]["status"] != "pass":
            raise RuntimeError("910C environment preflight failed: " + " | ".join(manifest["environment"]["errors"]))

        from benchmark.ascend_910c.cases import annotate_speedup, run_cann_case, run_flagrand_case, run_preflight

        device = f"npu:{args.device}"
        manifest["preflight"] = run_preflight(
            args.seed,
            device,
            distributions=args.distributions,
            block_size=args.block_size,
            num_warps=args.num_warps,
            with_baseline=not args.skip_cann_baseline,
        )
        LOGGER.info("Preflight: %s", manifest["preflight"]["status"])
        if manifest["preflight"]["status"] != "pass":
            write_results(run_directory, manifest, records)
            LOGGER.info("Results: %s", run_directory)
            return 2
        if args.preflight_only:
            write_results(run_directory, manifest, records)
            LOGGER.info("Results: %s", run_directory)
            return 0

        for distribution, n in iter_cases(args):
            if not supports_size(distribution, n):
                LOGGER.info("SKIP %-15s N=%-9d requires even N", distribution, n)
                continue
            flagrand_record = run_flagrand_case(
                distribution,
                n,
                args.seed,
                device,
                args.profile_config,
                block_size=args.block_size,
                num_warps=args.num_warps,
            )
            baseline_record = None
            if not args.skip_cann_baseline:
                baseline_record = run_cann_case(
                    distribution,
                    n,
                    args.seed,
                    device,
                    args.profile_config,
                )
                annotate_speedup(flagrand_record, baseline_record)
            records.append(flagrand_record)
            print_record(flagrand_record)
            if baseline_record is not None:
                records.append(baseline_record)
                print_record(baseline_record)

        if not args.skip_cann_baseline:
            manifest["api_speedup_summary"] = build_api_summary(
                records,
                distributions=args.distributions,
                shape_map=args.shape_map,
                speedup_min=args.speedup_min,
                speedup_max=args.speedup_max,
            )
            print_api_summary(manifest["api_speedup_summary"])
    except Exception as exc:
        manifest["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
        write_diagnostics(run_directory, manifest, records)
        LOGGER.exception("Fatal error: %s: %s", type(exc).__name__, exc)
        LOGGER.info("Results: %s", run_directory)
        return 3

    write_results(run_directory, manifest, records)
    failures = sum(record.get("status") != "pass" for record in records)
    threshold_failure = manifest.get("api_speedup_summary", {}).get("status") == "fail"
    LOGGER.info("Completed %d records with %d failure(s).", len(records), failures)
    LOGGER.info("Results: %s", run_directory)
    return 1 if failures or threshold_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
