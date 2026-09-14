from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import triton

from benchmark.corex_matrix.baseline import corex_curand_library_path, probe_native_fp64
from benchmark.corex_matrix.cases import (
    annotate_speedup,
    run_corex_curand_case,
    run_flagrand_case,
    warm_flagrand_compilation_paths,
)
from benchmark.corex_matrix.config import parse_args
from benchmark.corex_matrix.reporting import (
    build_api_summary,
    create_run_directory,
    print_api_summary,
    print_record,
    write_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args(PROJECT_ROOT)
    run_directory = create_run_directory(args.results_dir, args.profile_config.name)
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "platform": "iluvatar_bi_v150_corex",
        "profile": args.profile_config.name,
        "shape_map": args.shape_map,
        "distributions": args.distributions,
        "seed": args.seed,
        "timing": {
            "scope": "persistent generator and preallocated output",
            "compilation": "preflight compilation probes excluded",
            "formal_metric": "total_ms_median",
            "speedup_formula": "corex_curand total_ms_median / FlagRAND total_ms_median",
            "api_aggregation": "arithmetic mean over configured shapes",
            "cross_api_average": None,
        },
        "environment": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "corex_curand": corex_curand_library_path(),
        },
    }
    manifest["preflight"] = warm_flagrand_compilation_paths(args.seed)
    manifest['native_fp64_probe'] = probe_native_fp64(args.seed)

    for distribution in args.distributions:
        for n in args.shape_map[distribution]:
            flagrand_record = run_flagrand_case(distribution, n, args.seed, args.profile_config)
            baseline_record = run_corex_curand_case(distribution, n, args.seed, args.profile_config, manifest['native_fp64_probe'])
            annotate_speedup(flagrand_record, baseline_record)
            records.extend((flagrand_record, baseline_record))
            print_record(flagrand_record)
            print_record(baseline_record)

    manifest["api_speedup_summary"] = build_api_summary(records, args.distributions)
    print_api_summary(manifest["api_speedup_summary"])
    write_results(run_directory, manifest, records)
    failures = [record for record in records if record["status"] not in {"pass", "unsupported"}]
    LOGGER.info("Completed %d records with %d failure(s).", len(records), len(failures))
    LOGGER.info("Results: %s", run_directory)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
