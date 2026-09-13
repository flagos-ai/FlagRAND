"""Seven-API FlagRAND/Hygon hipRAND matrix with the shared FlagOS summary schema."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import torch
import triton
from triton.runtime.driver import driver

from benchmark.hygon_matrix.baseline import hiprand_library_path, hiprand_version
from benchmark.hygon_matrix.cases import annotate_speedup, run_case, warm_flagrand_compilation_paths
from benchmark.hygon_matrix.config import parse_args
from benchmark.hygon_matrix.reporting import (
    build_api_summary,
    create_run_directory,
    print_api_summary,
    print_record,
    write_results,
)
from flagrand.runtime import current_vendor

ROOT = Path(__file__).resolve().parents[1]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args(ROOT)
    torch.cuda.set_device(args.device)
    directory = create_run_directory(args.results_dir, args.profile_config.name)
    records = []
    manifest = {
        "platform": "hygon_dtk_hip",
        "acceptance_baseline": "hygon_hiprand",
        "profile": args.profile_config.name,
        "shape_map": args.shape_map,
        "distributions": args.distributions,
        "seed": args.seed,
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted((ROOT / "src").rglob("*.py")))
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for folder in ("benchmark", "tools") for p in sorted((ROOT / folder).rglob("*.py")))
        ).hexdigest(),
        "timing": {
            "scope": "persistent generator, advancing RNG state and preallocated output",
            "formal_metric": "total_ms_median",
            "compilation": "warmup excluded",
            "speedup_formula": "hygon_hiprand total_ms_median / FlagRAND total_ms_median",
            "api_aggregation": "arithmetic mean of per-shape speedups",
            "cross_api_average": None,
        },
        "semantics": {
            "flagrand": "portable Philox stream; checked against independent CPU oracle",
            "baseline": "native hipRAND Philox host generator with default ordering",
            "native_sequence_equivalence": (
                "not required or asserted; each implementation is tested for replay and distribution"
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(args.device),
            "device_count": torch.cuda.device_count(),
            "vendor": current_vendor(),
            "target": str(driver.active.get_current_target()),
            "hip": torch.version.hip,
            "hiprand_library": hiprand_library_path(),
            "hiprand_version": hiprand_version(),
        },
    }
    manifest["preflight"] = warm_flagrand_compilation_paths(args.seed, args.distributions)
    if args.preflight_only:
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("Preflight:", manifest["preflight"]["status"], "Results:", directory)
        return int(manifest["preflight"]["status"] != "pass")
    for distribution in args.distributions:
        for index, n in enumerate(args.shape_map[distribution]):
            # Alternate order to reduce systematic warm-device/order bias.
            order = ["flagrand_public", "hygon_hiprand"] if index % 2 == 0 else ["hygon_hiprand", "flagrand_public"]
            pair = {backend: run_case(backend, distribution, n, args.seed, args.profile_config) for backend in order}
            annotate_speedup(pair["flagrand_public"], pair["hygon_hiprand"])
            for backend in order:
                records.append(pair[backend])
                print_record(pair[backend])
            manifest["api_speedup_summary"] = build_api_summary(records, args.distributions)
            write_results(directory, manifest, records)
    print_api_summary(manifest["api_speedup_summary"])
    print("Results:", directory)
    return int(manifest["preflight"]["status"] != "pass" or any(r["status"] != "pass" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
