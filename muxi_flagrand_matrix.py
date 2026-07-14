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
from muxi_matrix.reporting import create_run_directory, print_record, write_results


def main() -> int:
    args = parse_args(PROJECT_ROOT)
    run_directory = create_run_directory(args.results_dir, args.profile_config.name)
    manifest: dict[str, Any] = {
        "profile": args.profile_config.name,
        "sizes": args.profile_config.sizes,
        "distributions": args.distributions,
        "seed": args.seed,
        "preflight_only": args.preflight_only,
    }
    records: list[dict[str, Any]] = []

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false; no Muxi/CUDA device is visible.")
        manifest["source"] = verify_preview_source(PROJECT_ROOT)
        manifest["compatibility"] = apply_maca_compatibility()
        manifest["environment"] = collect_environment()

        from muxi_matrix.cases import run_flagrand_case, run_preflight, run_torch_references

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

        for n in args.profile_config.sizes:
            for distribution in args.distributions:
                record = run_flagrand_case(distribution, n, args.seed, args.profile_config)
                records.append(record)
                print_record(record)
            if not args.skip_torch_reference:
                for record in run_torch_references(n, args.profile_config):
                    records.append(record)
                    print_record(record)
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


if __name__ == "__main__":
    raise SystemExit(main())
