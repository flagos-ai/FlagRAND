"""Run the fixed RNG-family matrix with the existing total-API timing protocol."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from benchmark.rng_delivery.config import (
    COREX_OPERATORS,
    OPERATORS,
    PROFILES,
    parameter_values,
    shapes_for,
    split_operator,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--platform", choices=("metax", "thead", "iluvatar", "hygon"), default="metax")
    parser.add_argument("--native-support", type=Path, default=os.environ.get("FLAGRAND_PPU_NATIVE_SUPPORT"))
    parser.add_argument("--distributions", default=None)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--results-dir", type=Path, default=Path("rng_results"))
    args = parser.parse_args(argv)

    defaults = COREX_OPERATORS if args.platform == "iluvatar" else OPERATORS
    operators = args.distributions.split(",") if args.distributions is not None else list(defaults)
    unknown = set(operators) - set(OPERATORS)
    if unknown:
        parser.error(f"unknown operator IDs: {sorted(unknown)}")
    if args.platform == "iluvatar" and any(op.endswith("_f64") for op in operators):
        parser.error("BI-V150 current scope excludes FP64 APIs; select raw32, FP32 distributions or Poisson")

    from benchmark.muxi_matrix.compat import collect_environment
    from benchmark.rng_delivery.cases import run_case
    from benchmark.rng_delivery.native import discover_native_support, native_provider
    from benchmark.rng_delivery.reporting import write_results

    run_directory = args.results_dir / f"{datetime.now():%Y%m%d_%H%M%S}_rng_{args.profile}"
    run_directory.mkdir(parents=True)
    mtgp32_build = None
    corex_build = None
    if args.platform == "thead" and any(split_operator(op)[0] == "mtgp32" for op in operators):
        from benchmark.ppu_matrix.build import baseline_build_record, prepare_mtgp32_baseline

        prepare_mtgp32_baseline(run_directory / "native-build/mtgp32")
        mtgp32_build = baseline_build_record()
    if args.platform == "iluvatar":
        from benchmark.corex_matrix.corrected_build import prepare_operator_references

        corex_build = prepare_operator_references(operators, run_directory / "native-build/corex-corrected")
    native_support = json.loads(args.native_support.read_text()) if args.native_support is not None else {}
    if args.platform == "thead" and native_support:
        os.environ["FLAGRAND_ACRAND_PATH"] = native_support["environment"]["native_library"]
    baselines = {operator: native_provider(args.platform, *split_operator(operator))[0] for operator in operators}
    baseline_names = set(baselines.values())
    if args.native_support is not None:
        (run_directory / "native-support.json").write_text(json.dumps(native_support, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    source = sorted((root / "src/flagrand").rglob("*.py"))
    source += sorted((root / "src/flagrand").rglob("*.cpp"))
    source += sorted((root / "benchmark/rng_delivery").glob("*.py")) + [Path(__file__).resolve()]
    source += [
        root / path
        for path in (
            "benchmark/muxi_matrix/timing.py",
            "benchmark/muxi_matrix/mcrand.py",
            "benchmark/nvidia_matrix/config.py",
            "benchmark/summary_json.py",
            "tools/run_tests.py",
            "tests/accuracy/test_rng_families.py",
            "tests/conftest.py",
            "conf/operators.yaml",
        )
    ]
    if args.platform == "hygon":
        source += sorted((root / "benchmark/hygon_matrix").glob("*.py"))
    if args.platform == "thead":
        source += sorted((root / "benchmark/ppu_matrix").glob("*.py"))
        source += sorted((root / "benchmark/ppu_matrix").glob("*.hg"))
        source += [root / "tools/build_ppu_mtgp32_baseline.py"]
    elif args.platform == "iluvatar":
        source += sorted((root / "benchmark/corex_matrix").glob("*.py"))
        source += sorted((root / "benchmark/corex_matrix").glob("*.cu"))
        source += sorted((root / "benchmark/corex_matrix").glob("*.cuh"))
        source += [root / "benchmark/corex_matrix/SDK-LICENSE.txt"]
        source += [root / "benchmark/ppu_matrix/curand.py"]
    manifest = {
        "platform": args.platform,
        "acceptance_baseline": next(iter(baseline_names)) if len(baseline_names) == 1 else "per_operator",
        "acceptance_baselines": baselines,
        "profile": args.profile,
        "distributions": operators,
        "seed": args.seed,
        "environment": collect_environment(),
        "timing": {
            "metric": "total_ms_median",
            "warmup": PROFILES[args.profile].warmup,
            "repeats": PROFILES[args.profile].repeats,
            "timing_budget_ms": PROFILES[args.profile].timing_budget_ms,
            "scope": "persistent generator, preallocated output; per-call wall time with final synchronization",
            "excluded": "initialization, first compilation, first Poisson table construction "
            "and first workspace allocation",
            "secondary_metric": "device_ms_median",
        },
        "shape_map": {op: list(shapes_for(split_operator(op)[1], args.profile)) for op in operators},
        "lambda_map": {op: list(parameter_values(split_operator(op)[1])) for op in operators},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        "native_support": {
            "source": str(args.native_support) if args.native_support is not None else None,
            "sha256": hashlib.sha256(args.native_support.read_bytes()).hexdigest() if args.native_support else None,
        },
    }
    if mtgp32_build is not None:
        manifest["native_build"] = {"ppu_sdk_mtgp32_device": mtgp32_build}
    if args.platform == "iluvatar" and "corex_curand" in baseline_names:
        from benchmark.corex_matrix.baseline import corex_stock_metadata

        manifest["corex_stock"] = corex_stock_metadata()
    if corex_build is not None:
        manifest["native_build"] = {"corex_sdk_corrected": corex_build}
    records = []
    write_results(run_directory, manifest, records)
    for operator in operators:
        engine, api = split_operator(operator)
        baseline_name = baselines[operator]
        if baseline_name == "ppu_acrand":
            library_path = native_provider(args.platform, engine, api)[2]()
            discover_native_support(native_support, engine, api, library_path)
            support_path = run_directory / "native-support.json"
            support_path.write_text(json.dumps(native_support, indent=2) + "\n")
            manifest["native_support"].update(
                recorded_file=support_path.name,
                recorded_sha256=hashlib.sha256(support_path.read_bytes()).hexdigest(),
            )
        for lam in parameter_values(api):
            for n in shapes_for(api, args.profile):
                candidate = run_case(operator, n, lam, args.seed, PROFILES[args.profile], platform=args.platform)
                baseline = run_case(
                    operator,
                    n,
                    lam,
                    args.seed,
                    PROFILES[args.profile],
                    native=True,
                    platform=args.platform,
                    native_support=native_support,
                )
                if candidate["status"] == baseline["status"] == "pass":
                    candidate[f"speedup_vs_{baseline_name}"] = (
                        baseline["total_ms_median"] / candidate["total_ms_median"]
                    )
                    candidate[f"device_speedup_vs_{baseline_name}"] = (
                        baseline["device_ms_median"] / candidate["device_ms_median"]
                    )
                records.extend((candidate, baseline))
                print(
                    json.dumps(
                        {
                            "operator": operator,
                            "n": n,
                            "lambda": lam,
                            "candidate_status": candidate["status"],
                            "baseline_status": baseline["status"],
                            "speedup": candidate.get(f"speedup_vs_{baseline_name}"),
                        }
                    ),
                    flush=True,
                )
                write_results(run_directory, manifest, records)
    manifest["finished_at"] = datetime.now().isoformat()
    write_results(run_directory, manifest, records)
    print(f"Results: {run_directory}", flush=True)
    return int(any(record["status"] not in {"pass", "unsupported"} for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
