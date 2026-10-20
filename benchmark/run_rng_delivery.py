"""Run the fixed C550 family matrix with the existing total-API timing protocol."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from benchmark.rng_delivery.config import OPERATORS, PROFILES, parameter_values, shapes_for, split_operator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--distributions", default=",".join(OPERATORS))
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--results-dir", type=Path, default=Path("rng_results"))
    args = parser.parse_args(argv)

    from benchmark.muxi_matrix.compat import collect_environment
    from benchmark.rng_delivery.cases import run_case
    from benchmark.rng_delivery.reporting import write_results

    operators = args.distributions.split(",")
    unknown = set(operators) - set(OPERATORS)
    if unknown:
        parser.error(f"unknown operator IDs: {sorted(unknown)}")
    run_directory = args.results_dir / f"{datetime.now():%Y%m%d_%H%M%S}_rng_{args.profile}"
    run_directory.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    source = sorted((root / "src/flagrand").rglob("*.py"))
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
    manifest = {
        "platform": "metax",
        "acceptance_baseline": "mcrand",
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
    }
    records = []
    write_results(run_directory, manifest, records)
    for operator in operators:
        _, api = split_operator(operator)
        for lam in parameter_values(api):
            for n in shapes_for(api, args.profile):
                candidate = run_case(operator, n, lam, args.seed, PROFILES[args.profile])
                baseline = run_case(operator, n, lam, args.seed, PROFILES[args.profile], native=True)
                if candidate["status"] == baseline["status"] == "pass":
                    candidate["speedup_vs_mcrand"] = baseline["total_ms_median"] / candidate["total_ms_median"]
                    candidate["device_speedup_vs_mcrand"] = baseline["device_ms_median"] / candidate["device_ms_median"]
                records.extend((candidate, baseline))
                print(
                    json.dumps(
                        {
                            "operator": operator,
                            "n": n,
                            "lambda": lam,
                            "candidate_status": candidate["status"],
                            "baseline_status": baseline["status"],
                            "speedup": candidate.get("speedup_vs_mcrand"),
                        }
                    ),
                    flush=True,
                )
                write_results(run_directory, manifest, records)
    manifest["finished_at"] = datetime.now().isoformat()
    write_results(run_directory, manifest, records)
    print(f"Results: {run_directory}", flush=True)
    return int(any(record["status"] != "pass" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
