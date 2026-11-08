"""Compare RNG families with CANN native C distribution APIs on Ascend 910C."""

import argparse
import hashlib
import json
import math
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from benchmark.ascend_910c.native import NativeCannGenerator, baseline_metadata
from benchmark.ascend_910c.runtime import NativeRuntimeGenerator
from benchmark.ascend_910c.runtime import baseline_metadata as runtime_metadata
from benchmark.ascend_910c.timing import measure
from benchmark.rng_delivery.cases import DTYPES, invoke, validate_output
from benchmark.rng_delivery.config import OPERATORS, PROFILES, parameter_values, shapes_for, split_operator
from flagrand import curand


def native_configuration(api, dtype, provider):
    distribution = api.split("_", 1)[0]
    use_runtime = provider == "runtime" and (api == "raw32" or api.endswith("_f32"))
    generator_type = NativeRuntimeGenerator if use_runtime else NativeCannGenerator
    metadata = (runtime_metadata if use_runtime else baseline_metadata)(distribution, dtype)
    metadata["requested_provider"] = provider
    metadata["selected_provider"] = "runtime" if use_runtime else "graph"
    if provider == "runtime" and not use_runtime:
        metadata["selection_reason"] = "Runtime has no FP64 or Poisson entry; existing graph/ACLNN path retained"
    return generator_type, metadata


def run_case(operator, n, lam, seed, profile, build_dir, native, native_provider="graph"):
    engine, api = split_operator(operator)
    record = {
        "backend": "cann_native_c" if native else "flagrand_public",
        "generator": engine,
        "distribution": operator,
        "api": api,
        "dtype": str(DTYPES[api]).removeprefix("torch."),
        "n": n,
        "lambda": lam,
        "seed": seed,
    }
    if native:
        generator_type, metadata = native_configuration(api, DTYPES[api], native_provider)
        record.update(
            baseline_metadata=metadata,
            native_provider_requested=native_provider,
            native_provider=metadata["selected_provider"],
        )
    try:
        output = torch.empty(n, dtype=DTYPES[api], device="npu")
        if native:
            distribution = api.split("_", 1)[0]
            with generator_type(
                output, distribution, seed, lam, Path(build_dir) / record["native_provider"]
            ) as generator:
                record.update(
                    library_path=generator.library_path,
                    setup_and_first_call_ms=generator.setup_and_first_call_ms,
                    workspace_bytes=generator.workspace_bytes,
                )
                generator.reset()
                generator.generate()
                generator.synchronize()
                first = output.cpu()
                generator.reset()
                generator.generate()
                generator.synchronize()
                replay = output.cpu()
                # Keep the existing validation interval for both explicit providers.
                record["validation"] = validate_output(first, replay, api, lam, uniform_interval="[0, 1)")
                if distribution == "uniform":
                    record["baseline_metadata"]["uniform_validation_interval"] = "[0, 1)"
                    if record["native_provider"] == "graph":
                        record["baseline_metadata"]["uniform_interval"] = "[0, 1)"
                if api == "raw32":
                    record["validation"]["metrics"]["low_byte_unique"] = int((first & 255).unique().numel())
                    record["validation"]["metrics"]["low_byte_zero_fraction"] = float(
                        ((first & 255) == 0).float().mean()
                    )
                record.update(measure(generator.generate, profile, synchronize=generator.synchronize))
        else:
            generator = curand.create_generator(engine, seed=seed)
            invoke(generator, output, api, lam)
            torch.npu.synchronize()
            first = output.cpu()
            curand.set_pseudo_random_generator_seed(generator, seed)
            invoke(generator, output, api, lam)
            torch.npu.synchronize()
            replay = output.cpu()
            record["validation"] = validate_output(first, replay, api, lam)
            record.update(measure(lambda: invoke(generator, output, api, lam), profile))
        record["status"] = "pass" if record["validation"]["status"] == "pass" else "validation_fail"
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    return record


def summarize(records):
    pairs = defaultdict(dict)
    for row in records:
        pairs[(row["distribution"], row["n"], row["lambda"])][row["backend"]] = row
    ratios = defaultdict(list)
    by_lambda = defaultdict(list)
    for (operator, n, lam), pair in pairs.items():
        candidate, native = pair.get("flagrand_public"), pair.get("cann_native_c")
        if candidate and native and candidate["status"] == native["status"] == "pass":
            ratio = native["total_ms_median"] / candidate["total_ms_median"]
            ratios[operator].append(ratio)
            by_lambda[(operator, lam)].append(ratio)
    geometric_mean = lambda values: math.exp(math.fsum(math.log(x) for x in values) / len(values))
    per_api = {
        operator: {"paired_cases": len(values), "geometric_mean": geometric_mean(values)}
        for operator, values in ratios.items()
    }
    families = defaultdict(list)
    for operator, result in per_api.items():
        families[split_operator(operator)[0]].append(result["geometric_mean"])
    return {
        "scope": "CANN same-distribution native C APIs; no same-generator or bitstream equivalence",
        "aggregation": "equal weight per shape/lambda pair; family is equal weight per API geometric mean",
        "native_providers_requested": sorted(
            {row["native_provider_requested"] for row in records if row["backend"] == "cann_native_c"}
        ),
        "native_components": {
            row["distribution"]: row["baseline_metadata"]
            for row in records
            if row["backend"] == "cann_native_c"
        },
        "per_api": per_api,
        "per_lambda": [
            {"operator": op, "lambda": lam, "paired_cases": len(values), "geometric_mean": geometric_mean(values)}
            for (op, lam), values in by_lambda.items()
        ],
        "families": {
            key: {"api_count": len(values), "geometric_mean": geometric_mean(values)}
            for key, values in families.items()
        },
        "status_counts": {
            backend: {
                status: sum(row["backend"] == backend and row["status"] == status for row in records)
                for status in ("pass", "validation_fail", "error")
            }
            for backend in ("flagrand_public", "cann_native_c")
        },
        "incomplete_pairs": len(pairs) - sum(len(values) for values in ratios.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--distributions", default=",".join(OPERATORS))
    parser.add_argument("--sizes", help="Optional comma-separated sizes for a focused experiment")
    parser.add_argument("--lambdas", help="Optional comma-separated Poisson rates")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--native-provider",
        choices=("graph", "runtime"),
        default="graph",
        help="runtime uses direct random tasks for raw32/FP32; FP64 and Poisson retain graph/ACLNN",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("ascend_rng_results"))
    args = parser.parse_args()
    operators = args.distributions.split(",")
    unknown = set(operators) - set(OPERATORS)
    if unknown:
        parser.error(f"Unknown operators: {sorted(unknown)}")
    torch.npu.set_device(args.device)
    directory = args.results_dir / f"{datetime.now():%Y%m%d_%H%M%S}_rng_{args.profile}"
    directory.mkdir(parents=True)
    shape_map = {
        op: (
            list(map(int, args.sizes.split(",")))
            if args.sizes
            else list(shapes_for(split_operator(op)[1], args.profile))
        )
        for op in operators
    }
    lambda_map = {
        op: (
            list(map(float, args.lambdas.split(",")))
            if args.lambdas and op.endswith("_poisson")
            else list(parameter_values(split_operator(op)[1]))
        )
        for op in operators
    }
    root = Path(__file__).resolve().parents[1]
    sources = (
        sorted((root / "src/flagrand").rglob("*.py"))
        + sorted((root / "benchmark/ascend_910c").glob("*.*"))
        + [Path(__file__).resolve()]
    )
    profile = PROFILES[args.profile]
    manifest = {
        "platform": "ascend",
        "device": torch.npu.get_device_name(args.device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "profile": vars(profile),
        "seed": args.seed,
        "shape_map": shape_map,
        "lambda_map": lambda_map,
        "baseline": "CANN native C, same-distribution comparison",
        "native_provider_requested": args.native_provider,
        "native_components": {
            op: native_configuration(split_operator(op)[1], DTYPES[split_operator(op)[1]], args.native_provider)[1]
            for op in operators
        },
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (directory / "records.jsonl").open("w") as stream:
        for op in operators:
            for n in shape_map[op]:
                for lam in lambda_map[op]:
                    for native in (False, True):
                        print(f"BEGIN {op} n={n} lambda={lam} native={native}", flush=True)
                        row = run_case(
                            op, n, lam, args.seed, profile, directory / "native-build", native, args.native_provider
                        )
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        records.append(row)
                        print(
                            json.dumps(
                                {
                                    key: row.get(key)
                                    for key in (
                                        "backend",
                                        "native_provider",
                                        "distribution",
                                        "n",
                                        "lambda",
                                        "status",
                                        "total_ms_median",
                                        "error_type",
                                    )
                                }
                            ),
                            flush=True,
                        )
    summary = summarize(records)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"results": str(directory), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
