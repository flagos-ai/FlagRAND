"""Fixed-parameter corrected-baseline correctness probe, without performance timing."""

import argparse
import json
from pathlib import Path

import torch

from benchmark.corex_matrix.corrected import CorexCorrectedGenerator
from benchmark.corex_matrix.corrected_build import prepare_corrected_baseline
from benchmark.rng_delivery.cases import DTYPES, validate_output
from benchmark.rng_delivery.config import LAMBDAS, NATIVE_TYPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1048576)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_corrected_baseline(args.output_dir / "native-build")
    cases = [("mrg32k3a", api, None) for api in ("raw32", "uniform_f32", "normal_f32", "lognormal_f32")]
    cases += [(engine, "poisson", lam) for engine in NATIVE_TYPES for lam in LAMBDAS]
    result = {
        "status": "running",
        "scope": "CoreX SDK corrected reference; fixed API/parameters per context",
        "cases": [],
    }

    def save():
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def snapshot(generator, api, lam):
        torch.cuda.synchronize()
        return generator.state_snapshot(api, lam=lam)

    def differences(a, b):
        return int(torch.count_nonzero(a != b))

    for engine, api, lam in cases:
        value = 10.0 if lam is None else lam
        with CorexCorrectedGenerator(seed=args.seed, rng_type=NATIVE_TYPES[engine]) as generator:
            output = torch.empty(args.n, dtype=DTYPES[api], device="cuda")
            replay = torch.empty_like(output)
            split = torch.empty_like(output)
            continued = torch.empty(257, dtype=DTYPES[api], device="cuda")
            resumed = torch.empty_like(continued)
            generator.generate(api, output, lam=value)
            whole_state = snapshot(generator, api, value)
            generator.generate(api, continued, lam=value)
            continuation_state = snapshot(generator, api, value)
            generator.reset()
            generator.generate(api, replay, lam=value)
            replay_state = snapshot(generator, api, value)
            generator.reset()
            generator.generate(api, split[:17], lam=value)
            generator.generate(api, split[17:], lam=value)
            split_state = snapshot(generator, api, value)
            generator.generate(api, resumed, lam=value)
            resumed_state = snapshot(generator, api, value)
            row = {
                "engine": engine,
                "api": api,
                "lambda": lam,
                "n": args.n,
                "validation": validate_output(output, replay, api, lam),
                "reset_output_mismatches": differences(output, replay),
                "split_output_mismatches": differences(output, split),
                "continuation_output_mismatches": differences(continued, resumed),
                "reset_state_equal": whole_state == replay_state,
                "split_state_equal": whole_state == split_state,
                "continuation_state_equal": continuation_state == resumed_state,
                "metadata": generator.metadata(),
            }
            row["status"] = (
                "pass"
                if (
                    row["validation"]["status"] == "pass"
                    and not any(row[k] for k in row if k.endswith("_mismatches"))
                    and all(row[k] for k in row if k.endswith("_equal"))
                )
                else "validation_fail"
            )
            result["cases"].append(row)
            print(json.dumps({k: row[k] for k in ("engine", "api", "lambda", "status")}), flush=True)
            save()

    # A parameter switch creates another context. It must not consume or reuse
    # the first lambda's pending MTGP outputs; this is not shared-stream semantics.
    with (
        CorexCorrectedGenerator(seed=args.seed, rng_type=141) as mixed,
        CorexCorrectedGenerator(seed=args.seed, rng_type=141) as control,
    ):
        a = torch.empty(17, dtype=torch.int32, device="cuda")
        b = torch.empty_like(a)
        ignored = torch.empty(19, dtype=torch.int32, device="cuda")
        tail = torch.empty(257, dtype=torch.int32, device="cuda")
        expected = torch.empty_like(tail)
        mixed.generate("poisson", a, lam=0.1)
        mixed.generate("poisson", ignored, lam=10.0)
        mixed.generate("poisson", tail, lam=0.1)
        control.generate("poisson", b, lam=0.1)
        control.generate("poisson", expected, lam=0.1)
        result["independent_lambda_contexts"] = {
            "first_output_mismatches": differences(a, b),
            "resumed_output_mismatches": differences(tail, expected),
            "state_equal": snapshot(mixed, "poisson", 0.1) == snapshot(control, "poisson", 0.1),
            "semantics": "Independent fixed-lambda positions, not an original-SDK shared cross-lambda stream",
        }
    check = result["independent_lambda_contexts"]
    success = all(row["status"] == "pass" for row in result["cases"])
    success &= check["first_output_mismatches"] == check["resumed_output_mismatches"] == 0 and check["state_equal"]
    result["status"] = "pass" if success else "validation_fail"
    save()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
