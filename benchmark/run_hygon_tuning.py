"""Finite Hygon experiments; formal acceptance remains run_hygon.py's full matrix."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
from pathlib import Path


def main():
    ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import torch

    from benchmark.hygon_matrix.cases import annotate_speedup, run_case, warm_flagrand_compilation_paths
    from benchmark.hygon_matrix.config import ALL_DISTRIBUTIONS, Profile
    from benchmark.hygon_matrix.reference import reference
    from benchmark.hygon_matrix.timing import measure
    from benchmark.philox_common import DTYPES, invoke_flagrand, make_flagrand_generator
    from flagrand import curand
    from flagrand.runtime.backend._hygon import ops

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = Profile("tuning", (), 5, 5, 20.0)
    report = {"preflight": warm_flagrand_compilation_paths(1234), "comparisons": [], "sweep": [], "profiles": {}}

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))

    save()
    for distribution in ALL_DISTRIBUTIONS:
        for n in (4, 4096, 1048576, 16777216):
            ours = run_case("flagrand_public", distribution, n, 1234, profile)
            native = run_case("hygon_hiprand", distribution, n, 1234, profile)
            annotate_speedup(ours, native)
            report["comparisons"].append({"flagrand": ours, "native": native})
            print(distribution, n, ours.get("speedup_vs_hygon_hiprand"), flush=True)
            save()
        if distribution in ("raw32", "uniform_f64", "normal_f64"):
            out = torch.empty(4096, device="cuda", dtype=DTYPES[distribution])
            gen = make_flagrand_generator(1234)
            prof = cProfile.Profile()
            prof.enable()
            for _ in range(2000):
                invoke_flagrand(distribution, gen, out)
            torch.cuda.synchronize()
            prof.disable()
            output = io.StringIO()
            pstats.Stats(prof, stream=output).sort_stats("cumtime").print_stats(30)
            report["profiles"][distribution] = output.getvalue()
            save()
    for distribution in ("normal_f32", "lognormal_f32", "normal_f64", "lognormal_f64"):
        op = getattr(
            curand, "generate_" + distribution.split("_")[0] + ("_double" if distribution.endswith("f64") else "")
        )
        for n in (4096, 1048576, 16777216):
            for block, warps in (
                (64, 1),
                (128, 2),
                (128, 4),
                (256, 1),
                (256, 2),
                (256, 4),
                (512, 4),
                (1024, 4),
                (512, 8),
                (1024, 8),
            ):
                out = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
                gen = make_flagrand_generator(1234)
                op(gen, out, mean=0.0, stddev=1.0, block_size=block, num_warps=warps)
                expected = reference(
                    distribution.split("_")[0],
                    str(DTYPES[distribution]).split(".")[-1],
                    min(n, 4096),
                    1234,
                    0,
                    mean=0.0,
                    stddev=1.0,
                )
                torch.testing.assert_close(
                    out[: len(expected)].cpu().double(),
                    expected.double(),
                    rtol=1e-12 if distribution.endswith("f64") else 2e-5,
                    atol=1e-13 if distribution.endswith("f64") else 2e-6,
                )

                def fn():
                    return op(gen, out, mean=0.0, stddev=1.0, block_size=block, num_warps=warps)

                result = dict(distribution=distribution, n=n, block=block, warps=warps, **measure(fn, profile))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        fn()
                samples = []
                for _ in range(7):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 16)
                result["graph_ms"] = sorted(samples)[3]
                from flagrand.fused._internal import philox_direct

                launcher = (
                    ops._NORMAL
                    if distribution.endswith("f64")
                    else (
                        philox_direct._PHILOX_NORMAL_F32_LAUNCHER
                        if distribution.startswith("normal")
                        else philox_direct._PHILOX_LOGNORMAL_F32_LAUNCHER
                    )
                )
                compiled = next(reversed(launcher._keyed_entries.values())).compiled
                if n == 1048576:
                    stem = args.output / f"{distribution}-block{block}-warps{warps}"
                    for kind in ("ttgir", "llir", "amdgcn"):
                        if kind in compiled.asm:
                            stem.with_suffix("." + kind).write_text(compiled.asm[kind])
                result["metadata"] = {
                    key: getattr(compiled.metadata, key, None) for key in ("num_warps", "shared", "warp_size")
                }
                result["n_regs"] = compiled.n_regs
                result["n_spills"] = compiled.n_spills
                report["sweep"].append(result)
                print("sweep", distribution, n, block, warps, result["total_ms_median"], result["graph_ms"], flush=True)
                save()
    print("HYGON_TUNING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
