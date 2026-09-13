from __future__ import annotations

import torch

from benchmark.hygon_matrix.baseline import HiprandGenerator, hiprand_library_path
from benchmark.hygon_matrix.config import ALL_DISTRIBUTIONS
from benchmark.hygon_matrix.reference import reference
from benchmark.hygon_matrix.timing import measure
from benchmark.philox_common import (
    DTYPES,
    _base_record,
    _capture_error,
    _throughput,
    invoke_flagrand,
    make_flagrand_generator,
    validate_output,
)


def validate_reference(output, distribution, seed, offset=0):
    name, dtype = ("raw", "int32") if distribution == "raw32" else distribution.rsplit("_", 1)
    dtype = {"f32": "float32", "f64": "float64"}.get(dtype, dtype)
    n = min(output.numel(), 128)
    expected = reference(name, dtype, n, seed, offset)
    actual = output[:n].detach().cpu()
    if name == "raw":
        ok = torch.equal(actual, expected)
    else:
        rtol, atol = (1e-12, 1e-13) if dtype == "float64" else (2e-5, 2e-6)
        ok = bool(torch.allclose(actual.double(), expected.double(), rtol=rtol, atol=atol))
    return {
        "status": "pass" if ok else "fail",
        "sample_n": n,
        "reference": "independent CPU Philox integer arithmetic and distribution transform",
        "max_abs": float((actual.double() - expected.double()).abs().max()),
    }


def warm_flagrand_compilation_paths(seed, distributions=ALL_DISTRIBUTIONS):
    probes = []
    for distribution in distributions:
        n = 8 if distribution.startswith(("normal", "lognormal")) else 7
        for offset in (0, 1, 2, 3, (1 << 34) - 4):
            out = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
            invoke_flagrand(distribution, make_flagrand_generator(seed, offset=offset), out)
            check = validate_reference(out, distribution, seed, offset)
            probes.append(dict(distribution=distribution, n=n, offset=offset, **check))
    return {"status": "pass" if all(p["status"] == "pass" for p in probes) else "fail", "probes": probes}


def run_case(backend, distribution, n, seed, profile):
    record = _base_record(backend, distribution, n)
    record.update(
        seed=seed,
        implementation_path="hygon_hiprand_host_philox" if backend == "hygon_hiprand" else "hygon_triton_public_philox",
    )
    factory = HiprandGenerator if backend == "hygon_hiprand" else make_flagrand_generator

    def invoke(generator, output):
        if backend == "hygon_hiprand":
            generator.generate(distribution, output)
        else:
            invoke_flagrand(distribution, generator, output)

    try:
        output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        replay = torch.empty_like(output)
        for target in (output, replay):
            generator = factory(seed=seed)
            try:
                invoke(generator, target)
                torch.cuda.synchronize()
            finally:
                if backend == "hygon_hiprand":
                    generator.close()
        validation = validate_output(output, replay, distribution)
        if backend == "flagrand_public":
            validation["cpu_reference"] = validate_reference(output, distribution, seed)
            if validation["cpu_reference"]["status"] != "pass":
                validation["status"] = "fail"
        else:
            record["library_path"] = hiprand_library_path()
        record["validation"] = validation
        if validation["status"] != "pass":
            record["status"] = "validation_fail"
            return record
        # Identical timing contract: persistent generator, advancing offset/state,
        # current stream, preallocated output, host API through final synchronize.
        generator = factory(seed=seed)
        try:
            record.update(measure(lambda: invoke(generator, output), profile))
        finally:
            if backend == "hygon_hiprand":
                generator.close()
        record.update(_throughput(record, output.element_size(), n), status="pass")
    except Exception as exc:
        _capture_error(record, exc)
    return record


def annotate_speedup(flagrand, baseline):
    if flagrand["status"] == "pass" and baseline["status"] == "pass":
        flagrand["speedup_vs_hygon_hiprand"] = baseline["total_ms_median"] / flagrand["total_ms_median"]
        flagrand["device_speedup_vs_hygon_hiprand"] = baseline["device_ms_median"] / flagrand["device_ms_median"]
        baseline["speedup_reference_for"] = "flagrand_public"
