"""Public-API correctness and same-device native RNG comparisons."""

import math
import traceback

import torch

from benchmark.muxi_matrix.timing import measure
from benchmark.rng_delivery.config import NATIVE_TYPES, split_operator
from benchmark.rng_delivery.native import discover_native_support, native_provider
from flagrand import curand

DTYPES = {
    "raw32": torch.int32,
    "uniform_f32": torch.float32,
    "uniform_f64": torch.float64,
    "normal_f32": torch.float32,
    "normal_f64": torch.float64,
    "lognormal_f32": torch.float32,
    "lognormal_f64": torch.float64,
    "poisson": torch.int32,
}
METHODS = {
    "raw32": "generate",
    "uniform_f32": "generate_uniform",
    "uniform_f64": "generate_uniform_double",
    "normal_f32": "generate_normal",
    "normal_f64": "generate_normal_double",
    "lognormal_f32": "generate_lognormal",
    "lognormal_f64": "generate_lognormal_double",
    "poisson": "generate_poisson",
}


def invoke(generator, output, api, lam=None):
    function = getattr(curand, METHODS[api])
    if api == "poisson":
        return function(generator, output, lam)
    if api.startswith(("normal", "lognormal")):
        return function(generator, output, mean=0.0, stddev=1.0)
    return function(generator, output)


def poisson_cdf(lam, cutoff):
    """Independent Poisson CDF, evaluated in log space through k=floor(lambda)."""
    return math.fsum(math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) for k in range(cutoff + 1))


def validate_output(output, replay, api, lam=None):
    values = output.detach().cpu().double()
    checks = {"reproducible": bool(torch.equal(output, replay))}
    metrics = {
        "n": values.numel(),
        "first_values": values[:8].tolist(),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    if api == "raw32":
        checks["not_all_zero"] = bool((values != 0).any())
        checks["written"] = bool((values != -2147483648).any())
        metrics["unique"] = int(values.unique().numel())
        return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "metrics": metrics}
    checks["finite"] = bool(torch.isfinite(values).all())
    if api.startswith("uniform"):
        checks["range"] = bool(((values > 0) & (values <= 1)).all())
        mean0, var0, fourth = 0.5, 1 / 12, 1 / 80
        cutoff, cdf0 = 0.5, 0.5
    elif api == "poisson":
        checks["range"] = bool((values >= 0).all())
        mean0 = var0 = lam
        fourth = lam + 3 * lam * lam
        cutoff = math.floor(lam)
        cdf0 = poisson_cdf(lam, cutoff)
    else:
        if api.startswith("lognormal"):
            checks["positive"] = bool((values > 0).all())
            metrics["output_mean"] = float(values.mean())
            values = values.log()
            metrics["statistics_domain"] = "log(output)"
        mean0, var0, fourth = 0.0, 1.0, 3.0
        cutoff, cdf0 = 0.0, 0.5
    n = values.numel()
    mean, variance = float(values.mean()), float(values.var(unbiased=False))
    observed = float((values <= cutoff).double().mean())
    mean_z = (mean - mean0) / math.sqrt(var0 / n)
    # Population sample variance: E[v]=sigma^2*(n-1)/n.
    variance_se = math.sqrt(((n - 1) ** 2 * fourth - (n - 1) * (n - 3) * var0**2) / n**3)
    variance_z = (variance - var0 * (n - 1) / n) / variance_se if n > 1 else 0.0
    cdf_z = (observed - cdf0) / math.sqrt(cdf0 * (1 - cdf0) / n)
    metrics.update(
        mean=mean,
        variance=variance,
        expected_mean=mean0,
        expected_variance=var0,
        mean_z=mean_z,
        variance_z=variance_z,
        cdf_at=cutoff,
        cdf_observed=observed,
        cdf_reference=cdf0,
        cdf_z=cdf_z,
        statistics_domain=metrics.get("statistics_domain", "output"),
    )
    if n >= 4096:
        checks.update(
            mean_six_sigma=abs(mean_z) <= 6, variance_six_sigma=abs(variance_z) <= 6, cdf_six_sigma=abs(cdf_z) <= 6
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": metrics,
        "statistical_rule": "For N>=4096: abs(mean_z), abs(variance_z), abs(single_point_cdf_z) <= 6",
    }


def run_case(operator, n, lam, seed, profile, *, native=False, platform="metax", native_support=None):
    engine, api = split_operator(operator)
    baseline_name, generator_class, library_path, library_version = native_provider(platform, engine, api)
    record = {
        "backend": baseline_name if native else "flagrand_public",
        "generator": engine,
        "distribution": operator,
        "api": api,
        "dtype": str(DTYPES[api]).removeprefix("torch."),
        "n": n,
        "seed": seed,
        "lambda": lam,
        "status": "pending",
    }
    if native and baseline_name == "ppu_acrand":
        if native_support is None:
            native_support = {}
        observation = discover_native_support(native_support, engine, api, library_path())
        if observation["status"] != "pass":
            record.update(
                status=observation["status"],
                native_rng_type=NATIVE_TYPES[engine],
                reason=observation.get("reason") or observation.get("error") or observation["status"],
                native_probe=observation,
            )
            return record
    try:
        sentinel = -2147483648 if DTYPES[api] == torch.int32 else math.nan
        output = torch.full((n,), sentinel, dtype=DTYPES[api], device="cuda")
        replay = torch.full_like(output, sentinel)
        if native:
            record.update(
                library_path=library_path(),
                library_version=library_version(),
                native_rng_type=NATIVE_TYPES[engine],
            )
            with generator_class(seed=seed, rng_type=NATIVE_TYPES[engine]) as generator:
                generator.generate(api, output, lam=10.0 if lam is None else lam)
                if baseline_name in {"ppu_sdk_mtgp32_device", "corex_sdk_corrected_mrg", "corex_sdk_raw_alias_poisson"}:
                    record["baseline_metadata"] = generator.metadata()
            with generator_class(seed=seed, rng_type=NATIVE_TYPES[engine]) as generator:
                generator.generate(api, replay, lam=10.0 if lam is None else lam)
        else:
            generator = curand.create_generator(engine, seed=seed)
            invoke(generator, output, api, lam)
            curand.set_pseudo_random_generator_seed(generator, seed)
            invoke(generator, replay, api, lam)
        record["validation"] = validate_output(output, replay, api, lam)
        # Preserve timings even when correctness fails; status identifies the failure.
        if native:
            with generator_class(seed=seed, rng_type=NATIVE_TYPES[engine]) as generator:
                record.update(
                    measure(lambda: generator.generate(api, output, lam=10.0 if lam is None else lam), profile)
                )
        else:
            generator = curand.create_generator(engine, seed=seed)
            record.update(measure(lambda: invoke(generator, output, api, lam), profile))
            if engine == "xorwow":
                record["distribution_workspace"] = {
                    "streams": len(generator.engine._distribution_workspaces),
                    "bytes": sum(
                        entry[1].numel() * entry[1].element_size()
                        for entry in generator.engine._distribution_workspaces.values()
                    ),
                    "allocation_scope": "first allocation or capacity growth occurs in warmup before timed samples",
                }
            if engine == "mtgp32":
                record["raw_buffer"] = {
                    "stats": getattr(generator.engine, "_buffer_stats", None),
                    "consumed_words": generator.engine.offset,
                    "physical_frontier": generator.engine._ws_next_block_start,
                }
        record["status"] = "pass" if record["validation"]["status"] == "pass" else "validation_fail"
    except Exception as exc:
        record.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback_tail=traceback.format_exc().splitlines()[-12:],
        )
        if native and platform == "iluvatar":
            from benchmark.corex_matrix.baseline import CorexCurandError

            if isinstance(exc, CorexCurandError):
                record.update(native_return_code=exc.status, native_operation=exc.operation)
                if exc.status == 106:
                    record.update(status="unsupported", reason="CURAND_STATUS_DOUBLE_PRECISION_REQUIRED (106)")
    return record
