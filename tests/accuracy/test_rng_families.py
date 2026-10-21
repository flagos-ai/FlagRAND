"""C550 public families: exact recurrence, continuation, tails and distributions."""

import math
import struct
from decimal import Decimal, localcontext

import pytest
import torch

from benchmark.rng_delivery.cases import DTYPES, invoke, validate_output
from benchmark.rng_delivery.config import ENGINES, ODD_TAILS, OPERATORS, parameter_values, split_operator
from flagrand import curand
from flagrand.runtime import current_vendor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or current_vendor() != "metax", reason="C550 MACA runtime is required"
)
CASES = [pytest.param(op, marks=getattr(pytest.mark, op)) for op in OPERATORS]


@pytest.mark.mtgp32_lognormal_f64
def test_metax_exp_degree13_decimal_boundaries(record_property):
    import triton
    import triton.language as tl

    from flagrand.rng._xorwow._binding import bind_kernel
    from flagrand.rng._xorwow.range_math64 import exp64, exp64_metax, exp64_metax_finite

    def probe(X, OUT, N, MODE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + i, i < N, other=0.0)
        if MODE == 2:
            result = exp64_metax_finite(value)
        elif MODE == 1:
            result = exp64_metax(value)
        else:
            result = exp64(value)
        tl.store(OUT + i, result, i < N)

    values = {-746.0, 710.0, -1e308, 1e308, -10.0, -1.0, 0.0, 1.0, 10.0, -700.0, 700.0}
    values.update((math.nextafter(-700.0, 0.0), math.nextafter(700.0, 0.0)))
    with localcontext() as context:
        context.prec = 80
        ln2 = Decimal(2).ln()
        for k in (-1075, -1074, -1022, -1021, -54, -1, 0, 1, 53, 1023, 1024):
            for offset in (Decimal(0), Decimal("0.5")):
                center = float((Decimal(k) + offset) * ln2)
                values.update((math.nextafter(center, -math.inf), center, math.nextafter(center, math.inf)))
        inputs = sorted(values) + [math.inf, -math.inf, math.nan]
        references = [
            (
                math.nan
                if math.isnan(value)
                else math.inf if value > 710.0 else 0.0 if value < -746.0 else float(Decimal.from_float(value).exp())
            )
            for value in inputs
        ]
    kernel = bind_kernel(probe)
    general_bits = {}
    for mode in (0, 1, 2):
        selected = [
            (value, expected) for value, expected in zip(inputs, references) if mode != 2 or -700.0 <= value <= 700.0
        ]
        x = torch.tensor([value for value, _ in selected], dtype=torch.float64, device="cuda")
        output = torch.empty_like(x)
        kernel[(triton.cdiv(len(selected), 256),)](
            x, output, len(selected), mode, 256, num_warps=4, enable_fp_fusion=False
        )
        actual = output.cpu().tolist()
        if mode == 1:
            general_bits = {
                sample: struct.pack("<d", value)
                for (sample, _), value in zip(selected, actual)
                if -700.0 <= sample <= 700.0
            }
        distances = []
        for value, (_, expected) in zip(actual, selected):
            if math.isnan(expected):
                assert math.isnan(value)
            elif math.isinf(expected):
                assert value == expected
            else:
                assert math.isfinite(value) and value >= 0.0
                word = struct.unpack("<Q", struct.pack("<d", value))[0]
                reference_word = struct.unpack("<Q", struct.pack("<d", expected))[0]
                distances.append(abs(word - reference_word))
        result = {"samples": len(selected), "max_ulp": max(distances), "degree": 13 if mode else 16}
        if mode == 2:
            result["exact_to_general_degree13"] = all(
                struct.pack("<d", value) == general_bits[sample] for (sample, _), value in zip(selected, actual)
            )
            assert result["exact_to_general_degree13"], result
        name = "exp_finite_range" if mode == 2 else f"exp_degree{result['degree']}"
        record_property(name, result)
        assert result["max_ulp"] <= 1, result


@pytest.mark.mtgp32_lognormal_f64
@pytest.mark.parametrize("seed", [73, 20261020])
def test_mtgp32_lognormal_fp64_across_direct_threshold(seed, record_property):
    n = 1048576
    bulk = curand.create_generator("mtgp32", seed=seed)
    split = curand.create_generator("mtgp32", seed=seed)
    output = torch.empty(n, dtype=torch.float64, device="cuda")
    first = torch.empty(65536, dtype=torch.float64, device="cuda")
    second = torch.empty(n - first.numel(), dtype=torch.float64, device="cuda")
    for mean, stddev in ((0.0, 1.0), (-0.25, 1.5)):
        curand.generate_lognormal_double(bulk, output, mean=mean, stddev=stddev)
        curand.generate_lognormal_double(split, first, mean=mean, stddev=stddev)
        curand.generate_lognormal_double(split, second, mean=mean, stddev=stddev)
        checks = {
            "output": bool(torch.equal(output, torch.cat((first, second)))),
            "state": bool(torch.equal(bulk.engine.state, split.engine.state)),
            "offset": bulk.engine.offset == split.engine.offset,
        }
        record_property(f"mean_{mean}", checks)
        assert all(checks.values()), checks


@pytest.mark.mtgp32_lognormal_f64
def test_mtgp32_exp_range_switch_and_continuation(monkeypatch, record_property):
    """Finite-range and general exp agree across all three public paths."""
    from flagrand.rng import _mtgp32_buffered, _mtgp32_direct_words, mtgp32
    from flagrand.rng._mtgp32_transforms import exp_finite_range

    calls = (
        ("lognormal_f64", 65536, 680.0, 1.5),  # Direct words, finite range.
        ("lognormal_f64", 65536, 710.0, 1.5),  # Same layout, general range.
        ("lognormal_f64", 131072, 0.0, 1.0),  # Separate transform, finite range.
        ("lognormal_f64", 131072, -740.0, 1.0),  # General subnormal outputs.
        ("lognormal_f64", 129, -680.0, 1.5),  # Partial raw cache.
        ("lognormal_f64", 129, 700.0, 0.0),  # Conservative host boundary.
        ("lognormal_f64", 129, math.nextafter(700.0, 0.0), 0.0),
        ("normal_f64", 3, 0.0, 1.0),
        ("raw32", 17, 0.0, 1.0),
    )
    for seed in (73, 20261020):
        reference = curand.create_generator("mtgp32", seed=seed)
        candidate = curand.create_generator("mtgp32", seed=seed)
        observations = []
        for api, n, mean, stddev in calls:
            expected, actual = (torch.empty(n + 1, dtype=DTYPES[api], device="cuda")[1:] for _ in range(2))

            def generate(generator, output):
                if api == "lognormal_f64":
                    curand.generate_lognormal_double(generator, output, mean=mean, stddev=stddev)
                else:
                    invoke(generator, output, api, None)

            with monkeypatch.context() as patch:
                for module in (_mtgp32_buffered, _mtgp32_direct_words, mtgp32):
                    patch.setattr(module, "exp_finite_range", lambda mean, stddev: False)
                generate(reference, expected)
            generate(candidate, actual)
            left, right = reference.engine, candidate.engine
            checks = {
                "output": bool(torch.equal(expected, actual)),
                "state": bool(torch.equal(left.state, right.state)),
                "offset": left.offset == right.offset,
                "frontier": left._ws_next_block_start == right._ws_next_block_start,
                "cache_start": left._chunk_cache_start == right._chunk_cache_start,
                "cache_key": left._chunk_cache_key == right._chunk_cache_key,
                "cache": (
                    left._chunk_cache is right._chunk_cache
                    if left._chunk_cache is None or right._chunk_cache is None
                    else bool(torch.equal(left._chunk_cache, right._chunk_cache))
                ),
            }
            observations.append(
                {
                    "api": api,
                    "n": n,
                    "mean": mean,
                    "stddev": stddev,
                    "finite_range": api == "lognormal_f64" and exp_finite_range(mean, stddev),
                    **checks,
                }
            )
            assert all(checks.values()), observations[-1]
        record_property(f"seed_{seed}_range_continuation", observations)


@pytest.mark.philox_poisson
@pytest.mark.parametrize("n", [1, 2, 3])
def test_philox_poisson_small_repeated_calls(n, record_property):
    """Small calls cross the counter=1 JIT specialization during continuation."""
    for lam in parameter_values("poisson"):
        bulk = curand.create_generator("philox", seed=73)
        split = curand.create_generator("philox", seed=73)
        expected = torch.empty(8 * n, dtype=torch.int32, device="cuda")
        outputs = [torch.empty(n, dtype=torch.int32, device="cuda") for _ in range(8)]
        curand.generate_poisson(bulk, expected, lam)
        for output in outputs:
            curand.generate_poisson(split, output, lam)
        checks = {
            "split_output": bool(torch.equal(expected, torch.cat(outputs))),
            "same_offset": bulk.engine.offset == split.engine.offset,
        }
        record_property(f"lambda_{lam}", checks)
        assert all(checks.values()), checks


@pytest.mark.parametrize("operator", CASES)
@pytest.mark.parametrize("n", ODD_TAILS)
def test_tail_and_seed_reset(operator, n, record_property):
    engine, api = split_operator(operator)
    lam = 10.0 if api == "poisson" else None
    generator = curand.create_generator(engine, seed=73)
    output = torch.empty(n, dtype=DTYPES[api], device="cuda")
    replay = torch.empty_like(output)
    invoke(generator, output, api, lam)
    curand.set_pseudo_random_generator_seed(generator, 73)
    invoke(generator, replay, api, lam)
    result = validate_output(output, replay, api, lam)
    record_property("validation", result)
    assert result["status"] == "pass", result


@pytest.mark.parametrize("operator", CASES)
@pytest.mark.parametrize("seed", [73, 20261020])
def test_distribution_moments_and_cdf(operator, seed, record_property):
    engine, api = split_operator(operator)
    observations = []
    for lam in parameter_values(api):
        output = torch.empty(131072, dtype=DTYPES[api], device="cuda")
        replay = torch.empty_like(output)
        invoke(curand.create_generator(engine, seed=seed), output, api, lam)
        invoke(curand.create_generator(engine, seed=seed), replay, api, lam)
        result = validate_output(output, replay, api, lam)
        observations.append({"lambda": lam, **result})
        record_property(f"lambda_{lam}", result)
        assert result["status"] == "pass", result
        if api == "uniform_f64":
            assert bool((output != output.float().double()).any()), "FP64 output is only expanded FP32"


@pytest.mark.parametrize("operator", CASES)
def test_continuation_and_current_stream(operator, record_property):
    engine, api = split_operator(operator)
    lam = 10.0 if api == "poisson" else None
    bulk_generator = curand.create_generator(engine, seed=73)
    split_generator = curand.create_generator(engine, seed=73)
    if engine == "xorwow":
        streams = bulk_generator.engine.nstreams
        first_n = streams // 2 if api == "poisson" else streams * (2 if "normal" in api else 1)
    elif engine == "mtgp32" and "normal" in api:
        first_n = 128  # MTGP32 consumes a complete pair at an odd final tail.
    else:
        first_n = 13
    second_n = 257
    output = torch.empty(first_n + second_n, dtype=DTYPES[api], device="cuda")
    invoke(bulk_generator, output, api, lam)
    first = torch.empty(first_n, dtype=DTYPES[api], device="cuda")
    second = torch.empty(second_n, dtype=DTYPES[api], device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        invoke(split_generator, first, api, lam)
        invoke(split_generator, second, api, lam)
    stream.synchronize()
    a, b = (torch.empty(257, dtype=torch.int32, device="cuda") for _ in range(2))
    curand.generate(bulk_generator, a)
    curand.generate(split_generator, b)
    checks = {
        "split_output": bool(torch.equal(output, torch.cat((first, second)))),
        "next_raw": bool(torch.equal(a, b)),
        "same_offset": bulk_generator.engine.offset == split_generator.engine.offset,
    }
    record_property("checks", checks)
    record_property("split_sizes", [first_n, second_n])
    assert all(checks.values()), checks


@pytest.mark.parametrize(
    "cache_api",
    [
        pytest.param("uniform_f32", marks=pytest.mark.mtgp32_uniform_f32),
        pytest.param("uniform_f64", marks=pytest.mark.mtgp32_uniform_f64),
    ],
)
@pytest.mark.parametrize("seed", [73, 20261020])
def test_mtgp32_uniform_cache_lifetime(monkeypatch, record_property, seed, cache_api):
    from flagrand.rng import _mtgp32_buffered as buffered

    reference = curand.create_generator("mtgp32", seed=seed)
    candidate = curand.create_generator("mtgp32", seed=seed)
    observations = []
    bits = 64 if cache_api == "uniform_f64" else 32
    cache_attribute = f"_uniform{bits}_cache"

    def compare(api, n, label):
        owners = [torch.full((n + 2,), 171, dtype=DTYPES[api], device="cuda") for _ in range(2)]
        outputs = [owner[1:-1].view(1, n) for owner in owners]
        with monkeypatch.context() as patch:
            patch.setattr(buffered, f"MAX_UNIFORM{bits}_CACHE_OUTPUTS", 0)
            invoke(reference, outputs[0], api, 10.0 if api == "poisson" else None)
        invoke(candidate, outputs[1], api, 10.0 if api == "poisson" else None)
        a, b = reference.engine, candidate.engine
        entry = getattr(b, cache_attribute, None)
        values = [] if entry is None else list(entry[1].values()) if bits == 64 else [entry[1]]
        checks = {
            "output": bool(torch.equal(*outputs)),
            "guards": all(bool((owner[[0, -1]] == 171).all()) for owner in owners),
            "state": bool(torch.equal(a.state, b.state)),
            "offset": a.offset == b.offset,
            "frontier": a._ws_next_block_start == b._ws_next_block_start,
            "cache_start": a._chunk_cache_start == b._chunk_cache_start,
            "raw_cache": (
                a._chunk_cache is b._chunk_cache
                if a._chunk_cache is None or b._chunk_cache is None
                else bool(torch.equal(a._chunk_cache, b._chunk_cache))
            ),
            "derived_ownership": entry is None or entry[0] is b._chunk_cache,
            "derived_bound": entry is None
            or sum(value.numel() * value.element_size() for value in values)
            <= (bits // 8) * (buffered.MAX_PREFETCH_WORDS + 255),
        }
        observations.append({"api": api, "n": n, "label": label, **checks})
        assert all(checks.values()), observations[-1]

    for _ in range(35):
        compare(cache_api, 128, "complete_window_and_refill")
    for api, n in (
        ("raw32", 17),
        (cache_api, 129),
        (cache_api, 65536),
        ("uniform_f32" if bits == 64 else "uniform_f64", 129),
        ("normal_f64", 130),
        (cache_api, 17),
        ("poisson", 129),
        (cache_api, 65537),
        (cache_api, 1048577),
        (cache_api, 17),
    ):
        compare(api, n, "mixed")
    original, alternate = torch.cuda.current_stream(), torch.cuda.Stream()
    alternate.wait_stream(original)
    with torch.cuda.stream(alternate):
        compare(cache_api, 129, "alternate_stream")
    original.wait_stream(alternate)
    compare("raw32", 17, "return_stream")
    for handle in (reference, candidate):
        curand.set_pseudo_random_generator_seed(handle, seed + 1)
        assert handle.engine._uniform32_cache is None
        assert handle.engine._uniform64_cache is None
    compare(cache_api, 128, "reset")
    compare("raw32", 17, "raw_after_reset")
    record_property("cache_continuation", observations)


@pytest.mark.mtgp32_poisson
@pytest.mark.parametrize("seed", [73, 20261020])
def test_mtgp32_poisson_cache_lifetime(record_property, seed):
    from flagrand.fused.poisson import _DEVICE_TABLES
    from flagrand.rng import _mtgp32_buffered as buffered

    reference = curand.create_generator("mtgp32", seed=seed)
    reference.engine._use_buffered = False
    candidate = curand.create_generator("mtgp32", seed=seed)
    observations = []

    def reset(value):
        for handle in (reference, candidate):
            curand.set_pseudo_random_generator_seed(handle, value)
            assert handle.engine._poisson_cache is None
            assert handle.engine._uniform32_cache is None
            assert handle.engine._uniform64_cache is None

    def compare(api, n, label, lam=None):
        previous = candidate.engine._poisson_cache
        owners = [torch.full((n + 2,), 171, dtype=DTYPES[api], device="cuda") for _ in range(2)]
        outputs = [owner[1:-1].view(1, n) for owner in owners]
        invoke(reference, outputs[0], api, lam)
        invoke(candidate, outputs[1], api, lam)
        a, b = reference.engine, candidate.engine
        entry = b._poisson_cache
        checks = {
            "output": bool(torch.equal(*outputs)),
            "guards": all(bool((owner[[0, -1]] == 171).all()) for owner in owners),
            "offset": a.offset == b.offset,
            "derived_ownership": entry is None or entry[0] is b._chunk_cache,
            "derived_bound": entry is None
            or sum(value.numel() * value.element_size() for value in entry[5].values())
            <= 4 * (buffered.MAX_PREFETCH_WORDS + 255),
        }
        if (
            api == "poisson"
            and n <= buffered.MAX_POISSON_CACHE_OUTPUTS
            and b._chunk_cache is not None
            and b._chunk_cache.numel() <= buffered.MAX_PREFETCH_WORDS + 255
        ):
            threshold, aliases, first = _DEVICE_TABLES[(float(lam), 1 << 32, str(outputs[1].device))]
            checks["derived_present"] = entry is not None
            assert entry is not None, {"label": label, "n": n, "lambda": lam}
            checks["last_table"] = (
                entry[1] is threshold and entry[2] is aliases and entry[3] == first and entry[4] == threshold.numel()
            )
            parity = (b.offset - 2 * n - b._chunk_cache_start) & 1
            checks["parity"] = parity in entry[5] and set(entry[5]) <= {0, 1}
            checks["stream"] = entry[6] == b._buffer_stream == torch.cuda.current_stream().cuda_stream
            if previous is not None and previous[0] is entry[0] and previous[1] is not threshold:
                checks["table_replaced"] = all(
                    value is not old for value in entry[5].values() for old in previous[5].values()
                )
        observations.append({"api": api, "n": n, "lambda": lam, "label": label, **checks})
        assert all(checks.values()), observations[-1]

    def exhaust(label):
        engine = candidate.engine
        remaining = engine._ws_next_block_start - engine.offset
        compare("raw32", remaining, label)
        checks = {
            "raw_cache_empty": engine._chunk_cache is None,
            "poisson_cache_empty": engine._poisson_cache is None,
            "uniform32_cache_empty": engine._uniform32_cache is None,
            "uniform64_cache_empty": engine._uniform64_cache is None,
            "frontier": reference.engine._ws_next_block_start == engine._ws_next_block_start,
            "state": bool(torch.equal(reference.engine.state, engine.state)),
        }
        observations.append({"label": label + "_state", **checks})
        assert all(checks.values()), observations[-1]

    for n, lam in ((128, 0.1), (16384, 1000.0), (65536, 1e6)):
        reset(seed)
        compare("poisson", n, "fresh_aligned", lam)
        assert candidate.engine._ws_next_block_start > candidate.engine.offset
        exhaust("fresh_aligned_exhaust")

    reset(seed)
    compare("poisson", 128, "even_parity", 0.1)
    compare("raw32", 17, "odd_raw_cursor")
    compare("poisson", 129, "odd_parity", 0.1)
    assert set(candidate.engine._poisson_cache[5]) == {0, 1}
    for lam in (1000.0, 1e6, 0.1):
        compare("poisson", 17, "table_switch", lam)
    compare("uniform_f32", 129, "mixed_uniform32")
    compare("uniform_f64", 129, "mixed_uniform64")
    compare("poisson", 17, "after_uniforms", 1000.0)
    refills = candidate.engine._buffer_stats["refills"]
    compare("poisson", 65536, "refill", 1000.0)
    assert candidate.engine._buffer_stats["refills"] > refills
    compare("poisson", 65537, "above_output_limit", 0.1)
    compare("raw32", buffered.MAX_PREFETCH_WORDS + 257, "above_raw_window_limit")
    compare("poisson", 17, "after_large_raw", 1e6)
    exhaust("mixed_exhaust")

    compare("poisson", 128, "stream_setup", 1000.0)
    original, alternate = torch.cuda.current_stream(), torch.cuda.Stream()
    alternate.wait_stream(original)
    with torch.cuda.stream(alternate):
        compare("poisson", 129, "alternate_stream", 1000.0)
    original.wait_stream(alternate)
    compare("poisson", 17, "return_stream", 1000.0)
    reset(seed + 1)
    compare("poisson", 128, "reset", 0.1)
    compare("raw32", 17, "raw_after_reset")
    exhaust("reset_exhaust")
    record_property("poisson_cache_continuation", observations)


@pytest.mark.parametrize(
    "engine", [pytest.param(engine, marks=getattr(pytest.mark, f"{engine}_raw32")) for engine in ENGINES]
)
@pytest.mark.parametrize("seed", [73, 20261020])
def test_raw_scalar_reference_and_state(engine, seed, record_property):
    from flagrand.rng._mrg32k3a_math import initial_states, reference_step
    from flagrand.rng._mtgp32_reference import generate_reference
    from flagrand.rng._xorwow.reference import generate_raw, seed_words

    generator = curand.create_generator(engine, seed=seed)
    n = 49409 if engine == "mtgp32" else 2 * generator.engine.nstreams + 17
    output = torch.empty(n, dtype=torch.int32, device="cuda")
    curand.generate(generator, output)
    if engine == "xorwow":
        expected, state = generate_raw(seed_words(generator.engine.nstreams, seed), n)
    elif engine == "mrg32k3a":
        streams = generator.engine.nstreams
        state = [[value & 0xFFFFFFFF for value in row] for row in initial_states(seed, streams)]
        expected = []
        for index in range(n):
            column = index % streams
            value, updated = reference_step([row[column] for row in state])
            expected.append(value)
            for row, item in zip(state, updated):
                row[column] = item
    else:
        references = [generate_reference(seed, 512, index) for index in range(192)]
        expected = [references[(i // 256) % 192][(i // 49152) * 256 + i % 256] for i in range(n)]
    got = [value & 0xFFFFFFFF for value in output.cpu().tolist()]
    mismatches = sum(a != b for a, b in zip(got, expected))
    record_property("raw_mismatches", mismatches)
    record_property("raw_words", n)
    assert mismatches == 0
    if engine != "mtgp32":
        got_state = [[value & 0xFFFFFFFF for value in row] for row in generator.engine.state.cpu().tolist()]
        state_mismatches = sum(a != b for left, right in zip(got_state, state) for a, b in zip(left, right))
        record_property("state_mismatches", state_mismatches)
        assert state_mismatches == 0


@pytest.mark.mtgp32_raw32
@pytest.mark.parametrize(
    "first_api, first_n",
    [
        pytest.param("uniform_f32", 256, marks=pytest.mark.mtgp32_uniform_f32),
        pytest.param("uniform_f64", 128, marks=pytest.mark.mtgp32_uniform_f64),
    ],
)
def test_mtgp32_mixed_api_raw_cache_and_state(record_property, first_api, first_n):
    """A prefetched tail survives aligned calls, distribution changes and refill."""
    reference = curand.create_generator("mtgp32", seed=73)
    reference.engine._use_buffered = False
    candidate = curand.create_generator("mtgp32", seed=73)
    checks = {}
    for index, (api, n) in enumerate(
        (
            (first_api, first_n),  # Fresh aligned uniform also uses the bounded cache.
            ("raw32", 257),
            ("raw32", 255),  # Logical offset is aligned while the new cache is not empty.
            ("raw32", 256),
            ("uniform_f64", 129),
            ("uniform_f32", 16384),
            ("uniform_f32", 17),
            ("normal_f32", 3),
            ("lognormal_f64", 129),
            ("poisson", 257),
            ("normal_f64", 16386),
            ("raw32", 1048593),  # Request exceeds the prefetch cap.
            ("raw32", 17),
        )
    ):
        a, b = (torch.empty(n, dtype=DTYPES[api], device="cuda") for _ in range(2))
        for generator, output in ((reference, a), (candidate, b)):
            if api == "raw32" and n == 256:
                curand.generate(generator, output, num_warps=4, cached_launch=False)
            else:
                invoke(generator, output, api, 10.0 if api == "poisson" else None)
        tolerance = 1e-6 if a.dtype == torch.float32 else 1e-12
        checks[f"{index}_{api}"] = bool(
            torch.allclose(a, b, rtol=tolerance, atol=tolerance) if a.is_floating_point() else torch.equal(a, b)
        )
        checks[f"{index}_offset"] = reference.engine.offset == candidate.engine.offset
    engine = candidate.engine
    remaining = engine._ws_next_block_start - engine.offset
    expected = torch.empty(remaining, dtype=torch.int32, device="cuda")
    curand.generate(reference, expected)
    cached = engine._chunk_cache[engine.offset - engine._chunk_cache_start :] if remaining else expected
    checks["remaining_raw_cache"] = bool(torch.equal(expected, cached))
    checks["full_state_at_same_physical_frontier"] = bool(torch.equal(reference.engine.state, engine.state))
    record_property("checks", checks)
    record_property("buffer_stats", engine._buffer_stats)
    record_property("remaining_raw_words", remaining)
    assert all(checks.values()), checks


@pytest.mark.parametrize(
    "api",
    [
        pytest.param("normal_f64", marks=pytest.mark.xorwow_normal_f64),
        pytest.param("lognormal_f64", marks=pytest.mark.xorwow_lognormal_f64),
    ],
)
@pytest.mark.parametrize("seed", [73, 20261020])
def test_xorwow_two_stage_state_and_workspace(api, seed, record_property):
    """Large complete rows preserve the fused sequence after small calls and stream switches."""
    generator = curand.create_generator("xorwow", seed=seed)
    reference = curand.create_generator("xorwow", seed=seed)
    function = curand.generate_lognormal_double if api == "lognormal_f64" else curand.generate_normal_double
    observations = []

    def compare(n, mean, stddev):
        actual = torch.empty(n + 1, dtype=torch.float64, device="cuda")[1:]
        expected = torch.empty_like(actual)
        function(generator, actual, mean=mean, stddev=stddev)
        # An explicit kernel option retains the public fused path with identical math.
        function(reference, expected, mean=mean, stddev=stddev, cached_launch=True)
        row = {
            "n": n,
            "mean": mean,
            "stddev": stddev,
            "output_equal": bool(torch.equal(actual, expected)),
            "state_words": generator.engine.state.numel(),
            "state_equal": bool(torch.equal(generator.engine.state, reference.engine.state)),
            "offsets": [generator.engine.offset, reference.engine.offset],
            "state_offsets": [generator.engine._state_offset, reference.engine._state_offset],
            "workspace_bytes": [
                entry[1].numel() * entry[1].element_size()
                for entry in generator.engine._distribution_workspaces.values()
            ],
        }
        observations.append(row)
        record_property(f"call_{len(observations)}", row)
        assert row["output_equal"] and row["state_equal"], row
        assert row["offsets"][0] == row["offsets"][1] and row["state_offsets"][0] == row["state_offsets"][1], row
        assert not reference.engine._distribution_workspaces

    for index, n in enumerate((257, 1048576, 129, 1114112, 1048577, 16777216, 1048576)):
        compare(n, -0.25 if index % 2 else 1.25, 1.5 if index % 2 else 0.625)
    assert [entry[1].numel() for entry in generator.engine._distribution_workspaces.values()] == [2 * 16777216]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compare(1048576, -0.25, 1.5)
    torch.cuda.current_stream().wait_stream(stream)
    assert sorted(entry[1].numel() for entry in generator.engine._distribution_workspaces.values()) == [
        2 * 1048576,
        2 * 16777216,
    ]
    record_property("workspace_streams_before_reset", len(generator.engine._distribution_workspaces))
    for handle in (generator, reference):
        curand.set_pseudo_random_generator_seed(handle, seed + 1)
    assert not generator.engine._distribution_workspaces
    record_property("workspace_released_by_seed_reset", True)
    compare(1048576, 0.0, 1.0)
    raw = [torch.empty(257, dtype=torch.int32, device="cuda") for _ in range(2)]
    for handle, output in zip((generator, reference), raw):
        curand.generate(handle, output)
    checks = {
        "next_raw": bool(torch.equal(*raw)),
        "next_state": bool(torch.equal(generator.engine.state, reference.engine.state)),
        "next_offset": generator.engine.offset == reference.engine.offset,
    }
    record_property("after_reset_next_raw", checks)
    assert all(checks.values()), checks


@pytest.mark.mtgp32_raw32
@pytest.mark.mtgp32_normal_f64
@pytest.mark.mtgp32_lognormal_f64
def test_mtgp32_tle_region_and_partial_continuation(monkeypatch, record_property):
    """Switch recurrence storage across the size boundary and retain raw cache."""
    from flagrand.rng import _mtgp32_kernel

    select_launcher = _mtgp32_kernel._metax_tle_launcher
    if select_launcher("metax") is None:
        pytest.skip("The installed MetaX Triton has no TLE shared-memory interface.")
    reference = curand.create_generator("mtgp32", seed=20261020)
    candidate = curand.create_generator("mtgp32", seed=20261020)
    checks = {}
    for index, (api, n) in enumerate(
        (
            ("raw32", 1048320),
            ("raw32", 1048832),
            ("normal_f64", 1048576),
            ("raw32", 17),
            ("lognormal_f64", 129),
            ("raw32", 1048577),
            ("raw32", 17),
        )
    ):
        expected, actual = (torch.empty(n, dtype=DTYPES[api], device="cuda") for _ in range(2))
        monkeypatch.setattr(_mtgp32_kernel, "_metax_tle_launcher", lambda vendor: None)
        invoke(reference, expected, api, None)
        monkeypatch.setattr(_mtgp32_kernel, "_metax_tle_launcher", select_launcher)
        invoke(candidate, actual, api, None)
        left, right = reference.engine, candidate.engine
        checks[f"{index}_output"] = bool(torch.equal(expected, actual))
        checks[f"{index}_state"] = bool(torch.equal(left.state, right.state))
        checks[f"{index}_offset"] = left.offset == right.offset
        checks[f"{index}_frontier"] = left._ws_next_block_start == right._ws_next_block_start
        checks[f"{index}_cache_start"] = left._chunk_cache_start == right._chunk_cache_start
        checks[f"{index}_cache"] = (
            left._chunk_cache is right._chunk_cache
            if left._chunk_cache is None or right._chunk_cache is None
            else bool(torch.equal(left._chunk_cache, right._chunk_cache))
        )
    record_property("checks", checks)
    assert all(checks.values()), checks


@pytest.mark.mtgp32_raw32
@pytest.mark.mtgp32_uniform_f32
def test_mtgp32_refill_output_fusion_continuation(monkeypatch, record_property):
    """Writing public output during refill preserves cache and recurrence state."""
    from flagrand.rng import _mtgp32_buffered

    raw_window = _mtgp32_buffered.raw_window

    def separate_output(engine, words, device, output=None, output_mode=-1):
        return raw_window(engine, words, device, output=output, output_mode=-1)

    calls = (
        ("raw32", 257),
        ("uniform_f32", 255),
        ("raw32", 256),
        ("uniform_f64", 129),
        ("normal_f32", 3),
        ("lognormal_f64", 129),
        ("poisson", 257),
        ("uniform_f32", 65537),
        ("raw32", 1048577),
        ("uniform_f32", 1048577),
        ("raw32", 17),
    )
    for seed in (73, 20261020):
        reference = curand.create_generator("mtgp32", seed=seed)
        candidate = curand.create_generator("mtgp32", seed=seed)
        checks = {}
        stream = torch.cuda.Stream()
        for index, (api, n) in enumerate(calls):
            expected, actual = (torch.empty(n + 1, dtype=DTYPES[api], device="cuda")[1:] for _ in range(2))

            def compare():
                with monkeypatch.context() as patch:
                    patch.setattr(_mtgp32_buffered, "raw_window", separate_output)
                    invoke(reference, expected, api, 10.0 if api == "poisson" else None)
                invoke(candidate, actual, api, 10.0 if api == "poisson" else None)
                left, right = reference.engine, candidate.engine
                checks[f"{index}_{api}_output"] = bool(torch.equal(expected, actual))
                checks[f"{index}_state"] = bool(torch.equal(left.state, right.state))
                checks[f"{index}_offset"] = left.offset == right.offset
                checks[f"{index}_frontier"] = left._ws_next_block_start == right._ws_next_block_start
                checks[f"{index}_cache_start"] = left._chunk_cache_start == right._chunk_cache_start
                checks[f"{index}_cache"] = (
                    left._chunk_cache is right._chunk_cache
                    if left._chunk_cache is None or right._chunk_cache is None
                    else bool(torch.equal(left._chunk_cache, right._chunk_cache))
                )

            if index == 7:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    compare()
                torch.cuda.current_stream().wait_stream(stream)
            else:
                compare()
        record_property(f"seed_{seed}_checks", checks)
        record_property(f"seed_{seed}_buffer_stats", candidate.engine._buffer_stats)
        assert all(checks.values()), checks


@pytest.mark.mtgp32_uniform_f64
@pytest.mark.mtgp32_poisson
@pytest.mark.mtgp32_normal_f32
@pytest.mark.mtgp32_lognormal_f32
@pytest.mark.parametrize("seed", [73, 20261020])
def test_mtgp32_shared_pairs_boundary_continuation(monkeypatch, record_property, seed):
    """Shared pairs preserve the sequence, region boundaries and partial cache."""
    from flagrand.rng import _mtgp32_direct_pairs

    select_pairs = _mtgp32_direct_pairs._metax_shared_pairs
    select_normal = _mtgp32_direct_pairs._metax_shared_normal
    shared_normal = select_normal()
    if select_pairs() is None or shared_normal is None:
        pytest.skip("The installed MetaX Triton has no TLE shared-memory interface.")
    reference = curand.create_generator("mtgp32", seed=seed)
    candidate = curand.create_generator("mtgp32", seed=seed)
    checks, normal_calls = {}, []

    def tracked_normal(engine, out, distribution, **kwargs):
        normal_calls.append((distribution, out.numel()))
        return shared_normal(engine, out, distribution, **kwargs)

    monkeypatch.setattr(_mtgp32_direct_pairs, "_metax_shared_normal", lambda: tracked_normal)

    def compare(api, n, index):
        expected, actual = (torch.empty(n + 1, dtype=DTYPES[api], device="cuda")[1:] for _ in range(2))

        def call(generator, out):
            if api in ("normal_f32", "lognormal_f32"):
                function = curand.generate_normal if api == "normal_f32" else curand.generate_lognormal
                function(generator, out, mean=-0.25 if index % 2 else 1.25, stddev=0.625)
            else:
                invoke(generator, out, api, 10.0 if api == "poisson" else None)

        before = len(normal_calls)
        with monkeypatch.context() as patch:
            patch.setattr(_mtgp32_direct_pairs, "_metax_shared_pairs", lambda: None)
            patch.setattr(_mtgp32_direct_pairs, "_metax_shared_normal", lambda: None)
            call(reference, expected)
        call(candidate, actual)
        left, right = reference.engine, candidate.engine
        checks[f"{index}_output"] = bool(torch.equal(expected, actual))
        checks[f"{index}_state"] = bool(torch.equal(left.state, right.state))
        checks[f"{index}_offset"] = left.offset == right.offset
        checks[f"{index}_frontier"] = left._ws_next_block_start == right._ws_next_block_start
        checks[f"{index}_cache_start"] = left._chunk_cache_start == right._chunk_cache_start
        checks[f"{index}_cache_key"] = left._chunk_cache_key == right._chunk_cache_key
        checks[f"{index}_cache"] = (
            left._chunk_cache is right._chunk_cache
            if left._chunk_cache is None or right._chunk_cache is None
            else bool(torch.equal(left._chunk_cache, right._chunk_cache))
        )
        expected_shared = index in (3, 4, 9, 14)
        checks[f"{index}_shared_normal_dispatch"] = (len(normal_calls) > before) == expected_shared

    calls = (
        ("uniform_f64", 524160),
        ("normal_f32", 262144),
        ("normal_f32", 1048320),
        ("normal_f32", 1048575),
        ("lognormal_f32", 1048576),
        ("normal_f32", 1048832),
        ("lognormal_f32", 16777216),
        ("poisson", 524416),
        ("uniform_f64", 1048576),
        ("normal_f32", 1048576),
        ("raw32", 17),
        ("lognormal_f32", 1048577),
        ("poisson", 1048577),
        ("raw32", 17),
    )
    stream = torch.cuda.Stream()
    for index, (api, n) in enumerate(calls):
        if index == 9:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                compare(api, n, index)
            torch.cuda.current_stream().wait_stream(stream)
        else:
            compare(api, n, index)
    for generator in (reference, candidate):
        curand.set_pseudo_random_generator_seed(generator, seed + 1)
    compare("normal_f32", 1048576, 14)
    for label, options in (("gather", {"direct": False}), ("native_trig", {"native_trig": True})):
        expected, actual = (torch.empty(1048576, dtype=torch.float32, device="cuda") for _ in range(2))
        before = len(normal_calls)
        with monkeypatch.context() as patch:
            patch.setattr(_mtgp32_direct_pairs, "_metax_shared_normal", lambda: None)
            _mtgp32_direct_pairs.generate_direct_pairs(reference.engine, expected, "normal", **options)
        _mtgp32_direct_pairs.generate_direct_pairs(candidate.engine, actual, "normal", **options)
        checks[f"{label}_output"] = bool(torch.equal(expected, actual))
        checks[f"{label}_state"] = bool(torch.equal(reference.engine.state, candidate.engine.state))
        checks[f"{label}_offset"] = reference.engine.offset == candidate.engine.offset
        checks[f"{label}_frontier"] = reference.engine._ws_next_block_start == candidate.engine._ws_next_block_start
        checks[f"{label}_original_dispatch"] = len(normal_calls) == before
    record_property("checks", checks)
    record_property("shared_normal_calls", normal_calls)
    assert all(checks.values()), checks


@pytest.mark.mtgp32_uniform_f64
@pytest.mark.mtgp32_normal_f64
@pytest.mark.mtgp32_lognormal_f64
@pytest.mark.mtgp32_poisson
@pytest.mark.parametrize("seed", [73, 20261020])
def test_mtgp32_shared_refill_continuation(monkeypatch, record_property, seed):
    """Large refills preserve prefix words, complete state and mixed continuation."""
    from flagrand.rng import _mtgp32_buffered as buffered

    shared = buffered._metax_shared_refill()
    if shared is None:
        pytest.skip("The installed MetaX Triton has no TLE shared-memory interface.")
    reference = curand.create_generator("mtgp32", seed=seed)
    candidate = curand.create_generator("mtgp32", seed=seed)
    checks, selected = {}, []

    def select_shared():
        selected.append(True)
        return shared

    monkeypatch.setattr(buffered, "_metax_shared_refill", select_shared)

    def compare(api, n, index, expected_shared):
        expected, actual = (torch.empty(n + 1, dtype=DTYPES[api], device="cuda")[1:] for _ in range(2))
        before = len(selected)
        with monkeypatch.context() as patch:
            patch.setattr(buffered, "_metax_shared_refill", lambda: None)
            invoke(reference, expected, api, 10.0 if api == "poisson" else None)
        invoke(candidate, actual, api, 10.0 if api == "poisson" else None)
        left, right = reference.engine, candidate.engine
        checks[f"{index}_output"] = bool(torch.equal(expected, actual))
        checks[f"{index}_state"] = bool(torch.equal(left.state, right.state))
        checks[f"{index}_offset"] = left.offset == right.offset
        checks[f"{index}_frontier"] = left._ws_next_block_start == right._ws_next_block_start
        checks[f"{index}_cache_start"] = left._chunk_cache_start == right._chunk_cache_start
        checks[f"{index}_cache"] = bool(torch.equal(left._chunk_cache, right._chunk_cache))
        checks[f"{index}_shared_dispatch"] = (len(selected) > before) == expected_shared

    calls = (
        ("raw32", 17, False),
        ("uniform_f64", 1049089, True),
        ("lognormal_f64", 1048575, True),
        ("poisson", 1048577, True),
        ("normal_f64", 65538, False),
        ("raw32", 17, False),
    )
    stream = torch.cuda.Stream()
    for index, (api, n, expected_shared) in enumerate(calls):
        if index == 2:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                compare(api, n, index, expected_shared)
            torch.cuda.current_stream().wait_stream(stream)
        else:
            compare(api, n, index, expected_shared)
    for generator in (reference, candidate):
        curand.set_pseudo_random_generator_seed(generator, seed + 1)
    compare("poisson", 1048577, 6, True)
    compare("raw32", 17, 7, False)
    record_property("checks", checks)
    record_property("shared_refill_calls", len(selected))
    assert all(checks.values()), checks
