from __future__ import annotations

import time

import torch

from flagrand import c_operators, curand


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.cpu().tolist() == expected.cpu().tolist()


def _compare_raw() -> None:
    expected_seed_zero = [
        1713891541,
        -1135104948,
        -513161843,
        -1694442536,
        -2075847199,
        253345875,
        -259166550,
        -2080868880,
        107075191,
        639715526,
        -1232967847,
        2000445424,
        -257357124,
        -1092682108,
        291018424,
        -1200667194,
    ]
    out = torch.empty(16, device="cuda", dtype=torch.int32)
    generator = c_operators.create_generator(seed=0)
    c_operators.generate(generator, out)
    assert out.cpu().tolist() == expected_seed_zero
    assert generator.offset == 16

    actual = torch.empty(17, device="cuda", dtype=torch.int32)
    expected = torch.empty_like(actual)
    c_operators.generate(c_operators.create_generator(seed=23, offset=5), actual)
    curand.generate(curand.create_generator(seed=23, offset=5), expected)
    _assert_equal(actual, expected)


def _compare_distributions() -> None:
    cases = (
        (
            c_operators.generate_uniform,
            curand.generate_uniform,
            torch.float32,
            17,
            3,
            {},
            17,
        ),
        (
            c_operators.generate_uniform_double,
            curand.generate_uniform_double,
            torch.float64,
            9,
            1,
            {},
            18,
        ),
        (
            c_operators.generate_normal,
            curand.generate_normal,
            torch.float32,
            18,
            2,
            {"mean": 0.25, "stddev": 1.5},
            18,
        ),
        (
            c_operators.generate_lognormal,
            curand.generate_lognormal,
            torch.float32,
            18,
            1,
            {"mean": 0.2, "stddev": 0.5},
            18,
        ),
    )
    for cpp_op, python_op, dtype, n, offset, kwargs, consumed in cases:
        actual = torch.empty(n, device="cuda", dtype=dtype)
        expected = torch.empty_like(actual)
        cpp_generator = c_operators.create_generator(seed=73, offset=offset)
        cpp_op(cpp_generator, actual, **kwargs)
        python_op(curand.create_generator(seed=73, offset=offset), expected, **kwargs)
        _assert_equal(actual, expected)
        assert cpp_generator.offset == offset + consumed


def _check_statistics() -> None:
    n = 262144
    uniform32 = torch.empty(n, device="cuda", dtype=torch.float32)
    uniform64 = torch.empty(n, device="cuda", dtype=torch.float64)
    normal = torch.empty(n, device="cuda", dtype=torch.float32)
    lognormal = torch.empty(n, device="cuda", dtype=torch.float32)

    c_operators.generate_uniform(c_operators.create_generator(seed=41), uniform32)
    c_operators.generate_uniform_double(c_operators.create_generator(seed=43), uniform64)
    c_operators.generate_normal(c_operators.create_generator(seed=47), normal, mean=0.25, stddev=1.5)
    c_operators.generate_lognormal(c_operators.create_generator(seed=53), lognormal, mean=0.2, stddev=0.5)
    torch.cuda.synchronize()

    for device_values in (uniform32, uniform64):
        values = device_values.cpu()
        assert bool(torch.isfinite(values).all())
        assert 0.0 < float(values.min()) <= float(values.max()) <= 1.0
        assert abs(float(values.mean()) - 0.5) < 0.004
        assert abs(float(values.var()) - (1.0 / 12.0)) < 0.002

    normal_cpu = normal.cpu()
    assert bool(torch.isfinite(normal_cpu).all())
    assert abs(float(normal_cpu.mean()) - 0.25) < 0.015
    assert abs(float(normal_cpu.std()) - 1.5) < 0.015

    log_values = lognormal.cpu().log()
    assert bool(torch.isfinite(log_values).all())
    assert abs(float(log_values.mean()) - 0.2) < 0.01
    assert abs(float(log_values.std()) - 0.5) < 0.01


def _wall_time_ms(function, repeats: int = 100) -> float:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repeats


def _benchmark_wrapper() -> None:
    print("| N | Python Host API ms | C++ wrapper ms | Python/C++ |")
    print("|---:|---:|---:|---:|")
    for n in (4096, 1048576):
        python_out = torch.empty(n, device="cuda", dtype=torch.float32)
        cpp_out = torch.empty_like(python_out)

        def python_call() -> None:
            curand.generate_uniform(curand.create_generator(seed=97), python_out)

        def cpp_call() -> None:
            c_operators.generate_uniform(c_operators.create_generator(seed=97), cpp_out)

        python_ms = _wall_time_ms(python_call)
        cpp_ms = _wall_time_ms(cpp_call)
        print(f"| {n} | {python_ms:.6f} | {cpp_ms:.6f} | {python_ms / cpp_ms:.3f}x |")


def main() -> None:
    print("c_operators:", c_operators.__file__)
    print("device:", torch.cuda.get_device_name(0))
    _compare_raw()
    _compare_distributions()
    _check_statistics()
    _benchmark_wrapper()
    print("C++ WRAPPER ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
