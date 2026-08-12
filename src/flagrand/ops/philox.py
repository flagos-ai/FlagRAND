"""Canonical location of the seven public Philox distribution operators."""

from flagrand.curand import (
    generate,
    generate_lognormal,
    generate_lognormal_double,
    generate_normal,
    generate_normal_double,
    generate_uniform,
    generate_uniform_double,
)

raw32 = generate
uniform_f32 = generate_uniform
uniform_f64 = generate_uniform_double
normal_f32 = generate_normal
normal_f64 = generate_normal_double
lognormal_f32 = generate_lognormal
lognormal_f64 = generate_lognormal_double

__all__ = [
    "raw32",
    "uniform_f32",
    "uniform_f64",
    "normal_f32",
    "normal_f64",
    "lognormal_f32",
    "lognormal_f64",
]
