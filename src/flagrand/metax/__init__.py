"""MetaX MACA-specific Philox implementations."""

from flagrand.metax.philox import (
    generate_metax_normal_f32,
    generate_metax_normal_f64,
    generate_metax_raw_u32,
    generate_metax_uniform_f32,
    generate_metax_uniform_f64,
)

__all__ = [
    "generate_metax_normal_f32",
    "generate_metax_normal_f64",
    "generate_metax_raw_u32",
    "generate_metax_uniform_f32",
    "generate_metax_uniform_f64",
]
