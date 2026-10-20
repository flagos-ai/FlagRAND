"""Bind factory captures for JITs that inspect globals instead of closure cells."""

from types import FunctionType

import triton
import triton.language as tl


def bind_kernel(function, **jit_options):
    namespace = dict(function.__globals__)
    if function.__closure__:
        namespace.update(
            (name, cell.cell_contents) for name, cell in zip(function.__code__.co_freevars, function.__closure__)
        )
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    bound.__annotations__ = function.__annotations__
    bound.__qualname__ = function.__qualname__
    bound.__module__ = function.__module__
    return triton.jit(bound, **jit_options)


def bind_math_kernel(function, device_math, parameter_bits_constexpr=False):
    if parameter_bits_constexpr:
        # XPU's installed JIT reads inspect.signature(fn), hence annotations
        # must be set before bind_kernel creates the JITFunction. Raw/uniform
        # do not use these values; constexpr keeps them out of the runtime ABI.
        function.__annotations__.update(MU_BITS=tl.constexpr, SIGMA_BITS=tl.constexpr)
        return bind_kernel(function, do_not_specialize=["N"])
    return bind_kernel(function, do_not_specialize=["N", "MU_BITS", "SIGMA_BITS"])
