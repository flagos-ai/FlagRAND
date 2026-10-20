"""XPU software-f64 binding with optional runtime loops and split products."""

from functools import lru_cache
from types import FunctionType, SimpleNamespace

import triton
from triton.runtime.jit import JITFunction


@lru_cache(maxsize=8)
def bind_xpu_soft64(runtime_loops=False, split_product=False, inline_product=False):
    """Bind shared kernels and selected math in an XPU-only namespace.

    Static math and the original FMA-based two_prod are the defaults.
    split_product substitutes only the full normalized product residual;
    inline_product takes precedence and uses an explicit rounded multiply.
    runtime_loops selects the existing Horner and correction loop bodies.
    """
    from . import soft64_kernels, soft64_math

    modules = (soft64_math, soft64_kernels)
    if runtime_loops:
        from . import xpu_soft64_loop_math

        modules += (xpu_soft64_loop_math,)
    namespace = {}
    for module in modules:
        namespace.update(vars(module))
    # XPU already compiled the original annotated tuple constants into TTIR.
    # Preserve that representation rather than applying Ascend's constexpr
    # value wrappers. The copied annotations accompany the merged globals.
    namespace["__annotations__"] = dict(soft64_math.__annotations__)
    if inline_product:
        from .xpu_product import two_prod_inline

        namespace["two_prod"] = two_prod_inline
    elif split_product:
        from .ascend_soft64_math import two_prod_split

        namespace["two_prod"] = two_prod_split

    for module in modules:
        for name, original in vars(module).items():
            if (
                isinstance(original, JITFunction)
                and original.fn.__module__ == module.__name__
                and not ((inline_product or split_product) and name == "two_prod")
            ):
                function = original.fn
                bound = FunctionType(
                    function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
                )
                bound.__annotations__ = dict(function.__annotations__)
                bound.__kwdefaults__ = function.__kwdefaults__
                bound.__qualname__ = function.__qualname__
                bound.__module__ = function.__module__
                namespace[name] = triton.jit(
                    bound, noinline=original.noinline, do_not_specialize=original.do_not_specialize
                )
    return SimpleNamespace(**namespace)


def bind_xpu_soft64_loops():
    return bind_xpu_soft64(runtime_loops=True)
