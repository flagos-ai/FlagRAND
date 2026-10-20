"""Project-local XPU JIT binding that omits the StoreControl pass.

Use bind_xpu_without_store_control only for this project's software-f64
kernels. The installed compiler and all its other passes are reused through
private function globals; installed modules and pass registration stay intact.
"""

from functools import lru_cache
from importlib import import_module
from types import FunctionType, SimpleNamespace

COMPILER_REVISION = "xorwow-soft64-skip-store-control-v1"


def _clone_function(function, replacements):
    namespace = {**function.__globals__, **replacements}
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    bound.__annotations__ = dict(function.__annotations__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__qualname__ = function.__qualname__
    bound.__module__ = function.__module__
    return bound


def _omit_store_control(pass_manager):
    """Leave this one pass out of the private compilation pipeline."""


def _make_backend(target):
    installed = import_module("triton.backends.xpu.compiler")
    base = installed.XPUBackend

    # A fresh class keeps the installed static methods' buffer_len state
    # private to this backend instance, including buffer-size recompilation.
    class SoftwareF64XPUBackend(base):
        def hash(self):
            return super().hash() + ":" + COMPILER_REVISION

    xpu = installed.xpu
    ttxpuir = SimpleNamespace(
        **{
            **vars(xpu.passes.ttxpuir),
            "add_tritonxpu_store_control_pass": _omit_store_control,
        }
    )
    passes = SimpleNamespace(**{**vars(xpu.passes), "ttxpuir": ttxpuir})
    private_xpu = SimpleNamespace(**{**vars(xpu), "passes": passes})
    replacements = {"xpu": private_xpu, "XPUBackend": SoftwareF64XPUBackend}
    SoftwareF64XPUBackend.make_ttxir = staticmethod(_clone_function(base.make_ttxir, replacements))
    # make_llir reads the buffer_len that make_ttxir wrote through XPUBackend.
    SoftwareF64XPUBackend.make_llir = staticmethod(_clone_function(base.make_llir, replacements))
    return SoftwareF64XPUBackend(target)


@lru_cache(maxsize=1)
def _compile_function():
    compiler = import_module("triton.compiler.compiler")
    return _clone_function(compiler.compile, {"make_backend": _make_backend})


@lru_cache(maxsize=None)
def bind_xpu_without_store_control(kernel):
    """Return a separate JIT object with the scoped XPU compiler.

    Example: kernel = bind_xpu_without_store_control(
        generate_standard_soft64_kernel)
    Then launch kernel[grid](...) with its existing arguments and options.

    Separate JIT objects isolate both installed in-memory caches. The backend
    revision also enters the installed compiler's on-disk cache hash.
    """
    triton = import_module("triton")
    bound = triton.jit(
        kernel.fn,
        version=kernel.version,
        do_not_specialize=kernel.do_not_specialize,
        debug=kernel.debug,
        noinline=kernel.noinline,
        repr=kernel.repr,
        launch_metadata=kernel.launch_metadata,
    )
    # This installed JIT assigns these hooks in create_binder; initialize it
    # once before replacing only the new object's compiler entry points.
    bound.create_binder()
    bound.make_backend = _make_backend
    bound.compile = _compile_function()
    return bound
