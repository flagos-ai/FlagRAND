"""Host API for the independent Triton kernels. No native RNG fallback."""

import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

from ...runtime.compiled_launcher import CachedKernelLauncher, PreparedKernelLaunch
from ...runtime.specialization import launch_specialization_key
from .kernels import build_generate_kernel
from .reference import seed_words, signed_rows


@lru_cache(maxsize=8)
def _seed_tensor(nstreams, seed):
    return torch.tensor(signed_rows(seed_words(nstreams, seed)), dtype=torch.int32)


def create_state(nstreams=4096, seed=0, device="cuda"):
    """Seed on CPU and transfer int32 SoA state with shape (6, nstreams).

    Initialization is separate from generation. State is mutated by generate;
    clone it to replay a sequence or compare launch configurations.
    """
    return _seed_tensor(nstreams, seed).to(device=device, copy=True)


_ALIASES = {
    "maca": "metax",
    "hip": "hygon",
    "hcu": "hygon",
    "bw1000": "hygon",
    "cann": "ascend",
    "npu": "ascend",
    "910c": "ascend",
    "xpu": "kunlunxin",
    "p800": "kunlunxin",
    "kunlun": "kunlunxin",
    "ppu": "thead",
    "zw810e": "thead",
    "musa": "mthreads",
    "s5000": "mthreads",
    "corex": "iluvatar",
    "bi-v150": "iluvatar",
    "biv150": "iluvatar",
}


def resolve_backend(backend="auto", device=None):
    if backend != "auto":
        return _ALIASES.get(backend.lower(), backend.lower())
    target = triton.runtime.driver.active.get_current_target()
    name = str(target.backend).lower()
    if name != "cuda":
        return _ALIASES.get(name, name)
    # Several installed vendor runtimes expose a CUDA-compatible API/target.
    card = torch.cuda.get_device_name(device).lower()
    if "iluvatar" in card or "bi-v" in card:
        return "iluvatar"
    if "ppu" in card or "zw810" in card or "thead" in card:
        return "thead"
    if "metax" in card or "c550" in card:
        return "metax"
    if "p800" in card or "kunlun" in card:
        return "kunlunxin"
    return "cuda"


def allocate_output(n, dtype, device, backend="auto"):
    """Allocate output storage, retaining binary64 bytes on the XPU runtime."""
    if dtype == torch.float64 and resolve_backend(backend, device) == "kunlunxin":
        # Direct float64 allocation is rewritten to float32 by this runtime.
        # An int32 allocation followed by a dtype view keeps all eight bytes.
        return torch.empty(2 * n, dtype=torch.int32, device=device).view(torch.float64)
    return torch.empty(n, dtype=dtype, device=device)


@lru_cache(None)
def _math_module(backend):
    # The old PPU 3.5 runtime uses a CUDA-compatible math module; its 3.6
    # successor exposes extra.ppu. Choose the installed namespace explicitly.
    target = str(triton.runtime.driver.active.get_current_target().backend)
    namespace = {
        "hygon": "hip",
        "metax": "cuda",
        "iluvatar": "corex",
        "thead": "ppu" if target == "ppu" else "cuda",
        "mthreads": "musa",
        "ascend": "cann",
        "kunlunxin": "xpu",
        "cuda": "cuda",
    }[backend]
    if backend == "iluvatar" and str(triton.__version__).startswith("3.1"):
        namespace = "cuda"
    if backend == "kunlunxin":
        # XPU's libdevice log/exp declarations are Unsupported; its language
        # math IR is a separate path. Device measurements determine support.
        math = tl
    else:
        math = import_module("triton.language.extra." + namespace + ".libdevice")
    return math


@lru_cache(None)
def _compiled_entry(backend, transformed, f64, packed_uniform):
    math = _math_module(backend) if transformed else tl
    parameter_bits_constexpr = backend == "kunlunxin" and not transformed
    if packed_uniform or (backend == "ascend" and not (transformed and f64)):
        from .ascend_kernels import build_generate_kernel as build_ascend_kernel

        return (
            build_ascend_kernel(
                math,
                parameter_bits_constexpr=parameter_bits_constexpr,
            ),
            math.__name__,
        )
    return (
        build_generate_kernel(
            math,
            parameter_bits_constexpr=parameter_bits_constexpr,
        ),
        math.__name__,
    )


@lru_cache(None)
def _corex_uniform_entry():
    from .corex_packed64 import build_uniform64_kernel

    return build_uniform64_kernel(_math_module("iluvatar"))


@lru_cache(None)
def _standard_launcher(kernel, option_names):
    return CachedKernelLauncher(
        kernel,
        constexpr_names=(
            "NSTREAMS",
            "BLOCK",
            "ROUNDS",
            "DISTRIBUTION",
            "F64",
            "MATH_TAG",
            "REDUCED_TRIG",
            "ALIGNED",
            "INDEX32",
            "FAST_F32",
            "HCU_SFU",
            "INTRINSIC_TRIG",
            "RANGE_LOG64",
            "RANGE_EXP64",
            "FAST_EXP_F32",
            "TL_SQRT_F64",
            "RSQRT_REFINED_F64",
            "PACKED_INTERLEAVE",
            "STANDARD_PARAMS",
            "MUSA_PI32",
            "XPU_SCALAR_BASE",
        ),
        option_names=option_names,
    )


def _float_bits(value, f64):
    return struct.unpack("<Q" if f64 else "<I", struct.pack("<d" if f64 else "<f", value))[0]


@lru_cache(maxsize=16)
def _soft_parameters(mu_bits, sigma_bits, device):
    words = []
    for bits in (mu_bits, sigma_bits):
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        remainder, exponent = math.frexp(value)
        for _ in range(3):
            part = struct.unpack("<f", struct.pack("<f", remainder))[0]
            words.append(struct.unpack("<i", struct.pack("<f", part))[0])
            remainder -= part
        words.append(exponent)
    return torch.tensor(words, dtype=torch.int32, device=device)


def _launch_geometry(n, nstreams, pair, block, rounds, aligned, index32, packed):
    use_aligned = aligned and nstreams % block == 0 and n % (nstreams * pair * rounds) == 0
    index_limit = 2**30 if packed else 2**31
    use_index32 = index32 and max(n + nstreams * pair * rounds, 6 * nstreams) < index_limit
    grid = (triton.cdiv(min(nstreams, n), block),)
    return use_aligned, use_index32, grid


def _launch_options(backend, num_warps, num_stages, maxnreg, compiler_options):
    launch_options = dict(num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    if backend == "kunlunxin":
        launch_options["is_use_mask_zero"] = True
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    launch_options.update(compiler_options or {})
    return launch_options


def _standard_launch_spec(
    n,
    nstreams,
    kind,
    f64,
    mu,
    sigma,
    backend,
    block,
    rounds,
    math_mode,
    use_aligned,
    use_index32,
    packed_uniform,
    specialize_standard,
    xpu_scalar_base,
    launch_options,
):
    """Build the standard kernel's scalar arguments and constexpr values."""
    launch_options = dict(launch_options)
    kernel, math_tag = _compiled_entry(backend, kind >= 2, f64, packed_uniform)
    if backend == "kunlunxin":
        launch_options["XPU_SCALAR_BASE"] = xpu_scalar_base
    musa_pi32 = backend == "mthreads" and not f64 and kind >= 2 and math_mode == "musa_pi32"
    if musa_pi32:
        launch_options["MUSA_PI32"] = True
    if specialize_standard and backend == "mthreads" and kind >= 2 and mu == 0.0 and sigma == 1.0:
        launch_options["STANDARD_PARAMS"] = True
    scalars = (n, _float_bits(mu, f64), _float_bits(sigma, f64))
    constants = (
        nstreams,
        block,
        rounds,
        kind,
        f64,
        math_tag,
        math_mode
        in (
            "reduced",
            "reduced_exp32",
            "fast32",
            "hcu_sfu",
            "intrinsic32",
            "range_log64",
            "range_exp64",
            "range64",
            "reduced_sqrt64",
            "range_exp_sqrt64",
            "reduced_rsqrt64",
            "range_exp_rsqrt64",
        ),
        use_aligned,
        use_index32,
        math_mode in ("fast32", "hcu_sfu", "intrinsic32") or musa_pi32,
        backend == "hygon" and math_mode == "hcu_sfu",
        math_mode == "intrinsic32",
        f64 and math_mode in ("range_log64", "range64"),
        f64 and math_mode in ("range_exp64", "range64", "range_exp_sqrt64", "range_exp_rsqrt64"),
        math_mode == "reduced_exp32",
        f64 and math_mode in ("reduced_sqrt64", "range_exp_sqrt64"),
        f64 and math_mode in ("reduced_rsqrt64", "range_exp_rsqrt64"),
        math_mode == "packed_interleave",
    )
    return kernel, scalars, constants, launch_options


def _optional_constants(launch_options):
    return tuple(launch_options.pop(name, False) for name in ("STANDARD_PARAMS", "MUSA_PI32", "XPU_SCALAR_BASE"))


@dataclass
class _PreparedStandardLaunch:
    launcher: CachedKernelLauncher
    grid: tuple
    scalars: tuple
    constants: tuple
    options: tuple
    bound_launcher: PreparedKernelLaunch | None = None
    _signature: object = None
    _key: object = None

    def launch(self, state, out):
        arguments = (state, out) + self.scalars
        signatures = (
            (state.dtype, state.data_ptr() % 16 == 0),
            (out.dtype, out.data_ptr() % 16 == 0),
        )
        debug = self.launcher._kernel.debug
        signature = (driver.active.get_current_device(), signatures, debug)
        if signature != self._signature:
            self._key = launch_specialization_key(
                arguments, 2, self.constants, self.options, debug, tensor_signatures=signatures
            )
            self._signature = signature
        # Pointers remain dynamic; the existing launcher resolves the current
        # stream and honors runtime hooks and portable mode on every call.
        if self.bound_launcher is not None:
            generate.last_kernel = self.bound_launcher.launch(
                arguments, self.constants, self.options, specialization_key=self._key
            )
        else:
            generate.last_kernel = self.launcher.launch(
                self.grid, arguments, self.constants, self.options, specialization_key=self._key
            )
        return out


def prepare_standard_launch(
    n,
    nstreams,
    distribution,
    dtype,
    mu,
    sigma,
    backend,
    *,
    block=256,
    num_warps=4,
    rounds=1,
    num_stages=1,
    math_mode="native",
    aligned=False,
    index32=False,
    maxnreg=None,
    compiler_options=None,
    specialize_standard=False,
    xpu_scalar_base=False,
):
    """Prepare the default PPU single-kernel path without capturing tensors.

    The generator chooses this path only for its recorded PPU profiles.
    Packed/software-f64, explicit launch overrides and two-stage calls retain
    their existing dispatch. Flag construction is shared with generate.
    """
    kind = {"raw32": 0, "uniform": 1, "normal": 2, "lognormal": 3}[distribution]
    f64 = kind != 0 and dtype == torch.float64
    pair = 2 if kind >= 2 else 1
    use_aligned, use_index32, grid = _launch_geometry(n, nstreams, pair, block, rounds, aligned, index32, False)
    options = _launch_options(backend, num_warps, num_stages, maxnreg, compiler_options)
    kernel, scalars, constants, options = _standard_launch_spec(
        n,
        nstreams,
        kind,
        f64,
        mu,
        sigma,
        backend,
        block,
        rounds,
        math_mode,
        use_aligned,
        use_index32,
        False,
        specialize_standard,
        xpu_scalar_base,
        options,
    )
    constants += _optional_constants(options)
    launcher = _standard_launcher(kernel, tuple(options))
    return _PreparedStandardLaunch(
        launcher,
        grid,
        scalars,
        constants,
        tuple(options.values()),
        bound_launcher=launcher.prepare(grid) if backend == "thead" else None,
    )


def generate(
    state,
    n,
    distribution="uniform",
    dtype=torch.float32,
    mu=0.0,
    sigma=1.0,
    block=None,
    num_warps=4,
    rounds=1,
    backend="auto",
    out=None,
    num_stages=1,
    math_mode="native",
    aligned=False,
    index32=False,
    maxnreg=None,
    compiler_options=None,
    specialize_standard=False,
    xpu_scalar_base=False,
    cached_launch=True,
):
    """Fuse XORWOW and a distribution in one device launch, advancing state.

    distribution is raw32/uniform/normal/lognormal; raw32 returns signed int32
    storage containing the original uint32 bits. Each stream s owns offsets
    s, s+S, s+2*S, ...; block, warps and rounds do not change this mapping.
    rounds unrolls the loop; it does not create additional random streams.

    Uniform f32 is (top24+1)*2**-24, consuming one raw word. Uniform f64 is
    (K+1)*2**-53, with K assembled from 27+26 bits of two raw words.
    Normal/lognormal write Box--Muller pairs at 2*k*S+s and (2*k+1)*S+s.
    F32 uses rounded (float32(raw)+1)*2**-32 for its radial input and top24
    bits for angle; f64 uses 53 bits for each. A pair consumes 2/4 words.
    A pair with only its first output in range still consumes both samples;
    its second sample is discarded. No normal cache is stored. Streams with
    no output do not advance. Repeated calls restart interleaving at stream 0;
    concatenation matches a single call for lengths divisible by S (uniform)
    or 2*S (normal/lognormal).

    mu/sigma apply to normal; for lognormal they parameterize log(output).
    Parameters travel as integer IEEE bits, preserving Python binary64 input.
    specialize_standard skips the identity affine transform on MUSA when
    mu=0 and sigma=1; other parameters use the original transform.
    Backend libdevice determines native transcendental rounding and overflow.
    range_log64/range_exp64/range64 use reduced trig and replace native f64
    log/exp/both with range-math candidates; f32 uses the existing reduced path.
    reduced_exp32 keeps vendor log/sqrt and reduced sin/cos, selecting tl.exp
    only for f32 lognormal (MetaX codegen separates these math-library costs).
    musa_pi32 selects MUSA's pi-scaled f32 sin/cos externs on twice the input
    turns, keeping intrinsic32's tl.log, tl.sqrt and tl.exp paths.
    reduced_sqrt64/range_exp_sqrt64 separately select tl.sqrt for f64.
    reduced_rsqrt64/range_exp_rsqrt64 use an f32 rsqrt seed and two f64 FMA
    residual corrections for the radial sqrt; all subsequent math stays f64.
    Their sampled CPU error is not a device or full-domain accuracy guarantee.
    Ascend/CoreX/P800 f64 normal/lognormal, or explicit soft64/soft64_outlined modes, use a software
    f64 path with three f32 components and integer binary64 packing. It is
    distinct from casting a single f32 result. Its finite-parameter prototype
    has initial CoreX device comparisons; other backend validation and parameter
    coverage are recorded separately in the measurement reports.
    soft64_outlined retains the failed noinline experiment: CoreX 3.1 rejects
    vector arguments to noinline functions. It is not selected by profiles.
    Ascend soft64_loop_gather uses constant-table coefficient gathers;
    soft64_loop_gather_direct also selects the direct product-splitting candidate.
    P800 defaults to soft64_xpu: explicit mul.f.rn preserves product rounding,
    and a private JIT compiler binding omits the unnecessary StoreControl pass.
    Its separate backend hash keeps that compilation distinct in the cache.
    xpu_scalar_base selects P800 continuous DMA for complete program blocks
    and elementwise DMA for partial blocks; packed f64 stores uint64 bit patterns.
    Ascend, CoreX and P800 uniform f64 use integer bit packing, preserving all 53 random bits.
    packed_clz selects the measured CoreX CLZ/i64-store uniform-f64 candidate.
    packed_interleave combines Ascend's low/high i32 words before a contiguous
    store, using tl.interleave.
    maxnreg forwards an explicitly requested compiler register limit; its
    availability and spilling effects are backend-specific.
    compiler_options forwards recorded backend launch settings such as XPU
    core_num/groups_per_cluster. XPU defaults to is_use_mask_zero=True because
    the installed compiler otherwise drops output-store masks during lowering.
    This uses the existing device DMA setting recorded in the P800 report.
    aligned and index32 enable optional lowering specializations when the
    current shape fits their index and full-batch conditions.
    The latest launched CompiledKernel is available as generate.last_kernel.
    cached_launch reuses the standard Hygon kernel's complete specialization
    through the shared runtime launcher; pass False for the ordinary JIT comparison.
    Other backend and packed/software-f64 paths keep their existing launch.
    """
    kind = {"raw32": 0, "uniform": 1, "normal": 2, "lognormal": 3}[distribution]
    if kind == 0:
        dtype = torch.int32
    if out is None:
        out = allocate_output(n, dtype, state.device, backend)
    if n == 0:
        generate.last_kernel = None
        return out
    f64 = dtype == torch.float64
    backend = resolve_backend(backend, state.device)
    if block is None:
        block = 64 if backend == "ascend" and f64 and kind >= 2 else 256
    if backend == "ascend" and f64 and kind >= 2 and math_mode == "native":
        math_mode = "soft64_loop_direct"
        compiler_options = {"sync_solver": False, **(compiler_options or {})}
    if backend == "kunlunxin" and f64 and kind >= 2 and math_mode == "native":
        math_mode = "soft64_xpu"
    nstreams = state.shape[1]
    pair = 2 if kind >= 2 else 1
    soft64 = (
        f64
        and kind >= 2
        and (
            backend in ("ascend", "iluvatar")
            or math_mode in ("soft64", "soft64_outlined", "soft64_joined", "soft64_flat")
            or (backend == "kunlunxin" and math_mode in ("soft64_loop", "soft64_xpu"))
        )
    )
    packed_uniform = (
        f64
        and kind == 1
        and (
            backend in ("ascend", "iluvatar", "kunlunxin")
            or math_mode in ("packed64", "soft64", "soft64_outlined", "soft64_joined", "soft64_flat")
        )
    )
    packed = soft64 or packed_uniform
    use_aligned, use_index32, grid = _launch_geometry(n, nstreams, pair, block, rounds, aligned, index32, packed)
    kernel_out = out.view(torch.int32) if packed else out
    launch_options = _launch_options(backend, num_warps, num_stages, maxnreg, compiler_options)
    if f64 and kind == 1 and math_mode == "packed_clz":
        generate.last_kernel = _corex_uniform_entry()[grid](
            state,
            out.view(torch.int64),
            n,
            nstreams,
            block,
            rounds,
            use_aligned,
            use_index32,
            **launch_options,
        )
        return out
    if soft64:
        if backend == "kunlunxin" and math_mode in ("soft64_loop", "soft64_xpu"):
            from .xpu_soft64_math import bind_xpu_soft64

            functions = bind_xpu_soft64(
                runtime_loops=math_mode == "soft64_loop", inline_product=math_mode == "soft64_xpu"
            )
            generate_standard_soft64_kernel = functions.generate_standard_soft64_kernel
            generate_affine_soft64_kernel = functions.generate_affine_soft64_kernel
        elif backend == "ascend":
            from .ascend_soft64_math import bind_ascend_soft64

            ascend_functions = bind_ascend_soft64(
                runtime_loops=math_mode
                in ("soft64_loop", "soft64_loop_direct", "soft64_loop_gather", "soft64_loop_gather_direct"),
                direct_product=math_mode in ("soft64_loop_direct", "soft64_loop_gather_direct"),
                gather_coefficients=math_mode in ("soft64_loop_gather", "soft64_loop_gather_direct"),
            )
            generate_standard_soft64_kernel = ascend_functions.generate_standard_soft64_kernel
            generate_affine_soft64_kernel = ascend_functions.generate_affine_soft64_kernel
        else:
            from .soft64_kernels import generate_affine_soft64_kernel, generate_standard_soft64_kernel
        if mu == 0.0 and sigma == 1.0:
            kernel = generate_standard_soft64_kernel
            arguments = (state, kernel_out, n)
            kernel_options = {
                "OUTLINED_EXP": math_mode == "soft64_outlined",
                "JOINED_PAIR": math_mode == "soft64_joined",
                "FLAT_PAIR": math_mode == "soft64_flat",
            }
        else:
            kernel = generate_affine_soft64_kernel
            parameters = _soft_parameters(_float_bits(mu, True), _float_bits(sigma, True), str(state.device))
            arguments = (state, kernel_out, n, parameters)
            kernel_options = {}
        if backend == "ascend":
            kernel_options["PACKED_INTERLEAVE"] = True
        if backend == "kunlunxin":
            kernel_options["INTEGER_QUADRANT_SIGN"] = True
            kernel_options["XPU_SCALAR_BASE"] = xpu_scalar_base
            if xpu_scalar_base:
                kernel_options["PACKED_INTERLEAVE"] = True
            if math_mode == "soft64_xpu":
                from .xpu_compiler import bind_xpu_without_store_control

                kernel = bind_xpu_without_store_control(kernel)
        generate.last_kernel = kernel[grid](
            *arguments,
            nstreams,
            block,
            rounds,
            kind,
            use_aligned,
            use_index32,
            **launch_options,
            **kernel_options,
        )
        return out
    kernel, scalars, constants, launch_options = _standard_launch_spec(
        n,
        nstreams,
        kind,
        f64,
        mu,
        sigma,
        backend,
        block,
        rounds,
        math_mode,
        use_aligned,
        use_index32,
        packed_uniform,
        specialize_standard,
        xpu_scalar_base,
        launch_options,
    )
    arguments = (state, kernel_out) + scalars
    # Reuse the compiled specialization and resolve the current stream on each call.
    if cached_launch and backend in ("hygon", "metax", "thead") and not packed_uniform:
        constants += _optional_constants(launch_options)
        launcher = _standard_launcher(kernel, tuple(launch_options))
        option_values = tuple(launch_options.values())
        key = launch_specialization_key(arguments, 2, constants, option_values, kernel.debug)
        generate.last_kernel = launcher.launch(
            grid,
            arguments,
            constants,
            option_values,
            specialization_key=key,
        )
    else:
        generate.last_kernel = kernel[grid](*arguments, *constants, **launch_options)
    return out


generate.last_kernel = None
