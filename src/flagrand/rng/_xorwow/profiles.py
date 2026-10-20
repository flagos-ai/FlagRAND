"""Recorded launch choices, kept separate from mutable generator state.

Profiles select among measured candidates; they do not claim a global optimum
for arbitrary sizes or a different SDK. State creation stays explicit because
changing nstreams changes the sequence and cannot preserve an existing state.
"""


def launch_profile(backend, n, distribution="uniform", dtype="f32"):
    backend = {
        "bw1000": "hygon",
        "hip": "hygon",
        "hcu": "hygon",
        "maca": "metax",
        "c550": "metax",
        "npu": "ascend",
        "910c": "ascend",
        "p800": "kunlunxin",
        "xpu": "kunlunxin",
        "musa": "mthreads",
        "s5000": "mthreads",
        "ppu": "thead",
        "zw810e": "thead",
        "ppu-zw810e": "thead",
        "corex": "iluvatar",
        "biv150": "iluvatar",
    }.get(backend, backend)
    return {
        "hygon": _hygon,
        "thead": _thead,
        "iluvatar": _iluvatar,
        "metax": _metax,
        "ascend": _ascend,
        "mthreads": _mthreads,
        "kunlunxin": _kunlunxin,
    }[backend](n, distribution, str(dtype).split(".")[-1])


def _kunlunxin(n, distribution, dtype):
    f64 = dtype in ("f64", "float64", "double")
    config = dict(
        block=1024,
        num_warps=-1,
        rounds=1,
        num_stages=1,
        index32=True,
        math_mode="native",
        xpu_scalar_base=True,
        compiler_options={"isCloseUnrollControl": True},
    )
    if f64 and distribution in ("normal", "lognormal"):
        nstreams = 6144
        config["block"] = 64 if distribution == "normal" or n <= 1024 else 128
        evidence = "20261001T012024Z-kunlunxin-r19-soft64-u64-output"
        measured_sizes = [1, 6147, 1048579]
    elif f64:
        if n < 1024:
            nstreams = 6144
            config["block"] = 64
            evidence = "20261001T011928Z-kunlunxin-r18-packed-u64-output"
            measured_sizes = [1, 6147, 1048579]
        else:
            nstreams = 12288
            config["block"] = 512
            evidence = "20261001T012228Z-kunlunxin-r20-uniform64-grid-rounds"
            measured_sizes = [4096, 1048579, 16777219]
    elif n < 1024:
        nstreams = 1536
        config.update(block=128, xpu_scalar_base=False)
        evidence = "20261001T010412Z-kunlunxin-r14-small-f32-grid"
        measured_sizes = [31, 129, 4096, 65536]
    elif distribution in ("normal", "lognormal") and n >= 16777216:
        nstreams = 24576
        config.update(block=512, rounds=4)
        evidence = "20261001T011556Z-kunlunxin-r16-contiguous-dma-grid"
        measured_sizes = [65536, 1048579, 16777219]
    else:
        nstreams = 12288
        if distribution in ("raw32", "uniform") and n >= 262144:
            config["rounds"] = 4
        evidence = "20261001T011136Z-kunlunxin-r15-contiguous-output-dma"
        measured_sizes = [4096, 12291, 1048579, 16777219]
    return dict(
        nstreams=nstreams,
        config=config,
        evidence=evidence,
        runtime="P800 arch3 / Torch 2.9.0+cu129 / Triton 3.0.0",
        measured_sizes=measured_sizes,
        timing_mode="graph",
        graph_captured_launches=2,
        graph_replays=1,
    )


def _mthreads(n, distribution, dtype):
    f64 = dtype in ("f64", "float64", "double")
    memory = distribution in ("raw32", "uniform")
    config = dict(block=128, num_warps=4, rounds=1, num_stages=1, aligned=True, index32=True, math_mode="native")
    measured_sizes = [1048576, 16777216]
    if n <= 4096:
        nstreams = 4096
        config.update(block=64, num_warps=2)
        measured_sizes = [4096]
        if memory:
            evidence = "20260930T102734Z-mthreads-r16-small-memory"
        elif f64:
            config.update(math_mode="range_exp_rsqrt64")
            evidence = "20260930T103512Z-mthreads-r18b-small-f64-jit-binding"
        else:
            config.update(math_mode="intrinsic32")
            evidence = "20260930T102946Z-mthreads-r17-small-f32"
    elif memory:
        config.update(rounds=4)
        if n >= 16777216:
            nstreams = 131072
            evidence = "20260930T101909Z-mthreads-r11-graph-capture-amortization"
            if n >= 67108864:
                evidence = "20260930T103554Z-mthreads-r19-large-memory-tail"
                measured_sizes = [1048579, 67108864]
        else:
            nstreams = 32768
            evidence = "20260930T102126Z-mthreads-r12-memory-final-timing"
    elif f64:
        nstreams = 131072
        config.update(math_mode="range_exp_rsqrt64", specialize_standard=distribution == "lognormal")
        evidence = "20260930T102642Z-mthreads-r15-f64-standard-parameters"
    else:
        config.update(block=64, num_warps=2, rounds=4, math_mode="intrinsic32")
        if distribution == "normal" and n < 16777216:
            nstreams = 32768
            evidence = "20260930T102259Z-mthreads-r13-f32-medium-final-timing"
        else:
            nstreams = 131072
            evidence = "20260930T102552Z-mthreads-r14-f32-standard-parameters"
    return dict(
        nstreams=nstreams,
        config=config,
        evidence=evidence,
        runtime="MTT S5000 arch31 warp32 / Torch-MUSA 2.5 / Triton 3.1 / MUSA 4.3.5",
        measured_sizes=measured_sizes,
        timing_mode="graph",
        graph_captured_launches=64,
        graph_replays=2,
    )


def _ascend(n, distribution, dtype):
    f64 = dtype in ("f64", "float64", "double")
    config = dict(block=256, num_warps=4, rounds=1, num_stages=1, aligned=True, index32=True, math_mode="reduced")
    timing_mode = "graph"
    if f64 and distribution in ("normal", "lognormal"):
        nstreams = 3072
        config.update(block=64, math_mode="soft64_loop_direct", compiler_options={"sync_solver": False})
        config.pop("aligned")
        evidence = "20260930T094212Z-ascend-r27-f64-48-programs"
        measured_sizes = [4096, 65536, 1048576]
        if n >= 16777216:
            evidence = "20260930T095933Z-ascend-r30-f64-large-selected"
            measured_sizes = [16777216]
    elif n <= 4096:
        nstreams = 4096
        measured_sizes = [4096]
        evidence = "20260930T083708Z-ascend-r10-small-vector"
        if distribution == "uniform" and f64:
            config.update(math_mode="packed_interleave")
            evidence = "20260930T083816Z-ascend-r11-small-uniform64"
    else:
        nstreams = 131072
        config.update(block=1024)
        measured_sizes = [1048576, 16777216]
        if distribution in ("raw32", "uniform"):
            config.update(rounds=2, math_mode="packed_interleave")
            evidence = "20260930T082822Z-ascend-r06-program-count-unroll"
            if n == 1048579 or n >= 67108864:
                evidence = "20260930T083857Z-ascend-r12-large-tail"
                measured_sizes = [1048579, 67108864]
        else:
            evidence = "20260930T083151Z-ascend-r08-transform-vector-unroll"
    return dict(
        nstreams=nstreams,
        config=config,
        evidence=evidence,
        runtime="Ascend910_9382 / CANN 9.0 / Triton 3.5.1 / FlagTree 0.6.1+ascend3.5",
        measured_sizes=measured_sizes,
        timing_mode=timing_mode,
    )


def _metax(n, distribution, dtype):
    f64 = dtype in ("f64", "float64", "double")
    config = dict(block=128, num_warps=2, rounds=1, num_stages=1, aligned=True, index32=True)
    if n <= 4096:
        nstreams = 4096
        config.update(aligned=False, index32=False, math_mode="reduced_exp32")
        if distribution in ("raw32", "uniform"):
            config.update(block=64, num_warps=1, math_mode="native")
        elif distribution == "normal" and not f64:
            config.update(block=64, num_warps=1)
        elif distribution == "lognormal" and f64:
            config.update(math_mode="range_exp64")
        evidence = "20260930T080015Z-metax-r10-small-output"
        measured_sizes = [4096]
    elif distribution in ("raw32", "uniform"):
        config.update(math_mode="native", rounds=8)
        if n >= 16777216:
            nstreams = 131072
            evidence = "20260930T075508Z-metax-r07-memory-parallelism"
            measured_sizes = [1048576, 16777216]
            if n >= 67108864:
                evidence = "20260930T080644Z-metax-r12-large-output-tail"
                measured_sizes = [1048579, 67108864]
        else:
            nstreams = 32768
            if f64:
                config.update(rounds=4)
            evidence = "20260930T074829Z-metax-r04-memory-mask-unroll"
            measured_sizes = [1048576, 16777216]
    else:
        nstreams = 131072
        measured_sizes = [1048576, 16777216]
        if not f64:
            config.update(rounds=4, math_mode="reduced_exp32")
            evidence = "20260930T075700Z-metax-r08-transform-unroll"
        elif distribution == "lognormal":
            config.update(math_mode="range_exp_rsqrt64")
            evidence = "20260930T081712Z-metax-r14-f64-rsqrt-refinement"
        else:
            config.update(rounds=4, math_mode="reduced_rsqrt64")
            evidence = "20260930T081712Z-metax-r14-f64-rsqrt-refinement"
    return dict(
        nstreams=nstreams,
        config=config,
        evidence=evidence,
        runtime="MetaX C550 warp64 / MACA 3.7 / Torch 2.8.0+metax3.7.0.7 / Triton 3.0.0",
        measured_sizes=measured_sizes,
        timing_mode="graph",
        scope="persistent state and preallocated output; graph replay timing",
        parameter_scope="normal/lognormal timing uses mu=0 and sigma=1",
        size_policy="Other sizes use the selected candidate; measured_sizes lists the timing evidence.",
    )


def _hygon(n, distribution, dtype):
    f64 = dtype in ("f64", "float64", "double")
    config = dict(block=128, num_warps=2, rounds=1, num_stages=1, aligned=True)
    if n <= 4096:
        nstreams = 4096
        config.update(block=256, num_warps=4, aligned=False, math_mode="native")
        evidence = "20260929T195604Z-hygon-r01-graph"
    elif distribution in ("raw32", "uniform"):
        nstreams = 32768
        config.update(rounds=4, math_mode="native")
        if distribution == "raw32":
            config.update(block=256, num_warps=4)
        evidence = "20260929T203010Z-hygon-r07-memory-unroll"
    elif f64:
        nstreams = 131072
        config.update(math_mode="reduced")
        evidence = "20260929T202206Z-hygon-r05-mask-index"
    else:
        nstreams = 131072
        config.update(math_mode="hcu_sfu", rounds=4 if n >= 16777216 else 1)
        evidence = "20260929T203502Z-hygon-r10-hcu-asm-newline"
    return dict(
        nstreams=nstreams,
        config=config,
        evidence=evidence,
        runtime="BW1000 gfx936 / Triton 3.6.0 / FlagTree 0.6.0+hcu3.6",
        measured_sizes=[4096, 1048576, 16777216],
        scope="persistent state and preallocated output; graph replay timing",
    )


def _thead(n, distribution, dtype):
    f64 = dtype in ("f64", "float64", "double")
    config = dict(block=64, num_warps=2, rounds=4, num_stages=1, aligned=True, index32=False)
    measured_sizes = [1048576, 16777216]
    if n <= 4096:
        nstreams = 4096
        config.update(block=128, num_warps=4, rounds=1, math_mode="fast32")
        evidence = "20260929T210640Z-thead-r10-small-graph-replay"
        measured_sizes = [4096]
    elif distribution in ("raw32", "uniform"):
        nstreams = 32768
        config.update(math_mode="native")
        if n >= 67108864:
            config.update(block=128, num_warps=4)
            evidence = "20260929T210410Z-thead-r08-large-output"
            measured_sizes = [67108864]
        elif distribution == "uniform" and f64:
            config.update(block=128, num_warps=4, aligned=False)
            evidence = "20260929T204600Z-thead-r02-warp-layout"
        else:
            config.update(rounds=8, index32=True)
            evidence = "20260929T210123Z-thead-r06-memory-packing"
    else:
        nstreams = 131072
        config.update(math_mode="reduced" if f64 else "fast32")
        evidence = "20260929T210306Z-thead-r07-f64-range-lowering" if f64 else "20260929T205045Z-thead-r04-residency"
    return dict(
        nstreams=nstreams,
        config=config,
        evidence=evidence,
        runtime="PPU-ZW810E / Torch 2.9.0 / private Triton 3.5.0",
        measured_sizes=measured_sizes,
        scope="persistent state and preallocated output; graph replay timing",
        size_policy=(
            "Other sizes use candidate extrapolation; "
            "measured_sizes lists timing evidence for the selected parameters."
        ),
    )


def _iluvatar(n, distribution, dtype):
    f64 = dtype in ("f64", "float64", "double")
    config = dict(block=256, num_warps=4, rounds=1, num_stages=1, aligned=True, math_mode="fast32")
    nstreams = 32768
    measured_sizes = [1048576, 16777216]
    timing_mode = "graph"
    scope = "persistent state and preallocated output; graph replay timing"
    size_policy = (
        "Other sizes use candidate extrapolation; "
        "measured_sizes lists timing evidence for the selected parameters and implementation."
    )
    if distribution == "uniform" and f64:
        config.update(block=128, num_warps=2, rounds=4, math_mode="packed_clz")
        evidence = "20260929T214414Z-iluvatar-r09-uniform64-clz"
        measured_sizes = [4096, 1048576, 16777216]
    elif f64 and distribution in ("normal", "lognormal"):
        config.update(math_mode="soft64")
        if n >= 16777216:
            nstreams = 131072
            evidence = "20260929T214500Z-iluvatar-r10-soft64-warp64-layout"
        elif distribution == "normal":
            evidence = "20260929T211717Z-iluvatar-r04-software-f64-constexpr"
            measured_sizes = [4096, 1048576]
        else:
            evidence = "20260929T213037Z-iluvatar-r06-standard-f64-pack"
            measured_sizes = [1048576]
            size_policy += (
                " The normal-only lognormal pack was measured at 1048576 in r06; "
                "r04's 4096 result used the earlier general pack."
            )
    elif n <= 4096:
        nstreams = 4096
        config.update(block=128, num_warps=4, rounds=1, aligned=False)
        if distribution in ("raw32", "uniform"):
            evidence = "20260929T222828Z-iluvatar-r15-small-graph-memory"
            measured_sizes = [4096]
        else:
            evidence = "20260929T211125Z-iluvatar-r01-f32-binding"
            measured_sizes = [4096, 1048576]
    elif distribution in ("raw32", "uniform"):
        if n >= 16777216:
            nstreams = 131072
            config.update(block=128, num_warps=2)
            evidence = "20260929T212922Z-iluvatar-r05-warp64-residency"
        else:
            evidence = "20260929T211428Z-iluvatar-r03-f32-layout"
    elif n >= 16777216:
        nstreams = 131072
        config.update(rounds=4)
        evidence = "20260929T213428Z-iluvatar-r07-f32-intrinsics"
    else:
        evidence = "20260929T211428Z-iluvatar-r03-f32-layout"
    return dict(
        nstreams=nstreams,
        config=config,
        evidence=evidence,
        runtime="BI-V150 warp64 / CoreX 4.4 / Torch 2.7.1 / Triton 3.1.0",
        measured_sizes=measured_sizes,
        timing_mode=timing_mode,
        scope=scope,
        parameter_scope="normal/lognormal timing and numerical evidence use mu=0 and sigma=1",
        size_policy=size_policy,
    )
