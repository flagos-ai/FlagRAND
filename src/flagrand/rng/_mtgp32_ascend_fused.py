"""Complete MTGP32 blocks and fused transforms using one cached Ascend launch."""

import torch
import triton
import triton.language as tl

from flagrand.rng._mtgp32_sequence import ensure_working_state
from flagrand.rng._xorwow.native_math import sincos_turns
from flagrand.rng._xorwow.packed64 import uniform64_words
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

from ._mtgp32_ascend import _step


@triton.jit
def fused_kernel(
    OUT,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    START,
    COUNT,
    STATES: tl.constexpr,
    MODE: tl.constexpr,
    MU: tl.constexpr,
    SIGMA: tl.constexpr,
):
    p = tl.program_id(0)
    first = START % STATES
    state_id = (first + p) % STATES
    first_round = (START // STATES) % 4 + (first + p) // STATES
    pos, sh1, sh2 = tl.load(POS + state_id), tl.load(SH1 + state_id), tl.load(SH2 + state_id)
    lane = tl.arange(0, 256)
    for k in range(tl.cdiv(COUNT - p, STATES)):
        offset = ((first_round + k) * 256) & 1023
        word = _step(STATE, PARAM, TEMPER, state_id * 1024, state_id * 16, lane, offset, pos, sh1, sh2)
        if MODE == 0:
            value = word
        elif MODE == 1:
            value = (((word >> 8) & 16777215).to(tl.float32) + 1.0) * 5.960464477539063e-8
        elif MODE == 4:
            a, b = tl.split(tl.reshape(word, (128, 2)))
            lo, hi = uniform64_words(a, b)
            value = tl.interleave(lo, hi)
        else:
            a, b = tl.split(tl.reshape(word, (128, 2)))
            u = (((a >> 8) & 16777215).to(tl.float32) + 1.0) * 5.960464477539063e-8
            v = ((b >> 8) & 16777215).to(tl.float32) * 5.960464477539063e-8
            radius = tl.sqrt(-2.0 * tl.log(u))
            sine, cosine = sincos_turns(v, False)
            x, y = MU + SIGMA * radius * cosine, MU + SIGMA * radius * sine
            if MODE == 3:
                x, y = tl.exp(x), tl.exp(y)
            value = tl.interleave(x, y)
        tl.store(OUT + (k * STATES + p) * 256 + lane, value)
        tl.debug_barrier()


_launcher = CachedKernelLauncher(
    fused_kernel,
    constexpr_names=("STATES", "MODE", "MU", "SIGMA"),
    option_names=("num_warps", "enable_fp_fusion", "sync_solver"),
)


def aligned(engine, out, mode, mean=0.0, stddev=1.0):
    words = out.numel() * (2 if mode == 4 else 1)
    ensure_working_state(engine, engine.seed, str(out.device))
    start, blocks = engine.offset // 256, words // 256
    state = engine._working_state
    output = out.view(torch.int32) if mode == 4 else out
    constants = (state.shape[0], mode, mean, stddev)
    _launcher.launch(
        (min(blocks, state.shape[0]),),
        (output, state) + engine._ws_params + (start, blocks),
        constants,
        (4, False, False),
        specialization_key=(
            out.device,
            out.dtype,
            out.data_ptr() % 16,
            blocks,
            integer_specialization_key(start),
            constants,
        ),
    )
    engine.offset += words
    engine._ws_next_block_start += words
    return out
