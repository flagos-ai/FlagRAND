"""Small scalar reference for MTGP32 recurrence and tempering.

The 351-word ring is intentionally different from the GPU's 1024-word ring
and 256-lane updates. It is used for correctness comparison, not generation
performance. Equations: rocrand_mtgp32.h para_rec/temper and the MTGP paper.
"""

from . import _mtgp32_params as params


def initial_state(seed: int, parameter_set: int = 0) -> list[int]:
    hidden = params.HIDDEN_SEEDS[parameter_set]
    folded = (hidden + (hidden >> 16)) & 0xFFFFFFFF
    folded = (folded + (folded >> 8)) & 0xFFFFFFFF
    state = [(folded & 255) * 0x01010101] * 351
    state[0] = (((seed ^ (seed >> 32)) & 0xFFFFFFFF) + parameter_set + 1) & 0xFFFFFFFF
    state[1] = hidden
    for i in range(1, 351):
        previous = state[i - 1]
        state[i] ^= (1812433253 * (previous ^ (previous >> 30)) + i) & 0xFFFFFFFF
    return state


def generate_reference(seed: int, count: int, parameter_set: int = 0) -> list[int]:
    state = initial_state(seed, parameter_set)
    position = params.POS[parameter_set]
    shift_left = params.SH1[parameter_set]
    shift_right = params.SH2[parameter_set]
    base = 16 * parameter_set
    output = []
    for i in range(count):
        index = i % 351
        x = (state[index] & 0xFFF80000) ^ state[(index + 1) % 351]
        x ^= (x << shift_left) & 0xFFFFFFFF
        y = x ^ (state[(index + position) % 351] >> shift_right)
        value = y ^ params.PARAM[base + (y & 15)]
        t = state[(index + position - 1) % 351]
        t ^= t >> 16
        t ^= t >> 8
        output.append(value ^ params.TEMPER[base + (t & 15)])
        state[index] = value
    return output
