"""Optional CoreX uniform-f64 experiment: CLZ plus integer binary64 packing.

The installed CoreX 4.4 / Triton 3.1 cuda.libdevice.clz declaration accepts
signed i32 or i64 and returns i32. This candidate uses the i32 overload once
per output. No device compilation, instruction lowering or timing is implied
by the local source and integer comparisons.
"""

import triton.language as tl

from ._binding import bind_kernel
from .ascend_kernels import _next_i32


def build_uniform64_kernel(device_math):
    """Capture the installed libdevice module and return a uniform-only JIT.

    ABI: (STATE, OUT, N, NSTREAMS, BLOCK, ROUNDS, ALIGNED=False, INDEX32=False).
    STATE is the common signed-i32 SoA [6, NSTREAMS]. OUT is a little-endian
    float64 allocation viewed as int64. Each output k*S+s consumes two raw
    words from stream s. Invalid outputs neither store nor advance state.

    ALIGNED assumes N is divisible by NSTREAMS*ROUNDS and NSTREAMS by BLOCK.
    INDEX32 assumes all state and output offsets fit signed i32.
    The host owns those existing launch choices. N=0 requires no launch.
    """

    def generate_corex_uniform64_kernel(
        STATE,
        OUT,
        N,
        NSTREAMS: tl.constexpr,
        BLOCK: tl.constexpr,
        ROUNDS: tl.constexpr,
        ALIGNED: tl.constexpr = False,
        INDEX32: tl.constexpr = False,
    ):
        INDEX: tl.constexpr = tl.int32 if INDEX32 else tl.int64
        s = tl.cast(tl.program_id(0), INDEX) * BLOCK + tl.cast(tl.arange(0, BLOCK), INDEX)
        n = tl.cast(N, INDEX)
        if ALIGNED:
            present = True
        else:
            present = (s < NSTREAMS) & (s < n)
        x = tl.load(STATE + s, present, other=1)
        y = tl.load(STATE + NSTREAMS + s, present, other=0)
        z = tl.load(STATE + 2 * NSTREAMS + s, present, other=0)
        w = tl.load(STATE + 3 * NSTREAMS + s, present, other=0)
        v = tl.load(STATE + 4 * NSTREAMS + s, present, other=0)
        d = tl.load(STATE + 5 * NSTREAMS + s, present, other=0)
        if ALIGNED:
            batches = n // (NSTREAMS * ROUNDS)
        else:
            batches = tl.cdiv(n, NSTREAMS * ROUNDS)
        for batch in range(0, batches):
            for unroll in tl.static_range(ROUNDS):
                k = tl.cast(batch, INDEX) * ROUNDS + unroll
                index = k * NSTREAMS + s
                if ALIGNED:
                    active = True
                else:
                    active = present & (index < n)
                a, nx, ny, nz, nw, nv, nd = _next_i32(x, y, z, w, v, d)
                b, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)

                top27 = ((a >> 5) & 0x07FFFFFF).to(tl.int64)
                bottom26 = ((b >> 6) & 0x03FFFFFF).to(tl.int64)
                numerator = ((top27 << 26) | bottom26) + 1
                high = (numerator >> 32).to(tl.int32)
                low = numerator.to(tl.int32)
                # The selected word is nonzero, including numerator=2**53.
                word = tl.where(high != 0, high, low)
                leading = device_math.clz(word) + tl.where(high != 0, 0, 32)
                # Shift 10..63 is valid for every numerator in [1, 2**53].
                # Mask after arithmetic right shift to remove sign extension.
                normalized = numerator << leading.to(tl.int64)
                fraction = (normalized >> 11) & 0x000FFFFFFFFFFFFF
                exponent = (1033 - leading).to(tl.int64) << 52
                bits = fraction | exponent
                tl.store(OUT + index, bits, active)

                if ALIGNED:
                    x, y, z, w, v, d = nx, ny, nz, nw, nv, nd
                else:
                    x = tl.where(active, nx, x)
                    y = tl.where(active, ny, y)
                    z = tl.where(active, nz, z)
                    w = tl.where(active, nw, w)
                    v = tl.where(active, nv, v)
                    d = tl.where(active, nd, d)
        tl.store(STATE + s, x, present)
        tl.store(STATE + NSTREAMS + s, y, present)
        tl.store(STATE + 2 * NSTREAMS + s, z, present)
        tl.store(STATE + 3 * NSTREAMS + s, w, present)
        tl.store(STATE + 4 * NSTREAMS + s, v, present)
        tl.store(STATE + 5 * NSTREAMS + s, d, present)

    return bind_kernel(generate_corex_uniform64_kernel, do_not_specialize=["N"])


def uniform64_bits_python(raw0, raw1):
    """Pure-integer equivalent; accepts signed or unsigned 32-bit patterns."""
    a, b = raw0 & 0xFFFFFFFF, raw1 & 0xFFFFFFFF
    numerator = (((a >> 5) << 26) | (b >> 6)) + 1
    high, low = numerator >> 32, numerator & 0xFFFFFFFF
    word = high if high else low
    leading = 32 - word.bit_length() + (0 if high else 32)
    normalized = (numerator << leading) & 0xFFFFFFFFFFFFFFFF
    fraction = (normalized >> 11) & 0x000FFFFFFFFFFFFF
    return fraction | ((1033 - leading) << 52)


def compare_python_bits(random_count=10000):
    """Compare boundary/random packing to Python's exact binary64 grid."""
    import random
    import struct

    limit = 1 << 53
    ks = {0, 1, limit - 1}
    for bit in range(54):
        ks.update(k for k in ((1 << bit) - 1, 1 << bit, (1 << bit) + 1) if 0 <= k < limit)
    inputs = []
    for k in sorted(ks):
        a, b = (k >> 26) << 5, (k & ((1 << 26) - 1)) << 6
        inputs.extend(((a, b), (a | 31, b | 63)))
    boundary_count = len(inputs)
    rng = random.Random(20260930)
    inputs.extend((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(random_count))
    for a, b in inputs:
        k = ((a >> 5) << 26) | (b >> 6)
        expected = struct.unpack("<Q", struct.pack("<d", (k + 1) * 2.0**-53))[0]
        actual = uniform64_bits_python(a, b)
        signed_a = a if a < 0x80000000 else a - 0x100000000
        signed_b = b if b < 0x80000000 else b - 0x100000000
        signed_actual = uniform64_bits_python(signed_a, signed_b)
        if actual != expected or signed_actual != expected:
            raise AssertionError((hex(a), hex(b), hex(actual), hex(expected), hex(signed_actual)))
    return {
        "boundary_inputs": boundary_count,
        "random_inputs": random_count,
        "signed_and_unsigned_patterns": True,
        "bit_exact": True,
        "comparison": "struct.pack('<d', (K+1)*2**-53)",
        "scope": "CPU integer packing only; device not compiled or executed",
    }
