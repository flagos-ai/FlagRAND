"""Pack 53-bit uniform values as two signed i32 words, without f64 arithmetic.

The returned order is (low_word, high_word), for a little-endian float64
output viewed as int32. All device shifts have compile-time constant counts;
right shifts explicitly mask sign extension for older Ascend lowering.
"""

import triton
import triton.language as tl


@triton.jit
def store_interleaved_words(
    OUT, base, low, high, valid_count, BLOCK: tl.constexpr, ALIGNED: tl.constexpr, XPU_SCALAR_BASE: tl.constexpr = False
):
    if XPU_SCALAR_BASE:
        bits = low.to(tl.uint32).to(tl.uint64) | (high.to(tl.uint32).to(tl.uint64) << 32)
        lane = tl.arange(0, BLOCK)
        output_base = (OUT + base * 2).to(tl.pointer_type(tl.uint64))
        mask = lane < valid_count
        if valid_count >= BLOCK:
            tl.store(output_base + lane, bits, mask, offset_state_policy="continuous")
        else:
            tl.store(output_base + lane, bits, mask, offset_state_policy="unknown")
    else:
        words = tl.interleave(low, high)
        offsets = base * 2 + tl.arange(0, BLOCK * 2)
        if ALIGNED:
            tl.store(OUT + offsets, words)
        else:
            tl.store(OUT + offsets, words, tl.arange(0, BLOCK * 2) < valid_count * 2)


@triton.jit
def _normalize_step(high, low, leading, SHIFT: tl.constexpr):
    take = ((high >> (32 - SHIFT)) & ((1 << SHIFT) - 1)) == 0
    next_high = (high << SHIFT) | ((low >> (32 - SHIFT)) & ((1 << SHIFT) - 1))
    high = tl.where(take, next_high, high)
    low = tl.where(take, low << SHIFT, low)
    leading += tl.where(take, SHIFT, 0)
    return high, low, leading


@triton.jit
def uniform64_words(raw0, raw1):
    """Return the exact binary64 words of (K + 1) * 2**-53.

    K = ((raw0 >> 5) << 26) | (raw1 >> 6), interpreting both
    arguments as logical uint32 bit patterns. Every intermediate is i32.
    """
    a = raw0.to(tl.int32, bitcast=True)
    b = raw1.to(tl.int32, bitcast=True)
    low = (((a & 0x7E0) << 21) | ((b >> 6) & 0x03FFFFFF)) + 1
    high = ((a >> 11) & 0x001FFFFF) + tl.where(low == 0, 1, 0)

    # Move the leading one of the two-word integer to bit 63. This also
    # handles K+1 == 2**53: its shift count is 10 and its fraction is zero.
    take32 = high == 0
    high = tl.where(take32, low, high)
    low = tl.where(take32, 0, low)
    leading = tl.where(take32, 32, 0)
    high, low, leading = _normalize_step(high, low, leading, 16)
    high, low, leading = _normalize_step(high, low, leading, 8)
    high, low, leading = _normalize_step(high, low, leading, 4)
    high, low, leading = _normalize_step(high, low, leading, 2)
    high, low, leading = _normalize_step(high, low, leading, 1)

    # 1023 + (63 - leading) - 53 = 1033 - leading.
    low_word = ((low >> 11) & 0x001FFFFF) | (high << 21)
    high_word = ((high >> 11) & 0x000FFFFF) | ((1033 - leading) << 20)
    return low_word, high_word


def _signed_i32(value):
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def uniform64_words_python(raw0, raw1):
    """CPU equivalent of the device's signed-i32 operations."""
    a, b = _signed_i32(raw0), _signed_i32(raw1)
    low = _signed_i32((((a & 0x7E0) << 21) | ((b >> 6) & 0x03FFFFFF)) + 1)
    high = ((a >> 11) & 0x001FFFFF) + int(low == 0)
    if high == 0:
        high, low, leading = low, 0, 32
    else:
        leading = 0
    for shift in (16, 8, 4, 2, 1):
        if ((high >> (32 - shift)) & ((1 << shift) - 1)) == 0:
            high = _signed_i32((high << shift) | ((low >> (32 - shift)) & ((1 << shift) - 1)))
            low = _signed_i32(low << shift)
            leading += shift
    low_word = _signed_i32(((low >> 11) & 0x001FFFFF) | (high << 21))
    high_word = ((high >> 11) & 0x000FFFFF) | ((1033 - leading) << 20)
    return low_word, high_word


def compare_python_words(random_count=10000):
    """Necessary bit-exact comparison; no device import or execution here."""
    import random
    import struct

    # Exercise carry into the upper word and both sides of every exponent.
    numerators = {1, 2, 3, (1 << 53) - 1, 1 << 53}
    for bit in range(1, 53):
        numerators.update(((1 << bit) - 1, 1 << bit, (1 << bit) + 1))
    inputs = []
    for numerator in sorted(numerators):
        k = numerator - 1
        a = (k >> 26) << 5
        b = (k & ((1 << 26) - 1)) << 6
        inputs.extend(((a, b), (a | 31, b | 63)))
    rng = random.Random(20260930)
    inputs.extend((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(random_count))
    for a, b in inputs:
        k = ((a >> 5) << 26) | (b >> 6)
        actual = struct.pack("<ii", *uniform64_words_python(a, b))
        expected = struct.pack("<d", (k + 1) * 2.0**-53)
        if actual != expected:
            raise AssertionError((hex(a), hex(b), actual.hex(), expected.hex()))
    return {
        "boundary_inputs": len(inputs) - random_count,
        "random_inputs": random_count,
        "bit_exact": True,
        "comparison": "struct.pack('<d', (K+1)*2**-53)",
    }
