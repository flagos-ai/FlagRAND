"""Integer reference and jump matrices for L'Ecuyer's MRG32k3a.

State triples are ordered oldest to newest. The seed mapping is FlagRand's;
the recurrence and 2**76 stream spacing are MRG32k3a's.
"""

from functools import lru_cache

M1 = 4294967087
M2 = 4294944443
A1 = ((0, 1, 0), (0, 0, 1), (M1 - 810728, 1403580, 0))
A2 = ((0, 1, 0), (0, 0, 1), (M2 - 1370589, 0, 527612))
_MASK64 = (1 << 64) - 1


def matmul(a, b, modulus):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) % modulus for j in range(3)) for i in range(3))


def matvec(a, state, modulus):
    return tuple(sum(a[i][j] * state[j] for j in range(3)) % modulus for i in range(3))


def matpow(a, exponent, modulus):
    result = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    while exponent:
        if exponent & 1:
            result = matmul(result, a, modulus)
        a = matmul(a, a, modulus)
        exponent >>= 1
    return result


def reference_step(state):
    x0, x1, x2, y0, y1, y2 = state
    x = (1403580 * x1 - 810728 * x0) % M1
    y = (527612 * y2 - 1370589 * y0) % M2
    z = x - y
    if z <= 0:
        z += M1
    return z, (x1, x2, x, y1, y2, y)


def seed_state(seed):
    """Map a 64-bit seed into six nonzero residues using SplitMix64."""
    words = []
    x = seed & _MASK64
    for modulus in (M1, M1, M1, M2, M2, M2):
        x = (x + 0x9E3779B97F4A7C15) & _MASK64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
        words.append(z % (modulus - 1) + 1)
    return tuple(words)


@lru_cache(maxsize=1)
def stream_jump():
    return matpow(A1, 1 << 76, M1), matpow(A2, 1 << 76, M2)


def initial_states(seed, nstreams, offset=0):
    """Return SoA states after skipping offset interleaved raw outputs.

    Stream i starts at A**(i * 2**76) applied to the seed state. Advancing
    offset raw outputs advances every stream offset//nstreams times and
    the first offset%nstreams streams once more.
    """
    base = seed_state(seed)
    j1, j2 = stream_jump()
    rounds, remainder = divmod(offset, nstreams)
    x = matvec(matpow(A1, rounds, M1), base[:3], M1)
    y = matvec(matpow(A2, rounds, M2), base[3:], M2)
    rows = [[] for _ in range(6)]
    for stream in range(nstreams):
        state = x + y
        if stream < remainder:
            _, state = reference_step(state)
        for row, value in zip(rows, state):
            row.append(value if value < (1 << 31) else value - (1 << 32))
        x = matvec(j1, x, M1)
        y = matvec(j2, y, M2)
    return rows
