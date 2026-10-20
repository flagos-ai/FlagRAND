"""Integer-only CPU reference for the standalone GPU kernels.

Rows have shape (6, nstreams), in x, y, z, w, v, d order. Words are unsigned
Python integers here and signed int32 bit patterns in device storage.
"""

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15


def splitmix64(value):
    """The SplitMix64 mixing permutation, without the counter increment."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def seed_words(nstreams=4096, seed=0):
    """Return six SoA rows, with six SplitMix64 counter values per stream.

    Counter (6*s+j+1) is mixed with seed, modulo 2**64. The low 32 bits are
    retained; x's low bit is set so the five-word xorshift state is nonzero.
    This defines project seeds, not cuRAND seeds or its sequence jump scheme.
    """
    rows = [[] for _ in range(6)]
    for stream in range(nstreams):
        for word in range(6):
            value = splitmix64((seed + GOLDEN64 * (6 * stream + word + 1)) & MASK64)
            value &= MASK32
            rows[word].append(value | 1 if word == 0 else value)
    return rows


def next_u32(words):
    """Return (raw_word, new_six_word_state) without mutating words."""
    x, y, z, w, v, d = (int(word) & MASK32 for word in words)
    t = x ^ (x >> 2)
    new_v = ((v ^ (v << 4)) ^ (t ^ (t << 1))) & MASK32
    new_d = (d + 362437) & MASK32
    return (new_v + new_d) & MASK32, [y, z, w, v, new_v, new_d]


def generate_raw(state_rows, n):
    """Generate unsigned words at k*nstreams+s and return copied final rows.

    A stream advances only for an output it owns. Calls restart interleaving
    at stream zero; concatenation equals a single call when intermediate
    lengths are multiples of nstreams.
    """
    rows = [[int(word) & MASK32 for word in row] for row in state_rows]
    nstreams = len(rows[0])
    output = []
    for index in range(n):
        stream = index % nstreams
        value, words = next_u32([row[stream] for row in rows])
        output.append(value)
        for row, word in zip(rows, words):
            row[stream] = word
    return output, rows


def signed_rows(rows):
    """Convert unsigned reference rows to int32-compatible Python values."""
    return [[word if word < (1 << 31) else word - (1 << 32) for word in row] for row in rows]
