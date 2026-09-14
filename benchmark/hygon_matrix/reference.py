"""Independent CPU Philox4x32-10 and explicitly defined distribution transforms."""


def words(seed, offset, count):
    import numpy as np

    counters = np.arange(offset // 4, (offset + count + 3) // 4, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    a = np.zeros_like(counters)
    b = a.copy()
    c = counters & mask
    d = counters >> np.uint64(32)
    k0, k1 = seed & 0xFFFFFFFF, seed >> 32
    for _ in range(10):
        p = a * np.uint64(0xD2511F53)
        q = c * np.uint64(0xCD9E8D57)
        a, b, c, d = (
            ((q >> np.uint64(32)) ^ b ^ np.uint64(k0)) & mask,
            q & mask,
            ((p >> np.uint64(32)) ^ d ^ np.uint64(k1)) & mask,
            p & mask,
        )
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    packed = np.stack([a, b, c, d], axis=1).reshape(-1).astype(np.uint32)
    return packed[offset % 4 : offset % 4 + count]


def reference(distribution, dtype, n, seed, offset, mean=0.0, stddev=1.0):
    import numpy as np
    import torch

    if distribution == "raw":
        return torch.from_numpy(words(seed, offset, n).view(np.int32).copy())
    # Normal pairs need their partner even when the final output is masked.
    count = n if distribution == "uniform" else ((n + 1) // 2) * 2
    raw = words(seed, offset, count * (2 if dtype == "float64" else 1)).astype(np.uint64)
    if dtype == "float64":
        u = (raw[0::2] ^ (raw[1::2] << np.uint64(21))).astype(np.float64) * 2.0**-53 + 2.0**-54
    else:
        u = (raw.astype(np.float32) * np.float32(2.0**-32) + np.float32(2.0**-33)).astype(np.float64)
    if distribution == "uniform":
        result = u
    else:
        r = np.sqrt(-2.0 * np.log(np.maximum(u[0::2], 2.0 ** (-53 if dtype == "float64" else -32))))
        angle = 2.0 * np.pi * u[1::2]
        result = np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1).reshape(-1) * stddev + mean
        if distribution == "lognormal":
            result = np.exp(result)
    return torch.from_numpy(result[:n].copy())
