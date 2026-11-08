"""Select fused blocks or bounded continuation for Ascend MTGP32 calls."""

import torch

from ._mtgp32_ascend import generate_distribution as generate_distribution_separate
from ._mtgp32_ascend_buffered import MAX_BUFFERED_OUTPUTS, buffered, prepare_stream
from ._mtgp32_ascend_fused import aligned
from ._mtgp32_sequence import generate_mtgp32_contiguous


def use_buffer(engine, n, words):
    return (
        n < MAX_BUFFERED_OUTPUTS
        or engine.offset % 256
        or words % 256
        or getattr(engine, "_chunk_cache", None) is not None
    )


def generate(engine, out, **kwargs):
    n = out.numel()
    if not n:
        return out
    if not kwargs and getattr(engine, "_use_buffered", True) and use_buffer(engine, n, n):
        return buffered(engine, out)
    if getattr(engine, "_ascend_mtgp_buffer", None) is not None:
        prepare_stream(engine, out.device)
    if not kwargs and n % 256 == 0 and engine.offset % 256 == 0 and getattr(engine, "_chunk_cache", None) is None:
        return aligned(engine, out, 0)
    generate_mtgp32_contiguous(engine, out, engine.seed, engine.offset, int(kwargs.get("num_warps", 4)))
    engine.offset += n
    return out


def generate_distribution(engine, out, distribution, mean=0.0, stddev=1.0, **kwargs):
    n = out.numel()
    if not n:
        return out
    f64 = out.dtype == torch.float64
    if f64 and distribution in ("normal", "lognormal") and mean == 0.0 and stddev == 1.0 and not kwargs:
        from ._mtgp32_ascend_soft64 import standard

        return standard(engine, out, distribution)
    words = n * (2 if f64 else 1) if distribution == "uniform" else (4 if f64 else 2) * ((n + 1) // 2)
    supported = not f64 or distribution == "uniform"
    if supported and not kwargs and getattr(engine, "_use_buffered", True) and use_buffer(engine, n, words):
        return buffered(engine, out, distribution, mean, stddev)
    if getattr(engine, "_ascend_mtgp_buffer", None) is not None:
        prepare_stream(engine, out.device)
    aligned_words = n * (2 if f64 else 1)
    if (
        supported
        and not kwargs
        and aligned_words % 256 == 0
        and engine.offset % 256 == 0
        and getattr(engine, "_chunk_cache", None) is None
    ):
        mode = 4 if f64 else {"uniform": 1, "normal": 2, "lognormal": 3}[distribution]
        return aligned(engine, out, mode, mean, stddev)
    return generate_distribution_separate(engine, out, distribution, mean, stddev, **kwargs)
