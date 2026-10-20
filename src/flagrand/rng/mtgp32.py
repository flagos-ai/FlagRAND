"""MTGP32 using the existing 192-state, 256-output-block implementation.

The state/parameter initialization and block ordering are reused from
FlagRand_RTX4060. Raw words are retained across partial calls, so changing
distribution or dtype does not discard the unused part of a generated block.
``offset`` counts consumed raw32 words; arbitrary skipahead is unsupported.

Uniform uses one word for f32 and two for f64. Box--Muller uses two/four words
per pair respectively; an odd final output consumes the complete pair.
Aligned raw and uniform-f32 calls use direct complete-block generation;
MetaX uses measured fusion paths for short transformed calls and separate
transforms for larger calls. Partial MetaX calls retain bounded raw prefetch:
physical ``state`` can lead logical ``offset``, so restoring a continuation
requires the raw cache and its positions as well as the state tensor.
Short MetaX uniform-f64 calls also reuse this bounded raw cache.
Device graph replay of partial-cache operations needs work.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache

import torch

from flagrand.runtime.backend import get_vendor_info
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE
from ._mtgp32_sequence import clear_chunk_cache, generate_mtgp32_contiguous
from ._mtgp32_transforms import _uniform_kernel, build_normal_kernel, exp_finite_range

_uniform_launcher = CachedKernelLauncher(_uniform_kernel, constexpr_names=("F64", "BLOCK"))


@lru_cache(maxsize=4)
def _normal_launcher(native_log, finite_exp=False):
    return CachedKernelLauncher(
        build_normal_kernel(native_log, finite_exp),
        constexpr_names=("F64", "LOGNORMAL", "BLOCK"),
        option_names=("num_warps", "enable_fp_fusion"),
    )


def _float_bits(value: float, f64: bool) -> int:
    return struct.unpack("<Q" if f64 else "<I", struct.pack("<d" if f64 else "<f", value))[0]


@dataclass
class Mtgp32Generator:
    seed: int = 0
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset != 0:
            raise ValueError("MTGP32 does not support a nonzero initial offset.")

    @property
    def dimensions(self) -> None:
        return None

    @property
    def state(self) -> torch.Tensor | None:
        """Physical state; a continuation also includes offset and raw cache."""
        return getattr(self, "_working_state", None)

    def reset(self, *, seed: int | None = None, offset: int = 0) -> None:
        if offset != 0:
            raise ValueError("MTGP32 does not support skipahead.")
        if seed is not None:
            self.seed = int(seed)
        self.offset = 0
        self._ws_seed = None
        self._working_state = None
        self._buffer_stats = None
        clear_chunk_cache(self)

    def _uses_raw_buffer(self, words: int) -> bool:
        if not getattr(self, "_use_buffered", True):
            return False
        backend = getattr(self, "_buffered_backend", None)
        if backend is None:
            backend = get_vendor_info().vendor_name == "metax"
            self._buffered_backend = backend
        return backend and (
            getattr(self, "_chunk_cache", None) is not None or self.offset % 256 != 0 or words % 256 != 0
        )

    def generate(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if offset is not None and int(offset) != self.offset:
            raise ValueError("MTGP32 explicit offset override is not supported.")
        if seed is not None:
            return Mtgp32Generator(seed=int(seed)).generate(out, **kwargs)
        if int(kwargs.get("block_size", MTGP32_BLOCK_SIZE)) != MTGP32_BLOCK_SIZE:
            raise ValueError("MTGP32 requires 256 outputs per state update.")
        if out.numel() == 0:
            return out
        if not kwargs and self._uses_raw_buffer(out.numel()):
            from ._mtgp32_buffered import generate_buffered

            return generate_buffered(self, out)
        if (
            self.offset % MTGP32_BLOCK_SIZE == 0
            and out.numel() % MTGP32_BLOCK_SIZE == 0
            and getattr(self, "_chunk_cache", None) is None
        ):
            from ._mtgp32_fused import generate_mtgp32_aligned

            return generate_mtgp32_aligned(
                self,
                out,
                num_warps=int(kwargs.get("num_warps", 4 if get_vendor_info().vendor_name in ("metax", "hygon") else 8)),
                cached_launch=bool(kwargs.get("cached_launch", True)),
                keyed_launch=bool(kwargs.get("keyed_launch", True)),
                linear_table=bool(kwargs.get("linear_table", get_vendor_info().vendor_name == "metax")),
                cache_tensor_signature=bool(kwargs.get("cache_tensor_signature", True)),
            )
        generate_mtgp32_contiguous(
            self,
            out,
            self.seed,
            self.offset,
            int(kwargs.get("num_warps", 4 if get_vendor_info().vendor_name in ("metax", "hygon") else 8)),
            cached_launch=bool(kwargs.get("cached_launch", True)),
            keyed_launch=bool(kwargs.get("keyed_launch", True)),
            linear_table=bool(kwargs.get("linear_table", get_vendor_info().vendor_name == "metax")),
        )
        self.offset += out.numel()
        return out

    def generate_uniform(self, out: torch.Tensor, **kwargs: object) -> torch.Tensor:
        n = out.numel()
        if n == 0:
            return out
        f64 = out.dtype == torch.float64
        if not kwargs and self._uses_raw_buffer(n * (2 if f64 else 1)):
            from ._mtgp32_buffered import generate_buffered

            return generate_buffered(self, out, "uniform")
        if (
            not kwargs
            and f64
            and n <= 65536
            and getattr(self, "_use_buffered", True)
            and getattr(self, "_buffered_backend", False)
        ):
            from ._mtgp32_buffered import generate_buffered

            return generate_buffered(self, out, "uniform")
        if (
            f64
            and n % 128 == 0
            and self.offset % 256 == 0
            and getattr(self, "_chunk_cache", None) is None
            and kwargs.get("fused_transform", True)
            and set(kwargs).issubset(
                {"cached_launch", "keyed_launch", "fused_transform", "num_warps", "cache_tensor_signature"}
            )
            and get_vendor_info().vendor_name == "metax"
        ):
            from ._mtgp32_direct_pairs import generate_direct_pairs

            generate_direct_pairs(
                self,
                out,
                "uniform",
                num_warps=int(kwargs.get("num_warps", 2)),
                cached_launch=bool(kwargs.get("cached_launch", True)),
                keyed_launch=bool(kwargs.get("keyed_launch", True)),
                cache_tensor_signature=bool(kwargs.get("cache_tensor_signature", True)),
            )
            return out
        if (
            not f64
            and set(kwargs).issubset(
                {
                    "cached_launch",
                    "num_warps",
                    "keyed_launch",
                    "fused_transform",
                    "linear_table",
                    "cache_tensor_signature",
                }
            )
            and self.offset % MTGP32_BLOCK_SIZE == 0
            and n % MTGP32_BLOCK_SIZE == 0
            and getattr(self, "_chunk_cache", None) is None
        ):
            from ._mtgp32_uniform_f32 import generate_uniform_f32_aligned

            return generate_uniform_f32_aligned(
                self,
                out,
                num_warps=int(kwargs.get("num_warps", 4 if get_vendor_info().vendor_name in ("metax", "hygon") else 8)),
                cached_launch=bool(kwargs.get("cached_launch", True)),
                keyed_launch=bool(kwargs.get("keyed_launch", True)),
                linear_table=bool(kwargs.get("linear_table", get_vendor_info().vendor_name == "metax")),
                cache_tensor_signature=bool(kwargs.get("cache_tensor_signature", True)),
            )
        raw = torch.empty(n * (2 if f64 else 1), dtype=torch.int32, device=out.device)
        self.generate(raw, **kwargs)
        if bool(kwargs.get("cached_launch", True)):
            args, constants = (raw, out, n), (f64, 256)
            key = (
                launch_specialization_key(args, 2, constants, (4,), _uniform_kernel.debug)
                if kwargs.get("keyed_launch", True)
                else None
            )
            _uniform_launcher.launch(((n + 255) // 256,), args, constants, (4,), specialization_key=key)
        else:
            _uniform_kernel[((n + 255) // 256,)](raw, out, n, f64, 256, num_warps=4)
        return out

    def generate_normal(
        self,
        out: torch.Tensor,
        *,
        mean: float = 0.0,
        stddev: float = 1.0,
        lognormal: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        n = out.numel()
        if n == 0:
            return out
        f64 = out.dtype == torch.float64
        pairs = (n + 1) // 2
        words = pairs * (4 if f64 else 2)
        if not kwargs and self._uses_raw_buffer(words):
            from ._mtgp32_buffered import generate_buffered

            return generate_buffered(self, out, "lognormal" if lognormal else "normal", mean=mean, stddev=stddev)
        if (
            n <= (65536 if f64 else 1048576)
            and words % 256 == 0
            and self.offset % 256 == 0
            and getattr(self, "_chunk_cache", None) is None
            and kwargs.get("fused_transform", True)
            and set(kwargs).issubset(
                {"cached_launch", "keyed_launch", "fused_transform", "num_warps", "cache_tensor_signature"}
            )
            and (not f64 or "num_warps" not in kwargs)
            and get_vendor_info().vendor_name == "metax"
        ):
            if f64:
                from ._mtgp32_direct_words import generate_direct_words

                generate_direct_words(
                    self,
                    out,
                    mean=mean,
                    stddev=stddev,
                    lognormal=lognormal,
                    cached_launch=bool(kwargs.get("cached_launch", True)),
                    keyed_launch=bool(kwargs.get("keyed_launch", True)),
                    cache_tensor_signature=bool(kwargs.get("cache_tensor_signature", True)),
                )
                return out
            from ._mtgp32_direct_pairs import generate_direct_pairs

            generate_direct_pairs(
                self,
                out,
                "lognormal" if lognormal else "normal",
                mean=mean,
                stddev=stddev,
                num_warps=int(kwargs.get("num_warps", 2)),
                cached_launch=bool(kwargs.get("cached_launch", True)),
                keyed_launch=bool(kwargs.get("keyed_launch", True)),
                cache_tensor_signature=bool(kwargs.get("cache_tensor_signature", True)),
            )
            return out
        raw = torch.empty(pairs * (4 if f64 else 2), dtype=torch.int32, device=out.device)
        self.generate(raw, **kwargs)
        native_log = f64 and get_vendor_info().vendor_name == "metax"
        finite_exp = native_log and lognormal and exp_finite_range(mean, stddev)
        args = (raw, out, n, _float_bits(mean, f64), _float_bits(stddev, f64))
        grid = ((pairs + 255) // 256,)
        if bool(kwargs.get("cached_launch", True)):
            launch = _normal_launcher(native_log, finite_exp)
            constants, options = (f64, lognormal, 256), (4, False)
            key = (
                launch_specialization_key(args, 2, constants, options, launch._kernel.debug)
                if kwargs.get("keyed_launch", True)
                else None
            )
            launch.launch(grid, args, constants, options, specialization_key=key)
        else:
            build_normal_kernel(native_log, finite_exp)[grid](
                *args,
                f64,
                lognormal,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out

    def generate_lognormal(self, out: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.generate_normal(out, lognormal=True, **kwargs)
