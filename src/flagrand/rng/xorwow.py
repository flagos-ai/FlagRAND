"""Stateful XORWOW integration using the existing multi-backend implementation.

The imported kernels keep six uint32 words per stream and fuse uniform,
normal and lognormal transforms. They use the project's SplitMix64 seeds,
not cuRAND's initialization or sequence-jump scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from ._xorwow.api import create_state
from ._xorwow.api import generate as generate_state
from ._xorwow.api import prepare_standard_launch, resolve_backend
from ._xorwow.profiles import launch_profile
from ._xorwow.two_stage import generate_two_stage, supports_two_stage


@lru_cache(maxsize=128)
def _launch_context(requested_backend, device, n, distribution, dtype):
    """Resolve stable device and shape metadata outside the repeated launch."""
    backend = resolve_backend(requested_backend, device)
    config = {} if backend == "cuda" else launch_profile(backend, n, distribution, dtype)["config"]
    return backend, config


@lru_cache(maxsize=128)
def _prepared_launch_context(device, n, nstreams, distribution, dtype, mean, mean_sign, stddev, stddev_sign):
    # Signed-zero tokens prevent numeric equality from merging distinct IEEE
    # parameters; float bits are constructed only when preparing a new plan.
    backend, config = _launch_context("auto", device, n, distribution, dtype)
    if backend != "thead" or supports_two_stage(backend, n, nstreams, distribution, dtype, config):
        return None
    return prepare_standard_launch(n, nstreams, distribution, dtype, mean, stddev, backend, **config)


@dataclass
class XorwowGenerator:
    """A fixed set of persistent XORWOW streams.

    Stream ``s`` writes output positions ``s + k * nstreams``. Each call
    starts that interleaving at stream zero. Changing block size or warps
    preserves it; changing nstreams defines a different sequence.

    ``offset`` at construction/reset discards that many interleaved raw32
    words before generation. During generation it counts consumed raw words.
    The six-word device state, rather than this scalar counter, determines
    continuation after arbitrary shapes or mixed distributions. In particular,
    restoring seed+offset does not reconstruct arbitrary earlier call patterns.
    Graph replay advances device state without updating the Python counter.

    Concatenating calls matches a single call when intermediate sizes are
    multiples of nstreams for raw/uniform, or 2*nstreams for normal/lognormal.
    A final unpaired normal output consumes a full Box--Muller pair.
    """

    seed: int = 0
    offset: int = 0
    nstreams: int = 32768
    _state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _state_seed: int | None = field(default=None, init=False, repr=False)
    _state_offset: int | None = field(default=None, init=False, repr=False)
    _distribution_workspaces: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def dimensions(self) -> None:
        return None

    @property
    def state(self) -> torch.Tensor | None:
        return self._state

    def reset(self, *, seed: int | None = None, offset: int = 0) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.offset = int(offset)
        self._state = None
        self._state_seed = None
        self._state_offset = None
        self._distribution_workspaces.clear()

    def _distribution_workspace(self, n, device):
        # Each stream retains its own maximum-sized buffer and the stream object.
        # Reusing RNG state across streams still requires the caller to order them.
        stream = torch.cuda.current_stream(device)
        key = (device, stream.cuda_stream)
        entry = self._distribution_workspaces.get(key)
        if entry is None or entry[1].numel() < 2 * n:
            workspace = torch.empty(2 * n, dtype=torch.int32, device=device)
            workspace.record_stream(stream)
            entry = (stream, workspace)
            self._distribution_workspaces[key] = entry
        return entry[1]

    def ensure_state(self, device, *, backend: str = "auto") -> torch.Tensor:
        if self._state is None or self.seed != self._state_seed or self.offset != self._state_offset:
            self._state = create_state(self.nstreams, self.seed, device=device)
            remaining = int(self.offset)
            # Whole stream rows preserve the original interleaving while
            # avoiding a scratch allocation proportional to a large offset.
            chunk_size = 32 * self.nstreams
            while remaining > 0:
                count = min(remaining, chunk_size)
                scratch = torch.empty(count, dtype=torch.int32, device=device)
                generate_state(self._state, count, distribution="raw32", out=scratch, backend=backend)
                remaining -= count
            self._state_seed = self.seed
            self._state_offset = self.offset
        elif self._state.device != torch.device(device):
            self._state = self._state.to(device=device)
        return self._state

    def generate(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.generate_distribution(out, "raw32", seed=seed, offset=offset, **kwargs)

    def generate_uniform(self, out: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.generate_distribution(out, "uniform", **kwargs)

    def generate_normal(
        self,
        out: torch.Tensor,
        *,
        mean: float = 0.0,
        stddev: float = 1.0,
        lognormal: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.generate_distribution(
            out, "lognormal" if lognormal else "normal", mean=mean, stddev=stddev, **kwargs
        )

    def generate_lognormal(
        self,
        out: torch.Tensor,
        *,
        mean: float = 0.0,
        stddev: float = 1.0,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.generate_distribution(out, "lognormal", mean=mean, stddev=stddev, **kwargs)

    def generate_distribution(
        self,
        out: torch.Tensor,
        distribution: str,
        *,
        mean: float = 0.0,
        stddev: float = 1.0,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if seed is not None or offset is not None:
            temporary = XorwowGenerator(
                seed=self.seed if seed is None else int(seed),
                offset=self.offset if offset is None else int(offset),
                nstreams=self.nstreams,
            )
            return temporary.generate_distribution(out, distribution, mean=mean, stddev=stddev, **kwargs)

        n = out.numel()
        if n == 0:
            return out
        prepared = None
        if not kwargs:
            prepared = _prepared_launch_context(
                out.device,
                n,
                self.nstreams,
                distribution,
                out.dtype,
                mean,
                math.copysign(1.0, mean),
                stddev,
                math.copysign(1.0, stddev),
            )
        if prepared is not None:
            prepared.launch(self.ensure_state(out.device, backend="thead"), out)
        else:
            options = dict(kwargs)
            backend, base_config = _launch_context(
                str(options.pop("backend", "auto")), out.device, n, distribution, out.dtype
            )
            state = self.ensure_state(out.device, backend=backend)
            config = dict(base_config)
            if "block_size" in options:
                options["block"] = options.pop("block_size")
            config.update(options)
            if not options and supports_two_stage(backend, n, self.nstreams, distribution, out.dtype, config):
                workspace = self._distribution_workspace(n, state.device)
                generate_two_stage(state, out, workspace, distribution, mean, stddev, backend=backend)
            else:
                generate_state(
                    state,
                    n,
                    distribution=distribution,
                    dtype=out.dtype,
                    mu=mean,
                    sigma=stddev,
                    out=out,
                    backend=backend,
                    **config,
                )
        if distribution in ("normal", "lognormal"):
            rows, tail = divmod(n, 2 * self.nstreams)
            pairs = rows * self.nstreams + min(tail, self.nstreams)
            consumed = pairs * (4 if out.dtype == torch.float64 else 2)
        else:
            consumed = n * (2 if distribution == "uniform" and out.dtype == torch.float64 else 1)
        self.offset += consumed
        self._state_offset = self.offset
        return out
