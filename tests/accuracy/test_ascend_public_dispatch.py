from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flagrand import curand
from flagrand.runtime.backend._ascend import ops as ascend_ops

pytestmark = [pytest.mark.accuracy, pytest.mark.runtime]


class _FakeTensor:
    def __init__(self, dtype: torch.dtype, count: int) -> None:
        self.dtype = dtype
        self.device = SimpleNamespace(type="npu")
        self._count = count
        self.numel_calls = 0

    def numel(self) -> int:
        self.numel_calls += 1
        return self._count

    def is_contiguous(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("operation", "dtype", "normal_kwargs", "kind", "flag"),
    (
        (ascend_ops.generate, torch.int32, {}, "raw_uniform", False),
        (ascend_ops.generate_uniform, torch.float32, {}, "raw_uniform", True),
        (ascend_ops.generate_normal, torch.float32, {"mean": 1.5, "stddev": 0.5}, "normal", False),
        (ascend_ops.generate_lognormal, torch.float32, {"mean": 1.5, "stddev": 0.5}, "normal", True),
    ),
)
def test_ascend_f32_public_dispatch_reaches_prepared_launcher_once(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    dtype: torch.dtype,
    normal_kwargs: dict[str, float],
    kind: str,
    flag: bool,
) -> None:
    calls: list[tuple[Any, ...]] = []

    def launch_raw_uniform(
        out: object,
        engine: Any,
        count: int,
        *,
        uniform: bool,
        num_warps: int,
    ) -> object:
        calls.append(("raw_uniform", out, engine, count, uniform, num_warps))
        engine.offset += count
        return out

    def launch_normal(
        out: object,
        engine: Any,
        count: int,
        *,
        mean: float,
        stddev: float,
        lognormal: bool,
        num_warps: int,
    ) -> object:
        calls.append(("normal", out, engine, count, mean, stddev, lognormal, num_warps))
        engine.offset += count
        return out

    monkeypatch.setattr(ascend_ops, "_generate_huawei910c_raw_uniform_f32_prepared", launch_raw_uniform)
    monkeypatch.setattr(ascend_ops, "_generate_huawei910c_normal_f32_prepared", launch_normal)

    generator = curand.create_generator(seed=17, offset=9)
    output = _FakeTensor(dtype, 23)
    result = operation(
        generator,
        output,
        block_size=64,
        num_warps=4,
        **normal_kwargs,
    )

    assert result is output
    assert output.numel_calls == 1
    assert len(calls) == 1
    assert calls[0][0] == kind
    assert calls[0][1] is output
    assert calls[0][2] is generator.engine
    assert calls[0][3] == 23
    assert calls[0][-2] == flag
    assert calls[0][-1] == 4
    assert generator.offset == 32


def test_ascend_public_dispatch_preserves_validation_and_empty_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = curand.create_generator(seed=3, offset=11)

    with pytest.raises(TypeError, match="generate_uniform"):
        ascend_ops.generate_uniform(generator, _FakeTensor(torch.float64, 1))

    with pytest.raises(ValueError, match="block_size"):
        ascend_ops.generate_uniform(generator, _FakeTensor(torch.float32, 1), block_size=0)

    with pytest.raises(ValueError, match="contiguous CUDA/NPU output"):
        ascend_ops.generate_uniform(generator, _FakeTensor(torch.float32, 1), unsupported=True)

    monkeypatch.setattr(
        ascend_ops,
        "_generate_huawei910c_raw_uniform_f32_prepared",
        lambda *args, **kwargs: pytest.fail("empty output must not launch"),
    )
    output = _FakeTensor(torch.float32, 0)
    assert ascend_ops.generate_uniform(generator, output) is output
    assert generator.offset == 11
