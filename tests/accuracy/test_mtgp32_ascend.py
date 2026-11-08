"""Ascend MTGP32 distribution, replay and integer-reference comparisons."""

import numpy as np
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from benchmark.rng_delivery.cases import DTYPES, invoke, validate_output
from benchmark.rng_delivery.config import OPERATORS, parameter_values, split_operator
from flagrand import curand
from flagrand.rng._mtgp32_reference import generate_reference

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.ascend_910c,
    pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required"),
]
CASES = [pytest.param(op, marks=getattr(pytest.mark, op)) for op in OPERATORS if op.startswith("mtgp32_")]


@pytest.mark.parametrize("operator", CASES)
def test_distribution_replay_and_moments(operator, record_property):
    engine, api = split_operator(operator)
    for lam in parameter_values(api):
        out = torch.empty(65536, dtype=DTYPES[api], device="npu")
        g = curand.create_generator(engine, seed=20261020)
        invoke(g, out, api, lam)
        first = out.cpu()
        curand.set_pseudo_random_generator_seed(g, 20261020)
        invoke(g, out, api, lam)
        result = validate_output(first, out.cpu(), api, lam)
        record_property(str(lam), result)
        assert result["status"] == "pass", result


@pytest.mark.mtgp32_raw32
def test_mtgp_integer_reference_and_split_state():
    seed, n = 20261108, 2 * 192 * 256
    reference = np.asarray([generate_reference(seed, 512, i) for i in range(192)], dtype=np.uint32)
    expected = reference.reshape(192, 2, 256).transpose(1, 0, 2).reshape(-1)
    whole = curand.create_generator("mtgp32", seed=seed)
    out = torch.empty(n, dtype=torch.int32, device="npu")
    curand.generate(whole, out)
    assert np.array_equal(out.cpu().numpy().view(np.uint32), expected)
    split = curand.create_generator("mtgp32", seed=seed)
    parts = []
    for count in (1, 255, 3, 50000, n - 50259):
        part = torch.empty(count, dtype=torch.int32, device="npu")
        curand.generate(split, part)
        parts.append(part.cpu())
    assert torch.equal(out.cpu(), torch.cat(parts))
    assert whole.engine.offset == split.engine.offset
    # Prefetch can advance physical state beyond the public offset. Consume
    # both streams to the same frontier without starting another prefetch.
    frontier = max(whole.engine._ws_next_block_start, split.engine._ws_next_block_start)
    tails = []
    for generator in (whole, split):
        generator.engine._use_buffered = False
        tail = torch.empty(frontier - generator.engine.offset, dtype=torch.int32, device="npu")
        curand.generate(generator, tail)
        tails.append(tail.cpu())
    assert torch.equal(*tails)
    assert whole.engine.offset == split.engine.offset == frontier
    assert torch.equal(whole.engine.state.cpu(), split.engine.state.cpu())
