"""Ascend Philox/XORWOW Poisson statistics and exact RNG-state comparisons."""

import numpy as np
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from benchmark.hygon_matrix.reference import words
from benchmark.rng_delivery.cases import validate_output
from benchmark.rng_delivery.config import parameter_values
from flagrand import curand
from flagrand.rng._poisson_table import poisson_alias_table
from flagrand.rng._xorwow.reference import generate_raw, seed_words, signed_rows

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.ascend_910c,
    pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required"),
]


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("philox", marks=pytest.mark.philox_poisson),
        pytest.param("xorwow", marks=pytest.mark.xorwow_poisson),
    ],
)
def test_poisson_replay_and_moments(engine, record_property):
    for lam in parameter_values("poisson"):
        out = torch.empty(65536, dtype=torch.int32, device="npu")
        g = curand.create_generator(engine, seed=20261020)
        curand.generate_poisson(g, out, lam)
        first = out.cpu()
        curand.set_pseudo_random_generator_seed(g, 20261020)
        curand.generate_poisson(g, out, lam)
        result = validate_output(first, out.cpu(), "poisson", lam)
        record_property(str(lam), result)
        assert result["status"] == "pass", result


def _sample_alias(raw, lam):
    cutoff, aliases, first = poisson_alias_table(lam)
    a = np.asarray(raw, dtype=np.uint64)
    column = a[::2] & np.uint64(len(cutoff) - 1)
    return first + np.where(a[1::2] < np.asarray(cutoff, dtype=np.uint64)[column], column, np.asarray(aliases)[column])


@pytest.mark.philox_poisson
@pytest.mark.parametrize("seed", [0, (1 << 63) + 19])
@pytest.mark.parametrize("offset", [1, (1 << 34) - 3])
def test_philox_poisson_integer_reference(seed, offset):
    g = curand.create_generator("philox", seed=seed, offset=offset)
    for n, lam in ((7, 10.0), (257, 1000.0)):
        start = g.engine.offset
        out = torch.empty(n, dtype=torch.int32, device="npu")
        curand.generate_poisson(g, out, lam)
        assert np.array_equal(out.cpu().numpy(), _sample_alias(words(seed, start, 2 * n), lam))
        assert g.engine.offset == start + 2 * n


@pytest.mark.xorwow_poisson
def test_xorwow_poisson_integer_reference_and_state():
    g = curand.create_generator("xorwow", seed=20261108)
    g.engine.nstreams = 512
    state = seed_words(512, 20261108)
    for n, lam in ((1, 0.1), (257, 10.0), (2049, 1000.0)):
        raw, state = generate_raw(state, 2 * n)
        out = torch.empty(n, dtype=torch.int32, device="npu")
        curand.generate_poisson(g, out, lam)
        assert np.array_equal(out.cpu().numpy(), _sample_alias(raw, lam))
        assert torch.equal(g.engine.state.cpu(), torch.tensor(signed_rows(state), dtype=torch.int32))
