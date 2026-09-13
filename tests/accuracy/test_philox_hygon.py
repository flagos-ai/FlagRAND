"""Portable-sequence correctness and native hipRAND distribution/stream tests."""

import pytest
import torch

from benchmark.hygon_matrix.baseline import HiprandGenerator
from benchmark.hygon_matrix.cases import validate_reference
from benchmark.philox_common import DTYPES, invoke_flagrand, make_flagrand_generator, validate_output
from flagrand.runtime import current_vendor

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.hygon,
    pytest.mark.skipif(current_vendor() != "hygon", reason="Hygon DTK hardware test"),
]
DISTRIBUTIONS = [
    pytest.param("raw32", marks=pytest.mark.raw32),
    pytest.param("uniform_f32", marks=pytest.mark.uniform_f32),
    pytest.param("uniform_f64", marks=pytest.mark.uniform_f64),
    pytest.param("normal_f32", marks=pytest.mark.normal_f32),
    pytest.param("normal_f64", marks=pytest.mark.normal_f64),
    pytest.param("lognormal_f32", marks=pytest.mark.lognormal_f32),
    pytest.param("lognormal_f64", marks=pytest.mark.lognormal_f64),
]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("offset", [0, 1, 3, (1 << 34) - 4])
def test_cpu_philox_reference_across_offsets(distribution, offset):
    seed = (1 << 40) + 31
    storage = torch.empty(131, device="cuda", dtype=DTYPES[distribution])
    out = storage[1:]
    invoke_flagrand(distribution, make_flagrand_generator(seed, offset=offset), out)
    assert validate_reference(out, distribution, seed, offset)["status"] == "pass"


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_native_hiprand_replay_statistics_and_current_stream(distribution):
    out = torch.empty(262144, device="cuda", dtype=DTYPES[distribution])
    replay = torch.empty_like(out)
    with HiprandGenerator(seed=73) as gen:
        gen.generate(distribution, out)
        torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), HiprandGenerator(seed=73) as gen:
        gen.generate(distribution, replay)
        stream.synchronize()
    assert validate_output(out, replay, distribution)["status"] == "pass"
