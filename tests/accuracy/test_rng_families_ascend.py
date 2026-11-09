"""Ascend family contracts and integer-reference state comparisons."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from benchmark.rng_delivery.cases import DTYPES, invoke, validate_output
from benchmark.rng_delivery.config import OPERATORS, parameter_values, split_operator
from flagrand import curand
from flagrand.rng._mrg32k3a_math import initial_states, reference_step

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.ascend_910c,
    pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required"),
]
CASES = [
    pytest.param(op, marks=getattr(pytest.mark, op))
    for op in OPERATORS
    if op not in ("philox_poisson", "xorwow_poisson") and not op.startswith("mtgp32_")
]


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


@pytest.mark.mrg32k3a_raw32
@pytest.mark.parametrize("streams", [16, 257])
@pytest.mark.parametrize("offset", [0, 3])
def test_mrg_integer_reference_and_state(streams, offset):
    from flagrand.rng.mrg32k3a import Mrg32k3aGenerator

    seed, mask = 1234, (1 << 32) - 1
    g = Mrg32k3aGenerator(seed=seed, nstreams=streams, offset=offset)
    rows = initial_states(seed, streams, offset)
    states = [tuple(rows[j][i] & mask for j in range(6)) for i in range(streams)]
    cursor = offset % streams
    for n in (1, 7, streams + 3):
        out = torch.empty(n, dtype=torch.int32, device="npu")
        g.generate(out)
        expected = []
        for i in range(n):
            s = (cursor + i) % streams
            value, states[s] = reference_step(states[s])
            expected.append(value)
        cursor = (cursor + n) % streams
        assert [x & mask for x in out.cpu().tolist()] == expected
        actual = g._state.cpu().tolist()
        assert [tuple(actual[j][i] & mask for j in range(6)) for i in range(streams)] == states
        assert g._cursor == cursor


@pytest.mark.parametrize("operator", [case for case in CASES if case.values[0].startswith("mrg32k3a_")])
def test_mrg_split_continuation(operator):
    engine, api = split_operator(operator)
    whole = curand.create_generator(engine, seed=4321)
    split = curand.create_generator(engine, seed=4321)
    output = torch.empty(1003, dtype=DTYPES[api], device="npu")
    invoke(whole, output, api, 10.0)
    parts = []
    for n in (1, 2, 257, 743):
        part = torch.empty(n, dtype=DTYPES[api], device="npu")
        invoke(split, part, api, 10.0)
        parts.append(part.cpu())
    assert torch.equal(output.cpu(), torch.cat(parts))
    assert torch.equal(whole.engine._state.cpu(), split.engine._state.cpu())
    assert whole.engine.offset == split.engine.offset
    assert whole.engine._cursor == split.engine._cursor


@pytest.mark.parametrize(
    "lognormal",
    [
        pytest.param(False, marks=pytest.mark.mrg32k3a_normal_f64, id="normal"),
        pytest.param(True, marks=pytest.mark.mrg32k3a_lognormal_f64, id="lognormal"),
    ],
)
def test_mrg_normal64_lookahead_window_continuation(lognormal):
    from flagrand.rng.mrg32k3a import Mrg32k3aGenerator

    whole = curand.CurandGenerator(
        curand.CURAND_RNG_PSEUDO_MRG32K3A, Mrg32k3aGenerator(seed=4321, offset=29, nstreams=32)
    )
    split = curand.CurandGenerator(
        curand.CURAND_RNG_PSEUDO_MRG32K3A, Mrg32k3aGenerator(seed=4321, offset=29, nstreams=32)
    )
    generate = curand.generate_lognormal_double if lognormal else curand.generate_normal_double
    output = torch.empty(72, dtype=torch.float64, device="npu")
    generate(whole, output)
    parts = []
    for index in range(9):
        part = torch.empty(8, dtype=torch.float64, device="npu")
        generate(split, part)
        parts.append(part.cpu())
        if index == 0:
            first_window = split.engine._ascend_normal64_window
    assert first_window.calls == first_window.position == 8
    assert split.engine._ascend_normal64_window is not first_window
    assert torch.equal(output.cpu().view(torch.int32), torch.cat(parts).view(torch.int32))
    assert torch.equal(whole.engine._state.cpu(), split.engine._state.cpu())
    assert whole.engine.offset == split.engine.offset
    assert whole.engine._cursor == split.engine._cursor
