"""CoreX carry, scalar-specialization and stream regression cases."""

import pytest
import torch

from benchmark.corex_matrix.cases import DTYPES, invoke_flagrand, make_flagrand_generator
from benchmark.corex_matrix.reference import reference
from flagrand.runtime import current_vendor

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.cuda,
    pytest.mark.skipif(current_vendor() != "iluvatar", reason="CoreX-specific mapping"),
]


@pytest.mark.parametrize(
    "api",
    [
        pytest.param("raw32", marks=pytest.mark.raw32),
        pytest.param("uniform_f32", marks=pytest.mark.uniform_f32),
        pytest.param("uniform_f64", marks=pytest.mark.uniform_f64),
        pytest.param("normal_f32", marks=pytest.mark.normal_f32),
        pytest.param("lognormal_f32", marks=pytest.mark.lognormal_f32),
    ],
)
@pytest.mark.parametrize("offset", [0, 4, 2**33 - 4, 2**34 - 4])
@pytest.mark.parametrize("aligned", [True, False])
def test_cpu_oracle_across_carry_and_current_stream(api, offset, aligned):
    n = 1024 if aligned else 1026 if api.endswith("normal_f32") else 1025
    seed = 0x1234567887654321
    out = torch.empty(n + 1, device="cuda", dtype=DTYPES[api])[1:]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        invoke_flagrand(api, make_flagrand_generator(seed, offset=offset), out)
    stream.synchronize()
    expected = torch.from_numpy(reference(api, seed, offset, n))
    exact = api in ("raw32", "uniform_f64")
    torch.testing.assert_close(out.cpu(), expected, rtol=0 if exact else 2e-5, atol=0 if exact else 2e-6)


@pytest.mark.raw32
def test_cached_scalar_key_tracks_corex_divisibility_by_four():
    # CoreX specializes %4, whereas several other Triton backends specialize %16.
    for offset in (16, 20, 24, 28, 64, 68, 4, 0, 20, 16):
        out = torch.empty(1025, device="cuda", dtype=torch.int32)
        invoke_flagrand("raw32", make_flagrand_generator(17, offset=offset), out)
        expected = torch.from_numpy(reference("raw32", 17, offset, out.numel()))
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
