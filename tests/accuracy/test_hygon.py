from types import SimpleNamespace

import pytest
import torch

from flagrand import curand
from flagrand.runtime import current_vendor
from flagrand.runtime.backend.device_finder import DeviceDetector


def test_hip_detection_does_not_classify_amd_as_hygon(monkeypatch):
    from triton.runtime.driver import driver

    monkeypatch.setattr(driver.active, "get_current_target", lambda: SimpleNamespace(backend="hip", arch="gfx90a"))
    assert DeviceDetector()._get_vendor_from_lib() is None
    monkeypatch.setattr(driver.active, "get_current_target", lambda: SimpleNamespace(backend="hip", arch="gfx936"))
    assert DeviceDetector()._get_vendor_from_lib() == "hygon"


@pytest.mark.skipif(current_vendor() != "hygon", reason="Hygon binary64 path")
@pytest.mark.parametrize("offset", [0, 1, 3, (1 << 34) + 3])
def test_binary64_parameters_and_exp_keep_double_precision(offset):
    n = 4096
    normal = torch.empty(n, device="cuda", dtype=torch.float64)
    affine = torch.empty_like(normal)
    lognormal = torch.empty_like(normal)
    constant = torch.empty_like(normal)
    seed = (1 << 40) + 31
    mean, stddev = 0.200000000123, 0.700000000321
    for out, op, mu, sigma in [
        (normal, curand.generate_normal_double, 0.0, 1.0),
        (affine, curand.generate_normal_double, mean, stddev),
        (lognormal, curand.generate_lognormal_double, mean, stddev),
        (constant, curand.generate_normal_double, mean, 0.0),
    ]:
        generator = curand.create_generator(seed=seed, offset=offset)
        op(generator, out, mean=mu, stddev=sigma)
        assert generator.offset == offset + 2 * n
    torch.cuda.synchronize()
    torch.testing.assert_close(affine.cpu(), normal.cpu() * stddev + mean, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(lognormal.cpu(), affine.cpu().exp(), rtol=1e-13, atol=1e-13)
    assert torch.equal(constant.cpu(), torch.full((n,), mean, dtype=torch.float64))


def _cpu_words(seed, offset, n):
    result = []
    mask = 0xFFFFFFFF
    for counter in range(offset // 4, (offset + n + 3) // 4):
        a, b, c, d = 0, 0, counter & mask, counter >> 32
        k0, k1 = seed & mask, seed >> 32
        for _ in range(10):
            p = a * 0xD2511F53
            q = c * 0xCD9E8D57
            a, b, c, d = ((q >> 32) ^ b ^ k0) & mask, q & mask, ((p >> 32) ^ d ^ k1) & mask, p & mask
            k0 = (k0 + 0x9E3779B9) & mask
            k1 = (k1 + 0xBB67AE85) & mask
        result.extend([a, b, c, d])
    return result[offset % 4 : offset % 4 + n]


@pytest.mark.skipif(current_vendor() != "hygon", reason="Hygon counter-width regression")
@pytest.mark.parametrize("offset", [(1 << 33) - 4, (1 << 34) - 4])
def test_raw_and_uniform_counters_cross_32_bit_boundaries(offset):
    seed = (1 << 40) + 31
    n = 64
    words = _cpu_words(seed, offset, n)
    expected = torch.tensor([x if x < 2**31 else x - 2**32 for x in words], dtype=torch.int32)
    out = torch.empty(n, device="cuda", dtype=torch.int32)
    curand.generate(curand.create_generator(seed=seed, offset=offset), out)
    assert torch.equal(out.cpu(), expected)
    uniform = torch.empty(n, device="cuda", dtype=torch.float32)
    curand.generate_uniform(curand.create_generator(seed=seed, offset=offset), uniform)
    expected_u = torch.tensor(words, dtype=torch.float32) * 2.0**-32 + 2.0**-33
    assert torch.equal(uniform.cpu(), expected_u)
