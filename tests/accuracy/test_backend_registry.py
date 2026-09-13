from __future__ import annotations

from flagrand import curand
from flagrand.runtime import current_backend, current_vendor, device
from flagrand.runtime.backend import get_backend_state, get_vendor_infos


def test_device_detector_activates_matching_backend_state() -> None:
    state = get_backend_state()
    assert state.vendor_info is device.info
    assert current_vendor() == device.vendor_name
    assert current_backend() == device.info.backend_name
    assert device.name == device.info.device_name


def test_registry_contains_supported_vendor_descriptors() -> None:
    descriptors = {info.vendor_name: info for info in get_vendor_infos()}
    assert set(descriptors) == {"nvidia", "metax", "iluvatar", "ascend", "thead", "hygon"}
    assert descriptors["hygon"].device_query_cmd == "hy-smi"
    assert descriptors["hygon"].portable_f32_math
    assert descriptors["hygon"].cached_binary_launch
    assert descriptors["iluvatar"].device_query_cmd == "ixsmi"
    assert descriptors["ascend"].device_query_cmd == "npu-smi info"
    assert descriptors["thead"].device_query_cmd == "ppu-smi"
    assert descriptors["metax"].device_query_cmd == "mx-smi"


def test_public_curand_uses_selected_vendor_override() -> None:
    expected_module = {
        "nvidia": "flagrand.runtime.backend._nvidia.ops",
        "metax": "flagrand.runtime.backend._metax.ops",
        "iluvatar": "flagrand.runtime.backend._iluvatar.ops",
        "ascend": "flagrand.runtime.backend._ascend.ops",
    }.get(device.vendor_name, "flagrand._curand_philox_fast_path")
    assert curand.try_generate_raw.__module__ == expected_module
    assert curand.try_generate_uniform.__module__ == expected_module
    assert curand.try_generate_normal.__module__ == (
        "flagrand.runtime.backend._hygon.ops" if device.vendor_name == "hygon" else expected_module
    )
    public_module = {
        "hygon": "flagrand.runtime.backend._hygon.ops",
        "nvidia": "flagrand.runtime.backend._nvidia.ops",
        "metax": "flagrand.runtime.backend._metax.ops",
        "ascend": "flagrand.runtime.backend._ascend.ops",
    }.get(device.vendor_name, "flagrand.curand")
    for name in (
        "generate",
        "generate_uniform",
        "generate_uniform_double",
        "generate_normal",
        "generate_normal_double",
        "generate_lognormal",
        "generate_lognormal_double",
    ):
        assert getattr(curand, name).__module__ == public_module
