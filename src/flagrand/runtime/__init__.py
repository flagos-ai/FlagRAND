from flagrand.runtime import backend, common, error
from flagrand.runtime.backend import (
    SpecOpRegistrar,
    activate_backend,
    current_backend,
    current_vendor,
    force_portable_backend,
    supports_cached_binary_launch,
    uses_portable_f32_math,
)
from flagrand.runtime.backend.device_finder import DeviceDetector

device = DeviceDetector()
activate_backend(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.libtuner import HAS_LIBTUNER, libtuner_status, require_libtuner, use_libtuner

__all__ = [
    "CachedKernelLauncher",
    "DeviceDetector",
    "HAS_LIBTUNER",
    "SpecOpRegistrar",
    "backend",
    "common",
    "current_backend",
    "current_vendor",
    "device",
    "error",
    "force_portable_backend",
    "libtuner_status",
    "require_libtuner",
    "supports_cached_binary_launch",
    "torch_device_fn",
    "use_libtuner",
    "uses_portable_f32_math",
]
