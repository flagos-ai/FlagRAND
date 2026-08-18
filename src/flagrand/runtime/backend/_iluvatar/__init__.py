from __future__ import annotations

from flagrand.runtime.backend.backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="iluvatar",
    device_name="cuda",
    device_query_cmd="ixsmi",
    triton_extra_name="corex",
    trademark="Iluvatar CoreX",
    fp64_enabled=True,
    backend_name="corex",
    cached_binary_launch=False,
    portable_f32_math=True,
)

CUSTOMIZED_UNUSED_OPS: tuple[str, ...] = ()

__all__ = ["CUSTOMIZED_UNUSED_OPS", "vendor_info"]
