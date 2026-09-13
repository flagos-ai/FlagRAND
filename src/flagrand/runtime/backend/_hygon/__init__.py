from __future__ import annotations

from flagrand.runtime.backend.backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="hygon",
    device_name="cuda",
    device_query_cmd="hy-smi",
    triton_extra_name="hip",
    trademark="Hygon DCU",
    backend_name="hip",
    cached_binary_launch=True,
    portable_f32_math=True,
)

CUSTOMIZED_UNUSED_OPS: tuple[str, ...] = ()

__all__ = ["CUSTOMIZED_UNUSED_OPS", "vendor_info"]
