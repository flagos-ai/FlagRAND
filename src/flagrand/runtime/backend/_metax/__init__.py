from __future__ import annotations

from flagrand.runtime.backend.backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="metax",
    device_name="cuda",
    device_query_cmd="mx-smi",
    trademark="MetaX MACA",
    backend_name="maca",
    cached_binary_launch=False,
    portable_f32_math=True,
)

CUSTOMIZED_UNUSED_OPS: tuple[str, ...] = ()

__all__ = ["CUSTOMIZED_UNUSED_OPS", "vendor_info"]
