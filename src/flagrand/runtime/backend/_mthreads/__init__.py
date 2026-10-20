from __future__ import annotations

from flagrand.runtime.backend.backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="mthreads",
    device_name="musa",
    device_query_cmd="mthreads-gmi",
    triton_extra_name="musa",
    trademark="Moore Threads MUSA",
    backend_name="musa",
    cached_binary_launch=False,
    portable_f32_math=True,
)

CUSTOMIZED_UNUSED_OPS: tuple[str, ...] = ()

__all__ = ["CUSTOMIZED_UNUSED_OPS", "vendor_info"]
