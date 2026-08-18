from __future__ import annotations

from flagrand.runtime.backend.backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="nvidia",
    device_name="cuda",
    device_query_cmd="nvidia-smi",
)

ARCH_MAP = {
    "9": "hopper",
    "8": "ampere",
}

CUSTOMIZED_UNUSED_OPS: tuple[str, ...] = ()

__all__ = ["ARCH_MAP", "CUSTOMIZED_UNUSED_OPS", "vendor_info"]
